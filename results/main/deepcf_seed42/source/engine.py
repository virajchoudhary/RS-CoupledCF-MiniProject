"""Matched training runs with validation selection and auditable result bundles."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import platform
import random
import shutil
import threading
import time

import numpy as np
import psutil
import torch

from .data import DataBundle, sha256, training_samples
from .metrics import positive_ranks, ranking_metrics
from .models import ModelConfig, build_model


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 4096
    learning_rate: float = .001
    negatives: int = 4
    threads: int = 2
    device: str = 'cpu'

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.negatives, self.threads) < 1 or self.batch_size < 2:
            raise ValueError('Training counts must be positive and batch size at least two')
        if self.learning_rate <= 0:
            raise ValueError('Learning rate must be positive')
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('Device must be cpu or cuda')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but this PyTorch installation has no available CUDA device')


def set_seed(seed: int, threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class ResourceMonitor:
    """Sample process RSS, while keeping wall and CPU time separate."""
    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.wait(.25):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    def __enter__(self):
        self.wall = time.perf_counter()
        self.cpu = time.process_time()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        self.wall_seconds = time.perf_counter() - self.wall
        self.cpu_seconds = time.process_time() - self.cpu


@torch.inference_mode()
def evaluate(model, candidates: np.ndarray, device: str, batch_size: int = 16384):
    model.eval()
    users = np.repeat(np.arange(len(candidates), dtype=np.int64), candidates.shape[1])
    items = candidates.reshape(-1)
    output = []
    for start in range(0, len(items), batch_size):
        u = torch.from_numpy(users[start:start + batch_size]).to(device)
        i = torch.from_numpy(items[start:start + batch_size]).to(device)
        output.append(model(u, i).cpu().numpy())
    scores = np.concatenate(output).reshape(candidates.shape)
    ranks = positive_ranks(scores, candidates)
    return ranking_metrics(ranks), ranks


def run_experiment(data: DataBundle, split_path: Path, output: Path,
                   model_config: ModelConfig, train_config: TrainConfig, seed: int):
    if (output / 'result.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    set_seed(seed, train_config.threads)
    config = {'model': asdict(model_config), 'training': asdict(train_config), 'seed': seed}
    write_json(output / 'config.json', config)
    source_dir = Path(__file__).resolve().parent
    source_hashes = {p.name: sha256(p) for p in sorted(source_dir.glob('*.py'))}
    write_json(output / 'source_hashes.json', source_hashes)
    (output / 'source').mkdir(exist_ok=True)
    for source in source_dir.glob('*.py'):
        shutil.copy2(source, output / 'source' / source.name)
    split_hash = sha256(split_path)
    model = build_model(data, model_config).to(train_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
    loss_function = torch.nn.BCEWithLogitsLoss()
    history = []
    best_ndcg = -1.
    best_epoch = 0
    if train_config.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    try:
        with ResourceMonitor() as resource:
            for epoch in range(1, train_config.epochs + 1):
                epoch_start = time.perf_counter()
                u, i, y = training_samples(data, seed * 10000 + epoch, train_config.negatives)
                u, i, y = torch.from_numpy(u), torch.from_numpy(i), torch.from_numpy(y)
                model.train()
                loss_sum, count, batches = 0., 0, 0
                for start in range(0, len(y), train_config.batch_size):
                    end = min(start + train_config.batch_size, len(y))
                    if end - start == 1:
                        raise ValueError('A singleton training batch cannot use batch normalization')
                    users = u[start:end].to(train_config.device)
                    items = i[start:end].to(train_config.device)
                    labels = y[start:end].to(train_config.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(users, items)
                    loss = loss_function(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(labels)
                    count += len(labels)
                    batches += 1
                if count != len(data.train_users) * (train_config.negatives + 1):
                    raise AssertionError('Training skipped examples')
                validation, _ = evaluate(model, data.validation, train_config.device)
                row = {'epoch': epoch, 'train_bce': loss_sum / count,
                       'examples': count, 'batches': batches,
                       'seconds': time.perf_counter() - epoch_start, 'validation': validation}
                history.append(row)
                write_json(output / 'history.json', history)
                print(json.dumps({'run': output.name, **row}), flush=True)
                if validation['NDCG@10'] > best_ndcg:
                    best_ndcg, best_epoch = validation['NDCG@10'], epoch
                    torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'config': config, 'epoch': epoch, 'validation': validation,
                                'split_sha256': split_hash, 'source_hashes': source_hashes},
                               output / 'best.pt')
            checkpoint = torch.load(output / 'best.pt', map_location=train_config.device, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            validation, validation_ranks = evaluate(model, data.validation, train_config.device)
            test, test_ranks = evaluate(model, data.test, train_config.device)
            np.savez_compressed(output / 'ranks.npz', validation=validation_ranks, test=test_ranks)
        result = {'status': 'COMPLETE', 'variant': model_config.variant, 'seed': seed,
                  'selected_epoch': best_epoch, 'completed_epochs': len(history),
                  'validation': validation, 'test': test,
                  'parameters': sum(p.numel() for p in model.parameters()),
                  'wall_seconds': resource.wall_seconds, 'cpu_seconds': resource.cpu_seconds,
                  'sampled_peak_rss_mb': resource.peak_rss / 1024 ** 2,
                  'cuda_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024 ** 2
                      if train_config.device == 'cuda' else None,
                  'split_sha256': split_hash, 'checkpoint_sha256': sha256(output / 'best.pt'),
                  'config_sha256': sha256(output / 'config.json'),
                  'software': {'python': platform.python_version(), 'torch': torch.__version__,
                               'numpy': np.__version__, 'os': platform.platform(),
                               'processor': platform.processor(), 'logical_cpus': os.cpu_count()},
                  'config': config}
        write_json(output / 'result.json', result)
        return result
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'FAILED', 'error': str(error), 'config': config})
        raise
