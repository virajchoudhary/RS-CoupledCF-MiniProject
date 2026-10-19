from pathlib import Path
import json
import numpy as np
import pytest
import torch
from coupledcf.data import DataBundle, sha256
from coupledcf.engine import TrainConfig, evaluate, run_experiment
from coupledcf.models import ModelConfig, build_model


def test_finished_training_checkpoint_can_reproduce_test_metrics(tmp_path):
    observed = np.zeros((4, 12), dtype=bool)
    for u in range(4):
        observed[u, [u, u + 1, u + 2]] = True
    validation, test = [], []
    for u in range(4):
        unseen = np.flatnonzero(~observed[u])
        validation.append([u + 1, *unseen[:3]])
        test.append([u + 2, *unseen[3:6]])
    data = DataBundle(np.arange(4), np.arange(4), np.array(validation), np.array(test),
                      np.zeros((4, 4), dtype=np.int64), np.eye(12, 18, dtype=np.float32),
                      observed, np.arange(4), np.arange(12))
    split = tmp_path / 'split.npz'
    data.save(split)
    run = tmp_path / 'run'
    config = ModelConfig(variant='coupled')
    result = run_experiment(data, split, run, config,
                            TrainConfig(epochs=2, batch_size=20, threads=1), seed=12)
    assert result['status'] == 'COMPLETE'
    assert result['completed_epochs'] == 2
    assert result['checkpoint_sha256'] == sha256(run / 'best.pt')
    history = json.loads((run / 'history.json').read_text())
    assert all(row['examples'] == 20 for row in history)
    assert result['validation']['NDCG@10'] == max(row['validation']['NDCG@10'] for row in history)
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    model = build_model(data, config)
    model.load_state_dict(checkpoint['state_dict'])
    metrics, ranks = evaluate(model, data.test, 'cpu')
    assert metrics == result['test']
    with np.load(run / 'ranks.npz') as z:
        np.testing.assert_array_equal(ranks, z['test'])
    with pytest.raises(FileExistsError, match='overwrite'):
        run_experiment(data, split, run, config, TrainConfig(), seed=12)
    data.train_items[0] = data.test[0, 0]
    with pytest.raises(ValueError, match='leaked'):
        run_experiment(data, split, tmp_path / 'invalid', config, TrainConfig(), seed=12)


@pytest.mark.parametrize('kwargs', [{'epochs': 0}, {'batch_size': 1}, {'device': 'typo'}])
def test_invalid_training_config_fails_before_training(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_unavailable_cuda_fails_instead_of_switching_to_cpu():
    if torch.cuda.is_available():
        pytest.skip('This guard targets a runtime without CUDA')
    with pytest.raises(RuntimeError, match='CUDA requested'):
        TrainConfig(device='cuda')
