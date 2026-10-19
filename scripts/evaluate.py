"""Recompute a saved checkpoint's metrics without training."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from coupledcf.data import DataBundle, sha256
from coupledcf.engine import evaluate, set_seed
from coupledcf.models import ModelConfig, build_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--split', type=Path, default=Path('data/processed/split.npz'))
    args = parser.parse_args()
    result = json.loads((args.run / 'result.json').read_text())
    if sha256(args.split) != result['split_sha256']:
        raise ValueError('Split hash differs from the trained experiment')
    if sha256(args.run / 'best.pt') != result['checkpoint_sha256']:
        raise ValueError('Checkpoint hash differs from the experiment result')
    data = DataBundle.load(args.split)
    set_seed(result['seed'], 2)
    model = build_model(data, ModelConfig(**result['config']['model']))
    checkpoint = torch.load(args.run / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    metrics, ranks = evaluate(model, data.test, 'cpu')
    if metrics != result['test']:
        raise AssertionError(f'Recomputed metrics differ: {metrics} != {result["test"]}')
    print(json.dumps({'status': 'PASS', 'recomputed_test': metrics}, indent=2))
