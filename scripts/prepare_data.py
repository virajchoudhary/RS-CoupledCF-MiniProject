from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coupledcf.data import prepare

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    manifest = prepare(Path('data/raw/ml-1m'), Path('data/processed'), args.seed)
    print({k: manifest[k] for k in ['ratings', 'users', 'items', 'train_positives', 'split_sha256', 'leakage_checks']})
