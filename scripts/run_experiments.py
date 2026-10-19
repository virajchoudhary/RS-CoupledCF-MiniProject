"""Run each configured model and seed on the exact same data bundle."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from coupledcf.data import DataBundle, sha256
from coupledcf.engine import TrainConfig, run_experiment
from coupledcf.models import ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/main.json')
    parser.add_argument('--seed', type=int, help='Run one configured seed in its own output directories')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    split_path = Path(config['split_path'])
    data = DataBundle.load(split_path)
    print(data.validate(), flush=True)
    training = TrainConfig(**config['training'])
    if args.seed is not None and args.seed not in config['seeds']:
        raise ValueError('Requested seed is not part of the frozen study configuration')
    seeds = [args.seed] if args.seed is not None else config['seeds']
    for seed in seeds:
        for variant in config['variants']:
            output = Path(config['output_dir']) / f'{variant}_seed{seed}'
            requested_model = ModelConfig(variant=variant, **config['model'])
            if (output / 'result.json').exists():
                saved = json.loads((output / 'result.json').read_text())
                requested = {'model': vars(requested_model), 'training': vars(training), 'seed': seed}
                if saved['status'] != 'COMPLETE' or saved['config'] != requested or saved['split_sha256'] != sha256(split_path):
                    raise ValueError(f'Existing run does not match the requested experiment: {output}')
                checkpoint = output / 'best.pt'
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f'Completed metrics have no retained checkpoint: {output}. '
                        'For an extracted submission archive, move results/main to '
                        'results/reference_main before retraining; see README.md.')
                if sha256(checkpoint) != saved['checkpoint_sha256']:
                    raise ValueError(f'Existing checkpoint hash differs from the completed run: {checkpoint}')
                print(f'Preserving completed run {output}', flush=True)
                continue
            run_experiment(data, split_path, output, requested_model, training, seed)


if __name__ == '__main__':
    main()
