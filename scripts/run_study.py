"""Schedule independent seeds with exclusive run directories and bounded CPU use."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/main.json')
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    records = []

    def run(seed):
        start = time.perf_counter()
        with (ROOT / f'.audit/training_seed{seed}.log').open('w') as log:
            process = subprocess.run([sys.executable, str(ROOT / 'scripts/run_experiments.py'),
                                      '--config', args.config, '--seed', str(seed)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT)
        return {'seed': seed, 'returncode': process.returncode,
                'wall_seconds': time.perf_counter() - start}

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, seed) for seed in config['seeds']]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            receipt = {'jobs': args.jobs, 'completed_seeds': records,
                       'status': 'RUNNING' if len(records) < len(config['seeds']) else
                           ('COMPLETE' if all(row['returncode'] == 0 for row in records) else 'FAILED')}
            (ROOT / '.audit/scheduler.json').write_text(json.dumps(receipt, indent=2))
            print(record, flush=True)
    if any(record['returncode'] != 0 for record in records):
        raise RuntimeError('A seed failed; inspect .audit/training_seed*.log')


if __name__ == '__main__':
    main()
