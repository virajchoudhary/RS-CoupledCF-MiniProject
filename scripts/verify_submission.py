"""Verify measured evidence and final files, including checkpoint prediction replay."""
from pathlib import Path
import argparse
import csv
import json
import re
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
import nbformat
from coupledcf.data import DataBundle, sha256
from coupledcf.engine import evaluate, set_seed, write_json
from coupledcf.metrics import ranking_metrics
from coupledcf.models import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments-only', action='store_true')
    parser.add_argument('--reuse-experiment-verification', action='store_true',
                        help='Reuse prediction replay only when all recorded artifacts and evaluation sources are byte-identical')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/main.json').read_text())
    split = ROOT / config['split_path']
    data = DataBundle.load(split)
    checks = {'data': data.validate(), 'runs': [], 'split_sha256': sha256(split),
              'study_config_sha256': sha256(ROOT / 'configs/main.json'),
              'verification_script_sha256': sha256(Path(__file__)),
              'evaluation_source_sha256': {name: sha256(ROOT / 'src/coupledcf' / name)
                  for name in ['models.py', 'metrics.py', 'engine.py', 'data.py']}}
    cache = {}
    if args.reuse_experiment_verification:
        if args.experiments_only:
            raise ValueError('Fresh experiment verification cannot reuse its own receipt')
        previous = json.loads((ROOT / '.audit/experiment_verification.json').read_text())
        assert previous['status'] == 'PASS'
        for key in ['split_sha256', 'study_config_sha256', 'verification_script_sha256', 'evaluation_source_sha256']:
            assert previous[key] == checks[key], f'Experiment verification is stale: {key}'
        cache = {row['name']: row for row in previous['runs']}
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    assert manifest['split_sha256'] == sha256(split)
    set_seed(0, 2)
    for seed in config['seeds']:
        for variant in config['variants']:
            path = ROOT / config['output_dir'] / f'{variant}_seed{seed}'
            result = json.loads((path / 'result.json').read_text())
            history = json.loads((path / 'history.json').read_text())
            assert result['status'] == 'COMPLETE'
            assert result['split_sha256'] == sha256(split)
            assert result['checkpoint_sha256'] == sha256(path / 'best.pt')
            assert result['config_sha256'] == sha256(path / 'config.json')
            assert result['config'] == {'model': {'variant': variant, **config['model']},
                                        'training': config['training'], 'seed': seed}
            assert len(history) == result['completed_epochs'] == config['training']['epochs']
            assert all(row['examples'] == 4940645 for row in history)
            selected = max(history, key=lambda row: row['validation']['NDCG@10'])
            assert selected['epoch'] == result['selected_epoch']
            with np.load(path / 'ranks.npz') as ranks:
                assert ranking_metrics(ranks['test']) == result['test']
                assert ranking_metrics(ranks['validation']) == result['validation']
            if cache:
                record = cache[path.name]
                for filename, key in [('result.json', 'result_sha256'), ('best.pt', 'checkpoint_sha256'),
                                      ('ranks.npz', 'ranks_sha256')]:
                    assert record[key] == sha256(path / filename), f'Cached replay is stale: {path.name}/{filename}'
                replayed = record['test']
                assert replayed == result['test']
            else:
                checkpoint = torch.load(path / 'best.pt', map_location='cpu', weights_only=True)
                assert checkpoint['epoch'] == result['selected_epoch']
                model = build_model(data, ModelConfig(**result['config']['model']))
                model.load_state_dict(checkpoint['state_dict'])
                replayed, ranks = evaluate(model, data.test, 'cpu')
                assert replayed == result['test'], f'Prediction replay differs for {path.name}'
                with np.load(path / 'ranks.npz') as saved:
                    np.testing.assert_array_equal(ranks, saved['test'])
                assert sum(p.numel() for p in model.parameters()) == result['parameters']
            for name, digest in json.loads((path / 'source_hashes.json').read_text()).items():
                assert sha256(path / 'source' / name) == digest
            checks['runs'].append({'name': path.name, 'status': 'PASS',
                                    'selected_epoch': selected['epoch'], 'test': replayed,
                                    'parameters': result['parameters'],
                                    'wall_seconds': result['wall_seconds'],
                                    'sampled_peak_rss_mb': result['sampled_peak_rss_mb'],
                                    'result_sha256': sha256(path / 'result.json'),
                                    'checkpoint_sha256': sha256(path / 'best.pt'),
                                    'ranks_sha256': sha256(path / 'ranks.npz'),
                                    'prediction_replay': 'reused matching verified bytes' if cache else 'recomputed exact ranks'})
            print('PASS checkpoint, split, history, source and prediction replay', path.name,
                  '(reused matching verified bytes)' if cache else '(recomputed)', flush=True)
    with (ROOT / 'results/summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == len(config['variants'])
    for row in summary:
        matching = [r for r in checks['runs'] if r['name'].startswith(row['variant'] + '_seed')]
        assert len(matching) == len(config['seeds'])
        assert int(row['runs']) == len(matching)
        assert all(int(row['parameters']) == r['parameters'] for r in matching)
        for metric in ['HR@5', 'NDCG@5', 'HR@10', 'NDCG@10', 'HR@20', 'NDCG@20',
                       'wall_seconds', 'sampled_peak_rss_mb']:
            values = [r['test'][metric] if metric in r['test'] else r[metric] for r in matching]
            assert np.isclose(float(row[metric + '_mean']), np.mean(values), atol=1e-12, rtol=0)
            assert np.isclose(float(row[metric + '_std']), np.std(values, ddof=1), atol=1e-12, rtol=0)
    checks['summary'] = 'PASS all ranking and cost means/sample standard deviations, counts and parameters'
    if not args.experiments_only:
        required = ['README.md', 'requirements.txt', 'docs/research/paper_analysis.md',
                    'docs/methodology.md', 'docs/final_report.md', 'docs/viva_questions.md',
                    'docs/requirements_checklist.md', 'deliverables/final_report.pdf',
                    'deliverables/CoupledCF_presentation.pptx',
                    'notebooks/CoupledCF_reproduction.ipynb',
                    'notebooks/CoupledCF_reproduction.executed.ipynb',
                    'results/ablation.csv', 'results/cohorts.csv',
                    'results/figures/model_comparison.png', 'results/figures/learning_curves.png',
                    'results/figures/topk.png']
        for name in required:
            assert (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0, name
        notebook = nbformat.read(ROOT / required[10], as_version=4)
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert all(cell.execution_count is not None for cell in cells)
        assert not any(o.output_type == 'error' for cell in cells for o in cell.outputs)
        report = (ROOT / 'docs/final_report.md').read_text(encoding='utf-8')
        for row in summary:
            for metric in ['HR@10', 'NDCG@10']:
                assert f'{float(row[metric + "_mean"]):.4f}' in report
        with zipfile.ZipFile(ROOT / 'deliverables/CoupledCF_presentation.pptx') as z:
            assert z.testzip() is None
            slides = [name for name in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', name)]
            assert len(slides) >= 12
        checks['deliverables'] = {'status': 'PASS', 'required_files': len(required),
                                   'executed_notebook_cells': len(cells), 'slides': len(slides)}
    checks['status'] = 'PASS'
    checks['limitations'] = ['Student identity and actual individual contributions need human completion',
                            'CUDA execution was not tested; all reported experiments used CPU',
                            'Partial reproduction with disclosed architecture and protocol adaptations']
    destination = ROOT / '.audit' / ('experiment_verification.json' if args.experiments_only else 'verification.json')
    write_json(destination, checks)
    print('PASS', len(checks['runs']), 'completed runs verified; receipt', destination)


if __name__ == '__main__':
    main()
