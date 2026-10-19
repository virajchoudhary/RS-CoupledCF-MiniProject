"""Completed run records must retain the checkpoint they claim to preserve."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('checkpoint_state', ['missing', 'changed', 'matching'])
def test_completed_run_requires_matching_checkpoint(tmp_path, monkeypatch, capsys, checkpoint_state):
    script = Path(__file__).resolve().parents[1] / 'scripts/run_experiments.py'
    spec = importlib.util.spec_from_file_location('study_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.chdir(tmp_path)
    split = tmp_path / 'split.npz'
    split.write_bytes(b'fixed split fixture')
    output = tmp_path / 'results/main/deepcf_seed42'
    output.mkdir(parents=True)
    checkpoint = output / 'best.pt'
    checkpoint.write_bytes(b'opaque checkpoint fixture')
    record = {
        'status': 'COMPLETE',
        'config': {'model': vars(runner.ModelConfig(variant='deepcf')),
                   'training': vars(runner.TrainConfig()), 'seed': 42},
        'split_sha256': runner.sha256(split),
        'checkpoint_sha256': runner.sha256(checkpoint),
    }
    (output / 'result.json').write_text(json.dumps(record))
    config = {'split_path': str(split), 'output_dir': 'results/main',
              'seeds': [42], 'variants': ['deepcf'], 'model': {}, 'training': {}}
    (tmp_path / 'config.json').write_text(json.dumps(config))
    class DataFixture:
        def validate(self):
            return {'fixture': 'valid'}
    monkeypatch.setattr(runner.DataBundle, 'load', lambda _: DataFixture())
    monkeypatch.setattr(sys, 'argv', ['run_experiments.py', '--config', 'config.json'])
    monkeypatch.setattr(runner, 'run_experiment', lambda *args: pytest.fail('Must not retrain completed records'))
    if checkpoint_state == 'missing':
        checkpoint.unlink()
        with pytest.raises(FileNotFoundError, match='reference_main'):
            runner.main()
    elif checkpoint_state == 'changed':
        checkpoint.write_bytes(b'changed checkpoint')
        with pytest.raises(ValueError, match='checkpoint hash differs'):
            runner.main()
    else:
        runner.main()
        assert 'Preserving completed run' in capsys.readouterr().out
