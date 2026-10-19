from pathlib import Path
import copy
import numpy as np
import pytest
from coupledcf.data import DataBundle, training_samples


@pytest.fixture
def toy_data():
    observed = np.zeros((4, 12), dtype=bool)
    for u in range(4):
        observed[u, [u, (u + 1) % 12, (u + 2) % 12]] = True
    validation, test = [], []
    for u in range(4):
        unseen = np.flatnonzero(~observed[u])
        validation.append([u + 1, *unseen[:3]])
        test.append([u + 2, *unseen[3:6]])
    return DataBundle(np.arange(4), np.arange(4), np.array(validation), np.array(test),
                      np.zeros((4, 4), dtype=np.int64), np.ones((12, 18), dtype=np.float32),
                      observed, np.arange(4), np.arange(12))


def test_split_roundtrip_and_validation(toy_data, tmp_path):
    path = tmp_path / 'split.npz'
    toy_data.save(path)
    loaded = DataBundle.load(path)
    assert loaded.validate()['train_positives'] == 4
    for key in vars(toy_data):
        np.testing.assert_array_equal(getattr(toy_data, key), getattr(loaded, key))


def test_training_negative_exclusion_and_determinism(toy_data):
    first = training_samples(toy_data, 17)
    second = training_samples(toy_data, 17)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    users, items, labels = first
    assert (labels == 1).sum() == 4
    assert (labels == 0).sum() == 16
    assert not toy_data.observed[users[labels == 0], items[labels == 0]].any()


def test_heldout_leak_is_rejected(toy_data):
    toy_data.train_items[0] = toy_data.test[0, 0]
    with pytest.raises(ValueError, match='leaked'):
        toy_data.validate()


def test_observed_negative_is_rejected(toy_data):
    toy_data.test[0, 1] = toy_data.validation[0, 0]
    with pytest.raises(ValueError, match='negative is an observed'):
        toy_data.validate()


def test_duplicate_candidates_are_rejected(toy_data):
    toy_data.test[0, 2] = toy_data.test[0, 1]
    with pytest.raises(ValueError, match='Duplicate evaluation'):
        toy_data.validate()


def test_actual_full_dataset_contract():
    path = Path('data/processed/split.npz')
    if not path.exists():
        pytest.skip('Download and prepare MovieLens 1M to run full-data contract')
    result = DataBundle.load(path).validate()
    assert result == {'users': 6040, 'items': 3883, 'train_positives': 988129,
                      'validation_positives': 6040, 'test_positives': 6040,
                      'candidates_per_user': 100, 'leakage_checks': 'PASS'}
