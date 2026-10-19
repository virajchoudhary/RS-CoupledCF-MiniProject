"""MovieLens encoding, fixed holdouts and uniform unobserved-item sampling."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class DataBundle:
    train_users: np.ndarray
    train_items: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    user_features: np.ndarray
    item_features: np.ndarray
    observed: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self):
        return len(self.user_ids)

    @property
    def n_items(self):
        return len(self.item_ids)

    @property
    def cardinalities(self):
        return tuple(int(self.user_features[:, j].max()) + 1 for j in range(4))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            return cls(**{key: z[key] for key in cls.__dataclass_fields__})

    def validate(self):
        if self.validation.shape != self.test.shape or self.test.shape[0] != self.n_users:
            raise ValueError('Candidate arrays must contain one row per user')
        if self.observed.shape != (self.n_users, self.n_items):
            raise ValueError('Observed mask shape is inconsistent')
        if len(self.train_users) != len(self.train_items):
            raise ValueError('Training pair lengths differ')
        train = np.zeros_like(self.observed)
        train[self.train_users, self.train_items] = True
        if train.sum() != len(self.train_users):
            raise ValueError('Duplicate training pairs')
        rows = np.arange(self.n_users)
        if np.any(self.validation[:, 0] == self.test[:, 0]):
            raise ValueError('Validation and test positives overlap')
        for candidates in (self.validation, self.test):
            if np.any(candidates < 0) or np.any(candidates >= self.n_items):
                raise ValueError('Candidate item outside catalogue')
            if np.any(train[rows, candidates[:, 0]]):
                raise ValueError('Held-out positive leaked into training')
            if not np.all(self.observed[rows, candidates[:, 0]]):
                raise ValueError('Held-out positive is not observed')
            if np.any(self.observed[rows[:, None], candidates[:, 1:]]):
                raise ValueError('Evaluation negative is an observed interaction')
            if any(len(np.unique(row)) != len(row) for row in candidates):
                raise ValueError('Duplicate evaluation candidates')
        expected = train.copy()
        expected[rows, self.validation[:, 0]] = True
        expected[rows, self.test[:, 0]] = True
        if not np.array_equal(expected, self.observed):
            raise ValueError('Interactions are missing from or repeated across splits')
        if not np.isfinite(self.item_features).all():
            raise ValueError('Invalid side features')
        return {'users': self.n_users, 'items': self.n_items,
                'train_positives': len(self.train_users),
                'validation_positives': self.n_users, 'test_positives': self.n_users,
                'candidates_per_user': self.test.shape[1], 'leakage_checks': 'PASS'}


def prepare(raw: Path, output: Path, seed: int = 2026, negatives: int = 99):
    ratings = pd.read_csv(raw / 'ratings.dat', sep='::', engine='python',
                          names=['user', 'item', 'rating', 'timestamp'])
    users = pd.read_csv(raw / 'users.dat', sep='::', engine='python',
                        names=['user', 'gender', 'age', 'occupation', 'zip'],
                        dtype={'zip': str}).sort_values('user')
    movies = pd.read_csv(raw / 'movies.dat', sep='::', engine='python',
                         encoding='latin-1', names=['item', 'title', 'genres']).sort_values('item')
    actual = (len(ratings), len(users), len(movies), ratings['item'].nunique())
    if actual != (1000209, 6040, 3883, 3706):
        raise ValueError(f'Unexpected official MovieLens 1M statistics: {actual}')
    if ratings.duplicated(['user', 'item']).any():
        raise ValueError('Duplicate source interactions')
    uid = users.user.to_numpy(dtype=np.int64)
    iid = movies.item.to_numpy(dtype=np.int64)
    ru = np.searchsorted(uid, ratings.user.to_numpy())
    ri = np.searchsorted(iid, ratings.item.to_numpy())
    if not np.array_equal(uid[ru], ratings.user) or not np.array_equal(iid[ri], ratings.item):
        raise ValueError('Interaction references missing metadata')
    category_maps = {}
    features = []
    for column in ['gender', 'age', 'occupation', 'zip']:
        vocabulary = sorted(users[column].astype(str).unique())
        category_maps[column] = vocabulary
        mapping = {value: j for j, value in enumerate(vocabulary)}
        features.append(users[column].astype(str).map(mapping).to_numpy())
    user_features = np.stack(features, axis=1).astype(np.int64)
    genres = sorted(set('|'.join(movies.genres).split('|')))
    item_features = np.array([[int(g in entry.split('|')) for g in genres]
                              for entry in movies.genres], dtype=np.float32)
    observed = np.zeros((len(uid), len(iid)), dtype=bool)
    observed[ru, ri] = True
    rng = np.random.default_rng(seed)
    train_users, train_items, val, test = [], [], [], []
    for user in range(len(uid)):
        positives = np.flatnonzero(observed[user])
        heldout = rng.choice(positives, size=2, replace=False)
        remaining = positives[~np.isin(positives, heldout)]
        train_users.extend([user] * len(remaining))
        train_items.extend(remaining.tolist())
        unseen = np.flatnonzero(~observed[user])
        val.append(np.r_[heldout[0], rng.choice(unseen, negatives, replace=False)])
        test.append(np.r_[heldout[1], rng.choice(unseen, negatives, replace=False)])
    bundle = DataBundle(np.array(train_users, dtype=np.int64),
                        np.array(train_items, dtype=np.int64),
                        np.array(val, dtype=np.int64), np.array(test, dtype=np.int64),
                        user_features, item_features, observed, uid, iid)
    stats = bundle.validate()
    bundle.save(output / 'split.npz')
    manifest = {
        'dataset': 'MovieLens 1M', 'source': 'https://grouplens.org/datasets/movielens/1m/',
        'archive_md5': 'c4d9eecfca2ab87c1945afe126590906',
        'source_sha256': {p.name: sha256(p) for p in raw.iterdir() if p.is_file()},
        'split_sha256': sha256(output / 'split.npz'), 'split_seed': seed,
        'split_protocol': 'Seeded random validation and test positive per user; all ratings positive',
        'negative_protocol': 'Uniform unobserved catalogue items; 99 unique negatives per evaluation row',
        'ratings': len(ratings), 'rated_items': int(ratings.item.nunique()),
        'metadata_items': len(movies), 'max_original_item_id': int(iid.max()),
        'min_interactions': int(observed.sum(axis=1).min()),
        'genres': genres, 'user_category_maps': category_maps,
        'user_fields': ['gender', 'age', 'occupation', 'zip'], **stats,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def training_samples(data: DataBundle, seed: int, negatives: int = 4):
    if negatives < 1:
        raise ValueError('At least one training negative is required')
    rng = np.random.default_rng(seed)
    users = np.repeat(data.train_users, negatives + 1)
    items = np.empty(len(users), dtype=np.int64)
    items[::negatives + 1] = data.train_items
    positions = np.flatnonzero(np.arange(len(users)) % (negatives + 1) != 0)
    candidate = rng.integers(data.n_items, size=len(positions))
    invalid = data.observed[users[positions], candidate]
    while np.any(invalid):
        candidate[invalid] = rng.integers(data.n_items, size=int(invalid.sum()))
        invalid = data.observed[users[positions], candidate]
    items[positions] = candidate
    labels = np.zeros(len(users), dtype=np.float32)
    labels[::negatives + 1] = 1
    order = rng.permutation(len(users))
    return users[order], items[order], labels[order]
