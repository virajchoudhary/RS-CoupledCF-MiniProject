"""Single-positive sampled ranking metrics with item-ID tie breaking."""
import numpy as np


def positive_ranks(scores: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    if scores.shape != candidates.shape or scores.ndim != 2:
        raise ValueError('Scores and candidates must have matching 2D shapes')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite recommendation score')
    positive = scores[:, :1]
    ahead = (scores > positive) | ((scores == positive) & (candidates < candidates[:, :1]))
    return 1 + ahead.sum(axis=1)


def ranking_metrics(ranks: np.ndarray, ks=(5, 10, 20)) -> dict:
    ranks = np.asarray(ranks)
    if ranks.ndim != 1 or len(ranks) == 0 or np.any(ranks < 1):
        raise ValueError('Ranks must be a nonempty vector of positive integers')
    result = {}
    for k in ks:
        hit = ranks <= k
        result[f'HR@{k}'] = float(hit.mean())
        result[f'NDCG@{k}'] = float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean())
    return result
