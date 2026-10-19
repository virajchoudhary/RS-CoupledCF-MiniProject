import numpy as np
import pytest
from coupledcf.metrics import positive_ranks, ranking_metrics


def test_known_ranks_and_discount():
    scores = np.array([[3., 2., 1.], [1., 3., 2.], [2., 1., 3.]])
    candidates = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])
    ranks = positive_ranks(scores, candidates)
    np.testing.assert_array_equal(ranks, [1, 3, 2])
    result = ranking_metrics(ranks, (1, 2, 3))
    assert result['HR@1'] == pytest.approx(1 / 3)
    assert result['HR@2'] == pytest.approx(2 / 3)
    assert result['NDCG@2'] == pytest.approx((1 + 1 / np.log2(3)) / 3)


def test_ties_do_not_favor_positive_column():
    scores = np.ones((1, 4))
    candidates = np.array([[8, 2, 6, 9]])
    assert positive_ranks(scores, candidates)[0] == 3
    assert positive_ranks(scores[:, [0, 3, 1, 2]], candidates[:, [0, 3, 1, 2]])[0] == 3


def test_nonfinite_scores_rejected():
    with pytest.raises(ValueError, match='Non-finite'):
        positive_ranks(np.array([[np.nan, 1]]), np.array([[1, 2]]))


def test_random_reference_expectation():
    # Every possible positive rank occurs once, an exact uniform-rank reference.
    result = ranking_metrics(np.arange(1, 101), (10,))
    assert result['HR@10'] == .1
    assert result['NDCG@10'] == pytest.approx(sum(1 / np.log2(np.arange(2, 12))) / 100)
