"""Hand-computed checks for the ranking metrics."""

import math

import numpy as np
import pytest

from src import metrics


def test_ndcg_rank_1_is_one():
    assert metrics.ndcg_from_rank([1])[0] == 1.0


def test_ndcg_rank_2():
    assert metrics.ndcg_from_rank([2])[0] == pytest.approx(1 / math.log2(3))


def test_ndcg_rank_10_and_11():
    assert metrics.ndcg_from_rank([10])[0] == pytest.approx(1 / math.log2(11))
    assert metrics.ndcg_from_rank([11])[0] == 0.0


def test_hr_rank_11_is_zero():
    assert metrics.hit_rate_from_rank([11])[0] == 0.0
    assert metrics.hit_rate_from_rank([10])[0] == 1.0


def test_rank_5_one_positive():
    assert metrics.hit_rate_from_rank([5])[0] == 1.0
    assert metrics.recall_from_rank([5])[0] == 1.0
    assert metrics.mrr_from_rank([5])[0] == pytest.approx(0.2)
    assert metrics.ndcg_from_rank([5])[0] == pytest.approx(1 / math.log2(6))


def test_mrr_outside_k_is_zero():
    assert metrics.mrr_from_rank([11])[0] == 0.0
    assert metrics.mrr_from_rank([1])[0] == 1.0


def test_invalid_rank_raises():
    with pytest.raises(ValueError):
        metrics.ndcg_from_rank([0])


def test_vectorised_matches_list_based_single_relevant():
    ranked = list(range(100))
    for rank in [1, 2, 3, 7, 10, 11, 50]:
        target = ranked[rank - 1]
        assert metrics.ndcg_at_k(ranked, {target}) == pytest.approx(metrics.ndcg_from_rank([rank])[0])
        assert metrics.hit_rate_at_k(ranked, {target}) == metrics.hit_rate_from_rank([rank])[0]
        assert metrics.recall_at_k(ranked, {target}) == metrics.recall_from_rank([rank])[0]
        assert metrics.mrr_at_k(ranked, {target}) == pytest.approx(metrics.mrr_from_rank([rank])[0])


def test_recall_multiple_relevant():
    ranked = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    assert metrics.recall_at_k(ranked, {"a", "c", "z", "k"}) == pytest.approx(2 / 4)  # k is 11th
    assert metrics.recall_at_k(ranked, {"a", "b"}) == 1.0
    assert metrics.recall_at_k(ranked, {"k", "l"}) == 0.0
    assert metrics.recall_at_k(ranked, {"a", "c", "k"}, k=3) == pytest.approx(2 / 3)


def test_hr_multiple_relevant():
    ranked = list("abcdefghijkl")
    assert metrics.hit_rate_at_k(ranked, {"k", "j"}) == 1.0
    assert metrics.hit_rate_at_k(ranked, {"k", "l"}) == 0.0


def test_ndcg_multiple_relevant():
    ranked = list("abcdefghijkl")
    # relevant at positions 1 and 3; ideal: positions 1 and 2
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_k(ranked, {"a", "c"}) == pytest.approx(expected)
    assert metrics.ndcg_at_k(ranked, {"a", "b"}) == pytest.approx(1.0)


def test_empty_relevant_raises():
    with pytest.raises(ValueError):
        metrics.recall_at_k([1, 2], set())
