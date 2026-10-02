"""Full-ranking evaluation and baseline behaviour on a tiny hand-made split."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import item_popularity, popularity_scorer
from src.data import validate_split
from src.evaluate import evaluate, exclusion_lists, rank_targets

COLS = ["user", "item", "rating", "timestamp", "seq", "user_id", "item_id"]


def _df(rows):
    return pd.DataFrame(rows, columns=COLS)


@pytest.fixture
def toy_split():
    # 2 users, 7 items. Items 5 (user 0) and 6 (user 1) are low-rated observations.
    history = _df([
        (0, 0, 5, 1, 0, 1, 10), (0, 5, 2, 1, 1, 1, 15), (0, 1, 4, 2, 2, 1, 11),
        (0, 2, 5, 3, 3, 1, 12), (0, 3, 5, 4, 4, 1, 13),
        (1, 0, 5, 1, 0, 2, 10), (1, 3, 4, 3, 1, 2, 13), (1, 6, 1, 3, 2, 2, 16),
        (1, 4, 5, 4, 3, 2, 14),
    ])
    pos = history[history.rating >= 4]
    split = {
        "n_users": 2, "n_items": 7, "positive_threshold": 4, "history": history,
        "train": pos[pos.seq.isin([0, 2]) | ((pos.user == 1) & (pos.seq == 0))].reset_index(drop=True),
        "validation": _df([(0, 2, 5, 3, 3, 1, 12), (1, 3, 4, 3, 1, 2, 13)]),
        "test": _df([(0, 3, 5, 4, 4, 1, 13), (1, 4, 5, 4, 3, 2, 14)]),
    }
    validate_split(split)
    return split


def test_rank_ignores_excluded_items():
    scores = np.array([[9.0, 8.0, 1.0, 0.0]])
    # items 0 and 1 are excluded, so target 2 is ranked first
    assert rank_targets(scores, np.array([2]), [np.array([0, 1])])[0] == 1
    assert rank_targets(scores, np.array([2]), [np.array([], dtype=int)])[0] == 3


def test_pessimistic_ties():
    scores = np.array([[1.0, 1.0, 1.0, 0.0]])
    assert rank_targets(scores, np.array([1]), [np.array([], dtype=int)])[0] == 3
    assert rank_targets(scores, np.array([1]), [np.array([], dtype=int)], "optimistic")[0] == 1


def test_target_in_excluded_raises():
    with pytest.raises(ValueError):
        rank_targets(np.zeros((1, 3)), np.array([1]), [np.array([1])])


def test_non_finite_scores_raise():
    with pytest.raises(ValueError):
        rank_targets(np.array([[np.nan, 1.0]]), np.array([1]), [np.array([], dtype=int)])


def test_candidate_sets(toy_split):
    val_ex = exclusion_lists(toy_split, "validation")
    test_ex = exclusion_lists(toy_split, "test")
    # user 0: low-rated item 5 is observed before validation -> excluded for both
    assert set(val_ex[0]) == {0, 5, 1}
    assert set(test_ex[0]) == {0, 5, 1, 2}           # + validation item
    # user 1: low-rated item 6 comes after validation -> excluded for test only
    assert set(val_ex[1]) == {0}
    assert set(test_ex[1]) == {0, 3, 6}
    # targets are never excluded; future test target stays a validation candidate
    assert 2 not in val_ex[0] and 3 not in val_ex[0] and 3 not in test_ex[0]


def test_oracle_and_constant_scorers(toy_split):
    def oracle(target):
        items = toy_split[target].sort_values("user")["item"].to_numpy()
        def fn(users):
            s = np.zeros((len(users), 7))
            s[np.arange(len(users)), items[users]] = 1.0
            return s
        return fn

    for target in ("validation", "test"):
        summary, ranks = evaluate(oracle(target), toy_split, target)
        assert summary["ndcg@10"] == 1.0 and summary["hr@10"] == 1.0
        # constant scores must not be rewarded: pessimistic rank = number of candidates
        summary, ranks = evaluate(lambda u: np.zeros((len(u), 7)), toy_split, target)
        assert np.all(ranks == [7 - len(x) for x in exclusion_lists(toy_split, target)])


def test_popularity_uses_train_only(toy_split):
    pop = item_popularity(toy_split["train"], 7)
    assert pop.tolist() == [2, 1, 0, 0, 0, 0, 0]
    altered = dict(toy_split, test=toy_split["test"].assign(item=[6, 6]))
    assert np.array_equal(popularity_scorer(altered["train"], 7)([0, 1]),
                          popularity_scorer(toy_split["train"], 7)([0, 1]))
