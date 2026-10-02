"""Leakage and determinism checks for the chronological leave-two-out split."""

import numpy as np
import pandas as pd
import pytest

from src.data import (DATASETS, SplitValidationError, build, chronological_leave_two_out,
                      load_processed, load_ratings, previously_rated, split_fingerprint, tie_break_key,
                      validate_split)
from src.utils import load_config, project_path, sha256_file

CFG = load_config()
SEED = CFG["split"]["split_seed"]

# The split used by every model B0-B4. Changing it requires a dated RESEARCH_LOG.md entry.
FROZEN_SPLIT_FINGERPRINT = "6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989"


@pytest.fixture(scope="module")
def built():
    return build(CFG)


@pytest.fixture(scope="module")
def split(built):
    return built[0]


# ---------------------------------------------------------------- real MovieLens-100K

def test_raw_data_is_untouched():
    spec = DATASETS[CFG["dataset"]]
    path = project_path(CFG["paths"]["raw_dir"]) / spec["dirname"] / spec["ratings_file"]
    assert sha256_file(path) == spec["ratings_sha256"]


def test_raw_counts():
    r = load_ratings(CFG["dataset"], project_path(CFG["paths"]["raw_dir"]))
    assert (len(r), r.user_id.nunique(), r.item_id.nunique()) == (100000, 943, 1682)


def test_validation_not_in_train(split):
    train = set(zip(split["train"].user, split["train"].item))
    assert not any(p in train for p in zip(split["validation"].user, split["validation"].item))


def test_test_not_in_train(split):
    train = set(zip(split["train"].user, split["train"].item))
    assert not any(p in train for p in zip(split["test"].user, split["test"].item))


def test_validation_differs_from_test(split):
    v = split["validation"].sort_values("user").item.to_numpy()
    t = split["test"].sort_values("user").item.to_numpy()
    assert np.all(v != t)


def test_every_user_has_train_val_test(split):
    users = np.arange(split["n_users"])
    assert np.array_equal(np.unique(split["train"].user), users)
    assert np.array_equal(np.sort(split["validation"].user), users)
    assert np.array_equal(np.sort(split["test"].user), users)


def test_chronological_order(split):
    last_train = split["train"].groupby("user").timestamp.max().sort_index().to_numpy()
    v = split["validation"].sort_values("user").timestamp.to_numpy()
    t = split["test"].sort_values("user").timestamp.to_numpy()
    assert np.all(last_train <= v) and np.all(v <= t)


def test_only_positives(split):
    thr = CFG["positive_rating_threshold"]
    for name in ("train", "validation", "test"):
        assert (split[name].rating >= thr).all()


def test_all_positives_of_retained_users_are_used(split):
    r = load_ratings(CFG["dataset"], project_path(CFG["paths"]["raw_dir"]))
    pos = r[(r.rating >= CFG["positive_rating_threshold"]) & r.user_id.isin(split["user_map"].user_id)]
    total = sum(len(split[n]) for n in ("train", "validation", "test"))
    assert total == len(pos)


def test_validate_split_passes(split):
    validate_split(split)


def test_deterministic_rebuild(split):
    again, _ = build(CFG)
    for name in ("train", "validation", "test", "history", "user_map", "item_map"):
        pd.testing.assert_frame_equal(split[name], again[name])


def test_split_matches_frozen_fingerprint(split):
    assert split_fingerprint(split) == FROZEN_SPLIT_FINGERPRINT
    assert split_fingerprint(load_processed(CFG)) == FROZEN_SPLIT_FINGERPRINT


def test_training_seed_never_changes_split(split):
    for seed in (0, 1, 42, 12345):
        other, _ = build(dict(CFG, seed=seed))
        for name in ("train", "validation", "test", "history"):
            pd.testing.assert_frame_equal(split[name], other[name])


def test_split_seed_only_affects_tied_boundaries(split):
    other, _ = build(dict(CFG, split=dict(CFG["split"], split_seed=SEED + 1)))
    v1 = split["validation"].sort_values("user").set_index("user")
    v2 = other["validation"].sort_values("user").set_index("user")
    t1 = split["test"].sort_values("user").set_index("user")
    changed = (v1["item"] != v2["item"]) | (t1["item"] != other["test"].sort_values("user").set_index("user")["item"])
    assert changed.any()                                   # the tie-break is actually used ...
    last_train_ts = split["train"].groupby("user")["timestamp"].max()
    tied = (last_train_ts == v1["timestamp"]) | (v1["timestamp"] == t1["timestamp"])
    assert not (changed & ~tied).any()                     # ... and only where timestamps tie


def test_split_independent_of_row_order():
    r = load_ratings(CFG["dataset"], project_path(CFG["paths"]["raw_dir"]))
    a, _ = chronological_leave_two_out(r, 4, 3, SEED)
    shuffled = r.sample(frac=1.0, random_state=123).reset_index(drop=True)
    b, _ = chronological_leave_two_out(shuffled, 4, 3, SEED)
    for name in ("train", "validation", "test", "history"):
        pd.testing.assert_frame_equal(a[name], b[name])


def test_saved_files_match_rebuild(split):
    saved = load_processed(CFG)
    for name in ("train", "validation", "test", "history"):
        pd.testing.assert_frame_equal(split[name], saved[name])


# ---------------------------------------------------------------- synthetic cases

def _ratings(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_toy_split_and_filtering():
    r = _ratings([
        (1, 10, 5, 100), (1, 11, 4, 200), (1, 12, 5, 300), (1, 13, 2, 400),   # 3 positives
        (2, 10, 5, 100), (2, 11, 5, 200),                                     # 2 positives -> removed
        (3, 12, 1, 100),                                                      # 0 positives -> removed
    ])
    split, stats = chronological_leave_two_out(r, 4, 3, SEED)
    assert split["n_users"] == 1 and stats["users_removed"] == 2
    assert split["n_items"] == 4                          # full catalog kept
    assert split["train"].item_id.tolist() == [10]
    assert split["validation"].item_id.tolist() == [11]
    assert split["test"].item_id.tolist() == [12]         # low rating at t=400 ignored
    validate_split(split)


def test_tie_break_key_is_pinned():
    # Guards against any change to the hashing scheme, which would silently change the split.
    assert tie_break_key([1], [10], 2026)[0] == 14161937671155015590


def test_timestamp_tie_uses_hash_not_item_id():
    # Items 10 and 11 share a timestamp. Under split_seed 2026 the hash puts 11 before 10,
    # so 10 is the test item (item-id order would have made 11 the test item).
    k10, k11 = tie_break_key([7, 7], [10, 11], 2026)
    assert k11 < k10
    r = _ratings([(7, 30, 5, 100), (7, 11, 5, 500), (7, 10, 5, 500)])
    split, _ = chronological_leave_two_out(r, 4, 3, 2026)
    assert split["validation"].item_id.tolist() == [11]
    assert split["test"].item_id.tolist() == [10]


def test_tie_handling_deterministic_and_seed_dependent():
    r = _ratings([(7, 30, 5, 100), (7, 12, 5, 500), (7, 13, 5, 500)])
    a, _ = chronological_leave_two_out(r, 4, 3, 2026)
    b, _ = chronological_leave_two_out(r.iloc[::-1].reset_index(drop=True), 4, 3, 2026)
    assert a["test"].item_id.tolist() == b["test"].item_id.tolist() == [13]
    c, _ = chronological_leave_two_out(r, 4, 3, 2027)
    assert c["test"].item_id.tolist() == [12]


def test_candidates_exclude_all_prior_ratings_only():
    r = _ratings([
        (1, 10, 5, 100),   # train positive
        (1, 20, 2, 150),   # low rating before validation      -> excluded for val and test
        (1, 11, 4, 200),   # train positive
        (1, 12, 5, 300),   # VALIDATION
        (1, 21, 1, 350),   # low rating between val and test   -> excluded for test only
        (1, 13, 5, 400),   # TEST
        (1, 22, 3, 500),   # low rating after test             -> never excluded (future)
    ])
    split, _ = chronological_leave_two_out(r, 4, 3, SEED)
    item = dict(zip(split["item_map"].item_id, split["item_map"].item))
    assert split["validation"].item_id.tolist() == [12]
    assert split["test"].item_id.tolist() == [13]
    assert sorted(split["train"].item_id) == [10, 11]          # low ratings never become positives
    val_ex = set(previously_rated(split, "validation")[0])
    test_ex = set(previously_rated(split, "test")[0])
    assert val_ex == {item[i] for i in (10, 20, 11)}
    assert test_ex == {item[i] for i in (10, 20, 11, 12, 21)}  # includes the validation item
    assert item[12] not in val_ex and item[13] not in test_ex  # targets kept
    assert item[13] not in val_ex                             # future test item stays a val candidate
    assert item[22] not in test_ex                            # future low rating not used


def test_candidates_same_second_uses_frozen_order():
    # Low rating 11 shares the validation timestamp; whether it counts as "before" is decided
    # by the same hash order that defines the split, never by looking at the future.
    r = _ratings([(7, 30, 5, 100), (7, 31, 5, 200), (7, 10, 5, 500), (7, 11, 2, 500), (7, 40, 5, 900)])
    split, _ = chronological_leave_two_out(r, 4, 3, 2026)
    assert split["validation"].item_id.tolist() == [10]
    item = dict(zip(split["item_map"].item_id, split["item_map"].item))
    k10, k11 = tie_break_key([7, 7], [10, 11], 2026)
    assert (item[11] in set(previously_rated(split, "validation")[0])) == (k11 < k10)
    assert item[11] in set(previously_rated(split, "test")[0])


def test_min_positives_below_three_rejected():
    with pytest.raises(ValueError):
        chronological_leave_two_out(_ratings([(1, 1, 5, 1)]), 4, 2, SEED)


def test_validate_split_detects_leakage():
    r = _ratings([(1, 10, 5, 1), (1, 11, 5, 2), (1, 12, 5, 3), (1, 13, 5, 4)])
    split, _ = chronological_leave_two_out(r, 4, 3, SEED)
    leaky = dict(split, train=pd.concat([split["train"], split["test"]], ignore_index=True))
    with pytest.raises(SplitValidationError, match="test item"):
        validate_split(leaky)
    same = dict(split, validation=split["test"].copy())
    with pytest.raises(SplitValidationError, match="coincide"):
        validate_split(same)
    future = dict(split, train=split["train"].assign(seq=99))
    with pytest.raises(SplitValidationError, match="ordered after"):
        validate_split(future)
