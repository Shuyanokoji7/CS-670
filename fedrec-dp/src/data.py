"""Dataset loading, implicit-feedback conversion and chronological leave-two-out split.

Run `python -m src.data` to (re)build data/processed/<dataset>/ and the dataset
summary/report in results/. Raw files under data/raw/ are only ever read.

The split depends only on the raw data and `split.split_seed` in the config. The
model-training `seed` is never read here, so no training seed can change the split.
"""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import load_config, project_path, sha256_file, write_json

# Adding a dataset = adding an entry here. Every loader returns the same four
# columns (user_id, item_id, rating, timestamp) with raw ids.
DATASETS = {
    "movielens_100k": {
        "title": "MovieLens-100K",
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "dirname": "ml-100k",
        "ratings_file": "u.data",
        "sep": "\t",
        "ratings_sha256": "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490",
    },
    # Registered for the later generalisation study; not used in the current phase.
    "movielens_1m": {
        "title": "MovieLens-1M",
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "dirname": "ml-1m",
        "ratings_file": "ratings.dat",
        "sep": "::",
        "ratings_sha256": None,  # pin after first download
    },
}

RAW_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
# `seq` = position of the rating in the user's total order over ALL their ratings.
SPLIT_COLUMNS = ["user", "item", "rating", "timestamp", "seq", "user_id", "item_id"]
TABLES = ["train", "validation", "test", "history", "user_map", "item_map"]


class SplitValidationError(RuntimeError):
    """Raised when a train/validation/test split violates a leakage invariant."""


# --------------------------------------------------------------------------- raw data

def ensure_raw(name, raw_dir):
    """Return the path to the raw ratings file, downloading the archive only if missing."""
    spec = DATASETS[name]
    raw_dir = Path(raw_dir)
    ratings_path = raw_dir / spec["dirname"] / spec["ratings_file"]
    if not ratings_path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = raw_dir / f"{spec['dirname']}.zip"
        if not zip_path.exists():
            print(f"Downloading {spec['url']} ...")
            urllib.request.urlretrieve(spec["url"], zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
    expected = spec["ratings_sha256"]
    actual = sha256_file(ratings_path)
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"{ratings_path} has sha256 {actual}, expected {expected}. "
            "Raw data is modified or is not the official GroupLens release."
        )
    return ratings_path


def load_ratings(name, raw_dir):
    """Load raw explicit ratings as a DataFrame with columns RAW_COLUMNS (raw ids)."""
    spec = DATASETS[name]
    path = ensure_raw(name, raw_dir)
    df = pd.read_csv(
        path,
        sep=spec["sep"],
        names=RAW_COLUMNS,
        engine="python" if len(spec["sep"]) > 1 else "c",
        dtype="int64",
    )
    if df.duplicated(["user_id", "item_id"]).any():
        raise ValueError("Raw data contains duplicate (user, item) ratings.")
    return df


# --------------------------------------------------------------------------- split

def tie_break_key(user_ids, item_ids, split_seed):
    """Stable pseudo-random uint64 for each (user_id, item_id, split_seed).

    Only used to order a user's ratings that share the same timestamp: equal
    timestamps carry no reliable internal chronology. SHA-256 (not Python's salted
    `hash()`) makes the key identical across runs, machines and Python versions.
    """
    keys = np.empty(len(user_ids), dtype=np.uint64)
    for n, (u, i) in enumerate(zip(user_ids, item_ids)):
        digest = hashlib.sha256(f"{int(u)}:{int(i)}:{int(split_seed)}".encode()).digest()
        keys[n] = int.from_bytes(digest[:8], "big")
    return keys


def chronological_leave_two_out(ratings, threshold, min_positives, split_seed):
    """Convert to implicit feedback and split each user's positives chronologically.

    Every user's ratings (all ratings, not just positives) are put in one total order
    by (timestamp, tie_break_key). Positives = rating >= threshold. In that order the
    last positive is TEST, the second-last VALIDATION, the rest TRAIN. Users with
    fewer than `min_positives` positives are removed (and reported).

    `history` keeps all ratings of retained users with their order position `seq`;
    it defines which items were already observed before each held-out event.

    The item index space is the full catalog of items that appear in the raw ratings.

    Returns (split dict, stats dict).
    """
    if min_positives < 3:
        raise ValueError("min_positives must be >= 3 (1 train + 1 validation + 1 test).")

    ordered = ratings.assign(tie_key=tie_break_key(ratings["user_id"], ratings["item_id"], split_seed))
    ordered = ordered.sort_values(["user_id", "timestamp", "tie_key"], kind="mergesort").reset_index(drop=True)
    if ordered.duplicated(["user_id", "timestamp", "tie_key"]).any():
        raise RuntimeError("tie_break_key collision; choose a different split_seed.")
    ordered["seq"] = ordered.groupby("user_id").cumcount()

    positives = ordered[ordered["rating"] >= threshold]
    pos_counts = positives.groupby("user_id").size().reindex(
        np.sort(ratings["user_id"].unique()), fill_value=0
    )
    kept_users = pos_counts.index[pos_counts >= min_positives]
    removed = pos_counts[pos_counts < min_positives]

    user_ids = np.sort(kept_users.to_numpy())
    item_ids = np.sort(ratings["item_id"].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)

    history = ordered[ordered["user_id"].isin(kept_users)]
    history = history.assign(
        user=user_index.loc[history["user_id"]].to_numpy(),
        item=item_index.loc[history["item_id"]].to_numpy(),
    )[SPLIT_COLUMNS].reset_index(drop=True)

    pos = history[history["rating"] >= threshold]
    from_end = pos.groupby("user").cumcount(ascending=False)
    split = {
        "n_users": len(user_ids),
        "n_items": len(item_ids),
        "positive_threshold": int(threshold),
        "train": pos[from_end >= 2].reset_index(drop=True),
        "validation": pos[from_end == 1].reset_index(drop=True),
        "test": pos[from_end == 0].reset_index(drop=True),
        "history": history,
        "user_map": pd.DataFrame({"user": np.arange(len(user_ids)), "user_id": user_ids}),
        "item_map": pd.DataFrame({"item": np.arange(len(item_ids)), "item_id": item_ids}),
    }
    stats = {
        "positive_interactions": int(len(positives)),
        "users_removed": int(len(removed)),
        "users_removed_with_zero_positives": int((removed == 0).sum()),
        "removed_user_positive_counts": {str(k): int(v) for k, v in removed.items()},
    }
    return split, stats


def validate_split(split):
    """Check every leakage/consistency invariant; raise SplitValidationError on failure."""
    train, val, test, history = split["train"], split["validation"], split["test"], split["history"]
    n_users = split["n_users"]
    errors = []

    all_users = np.arange(n_users)
    for name, df in [("validation", val), ("test", test)]:
        if len(df) != n_users or not np.array_equal(np.sort(df["user"].to_numpy()), all_users):
            errors.append(f"{name} must contain exactly one item for every user")
    if not np.array_equal(np.unique(train["user"].to_numpy()), all_users):
        errors.append("every evaluated user must have at least one training positive")

    train_pairs = set(zip(train["user"], train["item"]))
    if any(p in train_pairs for p in zip(val["user"], val["item"])):
        errors.append("a validation item is among the user's training positives")
    if any(p in train_pairs for p in zip(test["user"], test["item"])):
        errors.append("a test item is among the user's training positives")

    v = val.set_index("user").sort_index()
    t = test.set_index("user").sort_index()
    if (v["item"] == t["item"]).any():
        errors.append("validation and test item coincide for some user")

    # Chronology in the user's total order (timestamp, then fixed hash tie-break) ...
    last_train = train.groupby("user")[["seq", "timestamp"]].max().sort_index()
    if not (last_train["seq"] < v["seq"]).all():
        errors.append("some training positive is ordered after the validation positive")
    if not (v["seq"] < t["seq"]).all():
        errors.append("validation positive is not ordered before the test positive")
    # ... which implies non-decreasing timestamps (ties are possible and reported).
    if (last_train["timestamp"] > v["timestamp"]).any() or (v["timestamp"] > t["timestamp"]).any():
        errors.append("timestamps violate train <= validation <= test")

    # The split must be exactly the positives of `history`, which drives candidate filtering.
    if history.duplicated(["user", "item"]).any():
        errors.append("history contains a duplicated (user, item)")
    hist_keys = set(zip(history["user"], history["item"], history["seq"]))
    split_rows = pd.concat([train, val, test])
    if len(split_rows) != int((history["rating"] >= split["positive_threshold"]).sum()) or \
            (split_rows["rating"] < split["positive_threshold"]).any() or \
            not all(k in hist_keys for k in zip(split_rows["user"], split_rows["item"], split_rows["seq"])):
        errors.append("train/validation/test are not exactly the positives in history")

    if errors:
        raise SplitValidationError("Split validation failed:\n  - " + "\n  - ".join(errors))


def timestamp_tie_stats(split):
    """How often a split boundary had to be decided by the hash tie-break."""
    train, v, t = split["train"], split["validation"], split["test"]
    last_train_ts = train.groupby("user")["timestamp"].max().sort_index().to_numpy()
    v_ts = v.sort_values("user")["timestamp"].to_numpy()
    t_ts = t.sort_values("user")["timestamp"].to_numpy()
    return {
        "users_with_train_val_timestamp_tie": int((last_train_ts == v_ts).sum()),
        "users_with_val_test_timestamp_tie": int((v_ts == t_ts).sum()),
    }


def split_fingerprint(split):
    """SHA-256 over the split's contents (format-independent). Pinned in the tests."""
    h = hashlib.sha256()
    for name in ("train", "validation", "test", "history"):
        df = split[name]
        h.update(name.encode())
        h.update(np.ascontiguousarray(df[SPLIT_COLUMNS].to_numpy(dtype=np.int64)).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- candidates

def previously_rated(split, target):
    """Per user: items rated (with ANY rating) strictly before the `target` event.

    These are excluded from the target's candidate set. "Before" is the user's total
    order, so ratings after the target (future) never filter its candidates, and the
    target itself is never excluded. For test this includes the validation item.
    """
    history = split["history"]
    target_seq = split[target].set_index("user")["seq"].sort_index().to_numpy()
    prior = history[history["seq"].to_numpy() < target_seq[history["user"].to_numpy()]]
    out = [np.empty(0, dtype=np.int64) for _ in range(split["n_users"])]
    for u, items in prior.groupby("user")["item"]:
        out[u] = items.to_numpy(dtype=np.int64)
    return out


def candidate_counts(split, target):
    """Number of ranked candidates per user (full ranking over the catalog)."""
    return np.array([split["n_items"] - len(x) for x in previously_rated(split, target)])


# --------------------------------------------------------------------------- I/O

def processed_dir(cfg):
    return project_path(cfg["paths"]["processed_dir"]) / DATASETS[cfg["dataset"]]["dirname"]


def build(cfg):
    """Full deterministic preprocessing from raw data. Returns (split, stats)."""
    ratings = load_ratings(cfg["dataset"], project_path(cfg["paths"]["raw_dir"]))
    split, stats = chronological_leave_two_out(
        ratings,
        threshold=cfg["positive_rating_threshold"],
        min_positives=cfg["split"]["min_positive_interactions"],
        split_seed=cfg["split"]["split_seed"],
    )
    validate_split(split)
    stats["raw"] = {
        "users": int(ratings["user_id"].nunique()),
        "items": int(ratings["item_id"].nunique()),
        "ratings": int(len(ratings)),
    }
    return split, stats


def save_split(split, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        split[name].to_csv(out_dir / f"{name}.csv", index=False)
    meta = {k: split[k] for k in ("n_users", "n_items", "positive_threshold")}
    write_json(meta, out_dir / "meta.json")


def load_processed(cfg=None):
    """Load the saved split (run `python -m src.data` first) and re-validate it."""
    cfg = cfg or load_config()
    d = processed_dir(cfg)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{d} not found; run `python -m src.data` first.")
    with open(d / "meta.json") as f:
        split = json.load(f)
    for name in TABLES:
        split[name] = pd.read_csv(d / f"{name}.csv", dtype="int64")
    validate_split(split)
    return split


def summarize(split, stats, cfg):
    train_counts = split["train"].groupby("user").size()
    retained_items = pd.concat([split["train"], split["validation"], split["test"]])["item"].nunique()
    thr = cfg["positive_rating_threshold"]

    def describe(x):
        return {
            "mean": round(float(np.mean(x)), 4),
            "median": float(np.median(x)),
            "min": int(np.min(x)),
            "max": int(np.max(x)),
        }

    def excluded_non_positive(target):
        low = split["history"]["rating"].to_numpy() < thr
        low_items = set(zip(split["history"]["user"][low], split["history"]["item"][low]))
        return [sum((u, i) in low_items for i in items) for u, items in enumerate(previously_rated(split, target))]

    train_items = set(split["train"]["item"])
    return {
        "dataset": cfg["dataset"],
        "config": cfg,
        "raw_ratings_sha256": DATASETS[cfg["dataset"]]["ratings_sha256"],
        "original_users": stats["raw"]["users"],
        "original_items": stats["raw"]["items"],
        "original_ratings": stats["raw"]["ratings"],
        "positive_rating_threshold": thr,
        "positive_interactions": stats["positive_interactions"],
        "min_positive_interactions": cfg["split"]["min_positive_interactions"],
        "split_seed": cfg["split"]["split_seed"],
        "split_fingerprint_sha256": split_fingerprint(split),
        "users_removed": stats["users_removed"],
        "users_removed_with_zero_positives": stats["users_removed_with_zero_positives"],
        "removed_user_positive_counts": stats["removed_user_positive_counts"],
        "retained_users": split["n_users"],
        "retained_users_total_ratings": int(len(split["history"])),
        "retained_items_with_any_positive": int(retained_items),
        "items_with_train_positive": len(train_items),
        "catalog_items_ranked": split["n_items"],
        "train_interactions": int(len(split["train"])),
        "validation_interactions": int(len(split["validation"])),
        "test_interactions": int(len(split["test"])),
        "validation_items_without_train_positive": int((~split["validation"]["item"].isin(train_items)).sum()),
        "test_items_without_train_positive": int((~split["test"]["item"].isin(train_items)).sum()),
        "train_positives_per_user": describe(train_counts.to_numpy()),
        "users_with_one_train_positive": int((train_counts == 1).sum()),
        "timestamp_ties": timestamp_tie_stats(split),
        "candidates_per_user": {
            "validation": describe(candidate_counts(split, "validation")),
            "test": describe(candidate_counts(split, "test")),
        },
        "excluded_previously_rated_per_user": {
            "validation": describe([len(x) for x in previously_rated(split, "validation")]),
            "test": describe([len(x) for x in previously_rated(split, "test")]),
        },
        "excluded_non_positive_ratings_per_user": {
            "validation": describe(excluded_non_positive("validation")),
            "test": describe(excluded_non_positive("test")),
        },
    }


def write_report(summary, path):
    s = summary
    tp = s["train_positives_per_user"]
    cv, ct = s["candidates_per_user"]["validation"], s["candidates_per_user"]["test"]
    ev, et = s["excluded_previously_rated_per_user"]["validation"], s["excluded_previously_rated_per_user"]["test"]
    lv, lt = s["excluded_non_positive_ratings_per_user"]["validation"], s["excluded_non_positive_ratings_per_user"]["test"]
    ties = s["timestamp_ties"]
    title = DATASETS[s["dataset"]]["title"]
    lines = [
        f"# Dataset Report — {title}",
        "",
        "Generated by `python -m src.data` from `configs/dataset.yaml`. Do not edit by hand.",
        "",
        f"## {title} raw data",
        "",
        f"- Users: {s['original_users']}",
        f"- Items: {s['original_items']}",
        f"- Ratings: {s['original_ratings']}",
        f"- Ratings file SHA-256: `{s['raw_ratings_sha256']}`",
        "",
        "## Implicit conversion",
        "",
        f"- Threshold: rating >= {s['positive_rating_threshold']} is a positive; ratings below are *not* positives "
        "(and are not treated as explicit negatives, but they do count as *observed* for candidate filtering).",
        f"- Number of positive interactions: {s['positive_interactions']}",
        "",
        "## Filtering",
        "",
        f"- Minimum positives per user: {s['min_positive_interactions']} (1 train + 1 validation + 1 test)",
        f"- Users before: {s['original_users']}",
        f"- Users after: {s['retained_users']}",
        f"- Users removed: {s['users_removed']} "
        f"({s['users_removed_with_zero_positives']} with zero positives); positive counts of removed users: "
        f"{s['removed_user_positive_counts']}",
        f"- Items: no item filtering. All {s['catalog_items_ranked']} catalog items are rankable; "
        f"{s['retained_items_with_any_positive']} have at least one positive among retained users, "
        f"{s['items_with_train_positive']} have at least one training positive.",
        "",
        "## Final split (chronological leave-two-out)",
        "",
        f"Per-user order: timestamp, then a fixed pseudo-random tie-break "
        f"SHA-256(user_id, item_id, split_seed={s['split_seed']}). Equal timestamps carry no reliable "
        "internal chronology.",
        "",
        f"- Training positives: {s['train_interactions']}",
        f"- Validation positives: {s['validation_interactions']}",
        f"- Test positives: {s['test_interactions']}",
        f"- Split fingerprint (SHA-256 of train/validation/test/history contents): `{s['split_fingerprint_sha256']}`",
        "",
        f"- Average train interactions per user: {tp['mean']}",
        f"- Median: {tp['median']}",
        f"- Min: {tp['min']}",
        f"- Max: {tp['max']}",
        f"- Users with exactly one train positive: {s['users_with_one_train_positive']}",
        "",
        f"- Validation targets never seen as a train positive (cold items): {s['validation_items_without_train_positive']}",
        f"- Test targets never seen as a train positive (cold items): {s['test_items_without_train_positive']}",
        f"- Users whose validation timestamp equals their last train timestamp: {ties['users_with_train_val_timestamp_tie']}",
        f"- Users whose test timestamp equals their validation timestamp: {ties['users_with_val_test_timestamp_tie']}",
        "",
        "## Evaluation",
        "",
        "Full ranking over the catalog. For each held-out event, every item the user rated (any rating) "
        "strictly before that event is removed; the target is kept. Validation: ratings before the validation "
        "event. Test: ratings before the test event (includes all training-period ratings and the validation item). "
        "Ratings after the event never filter its candidates.",
        "",
        "| Per user | Validation | Test |",
        "|---|---|---|",
        f"| Candidates — average | {cv['mean']} | {ct['mean']} |",
        f"| Candidates — median | {cv['median']} | {ct['median']} |",
        f"| Candidates — min | {cv['min']} | {ct['min']} |",
        f"| Candidates — max | {cv['max']} | {ct['max']} |",
        f"| Excluded previously rated — average | {ev['mean']} | {et['mean']} |",
        f"| … of which rating < {s['positive_rating_threshold']} — average | {lv['mean']} | {lt['mean']} |",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to dataset yaml")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    split, stats = build(cfg)
    out = processed_dir(cfg)
    save_split(split, out)

    results = project_path(cfg["paths"]["results_dir"])
    summary = summarize(split, stats, cfg)
    write_json(summary, results / "data_summary.json")
    write_report(summary, results / "DATASET_REPORT.md")

    print(f"Split validated and saved to {out}")
    print(f"users {summary['retained_users']} (removed {summary['users_removed']}), "
          f"items {summary['catalog_items_ranked']}, train/val/test "
          f"{summary['train_interactions']}/{summary['validation_interactions']}/{summary['test_interactions']}")
    print(f"split fingerprint {summary['split_fingerprint_sha256']}")
    print(f"Wrote {results / 'data_summary.json'} and {results / 'DATASET_REPORT.md'}")


if __name__ == "__main__":
    main()
