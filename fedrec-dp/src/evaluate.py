"""Full-ranking evaluation shared by every model.

A model is any function `score_fn(users) -> scores` returning an array of shape
(len(users), n_items); higher = more recommended. The protocol below is the only
one used for main results, for every model:

  candidates(u, event) = all catalog items
                         - every item u rated (ANY rating) strictly before the event

  validation: removes ratings before the validation event (train positives and
              earlier low ratings); the future test item stays a candidate.
  test:       removes ratings before the test event, incl. the validation item.

"Before" is the user's frozen total order (timestamp, hash tie-break) from src.data,
so ratings after the event never filter its candidates.

The single held-out item is ranked against all its candidates. Candidates with a
score exactly equal to the target's are ranked above it (pessimistic tie policy),
so a model cannot gain from constant or coarse scores.
"""

import numpy as np

from src import metrics
from src.data import previously_rated


def exclusion_lists(split, target):
    """Per-user array of item indices that are NOT candidates for `target`."""
    return previously_rated(split, target)


def target_items(split, target):
    df = split[target].sort_values("user")
    assert np.array_equal(df["user"].to_numpy(), np.arange(split["n_users"]))
    return df["item"].to_numpy()


def rank_targets(scores, targets, excluded, tie_policy="pessimistic"):
    """1-indexed rank of each user's target among that user's candidates.

    scores: (n, n_items) float array; targets: (n,) item indices;
    excluded: list of n arrays of non-candidate items.
    """
    scores = np.array(scores, dtype=np.float64)  # copy: we mask in place
    if scores.ndim != 2 or len(scores) != len(targets):
        raise ValueError("scores must have shape (n_users, n_items)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    rows = np.arange(len(targets))
    for r, ex in enumerate(excluded):
        if np.isin(targets[r], ex):
            raise ValueError(f"target item of row {r} is excluded from its own candidate set")
        scores[r, ex] = -np.inf
    target_scores = scores[rows, targets][:, None]
    higher = (scores > target_scores).sum(axis=1)
    if tie_policy == "pessimistic":
        ties = (scores == target_scores).sum(axis=1) - 1  # minus the target itself
        return 1 + higher + ties
    if tie_policy == "optimistic":  # diagnostic only, never for reported results
        return 1 + higher
    raise ValueError(f"unknown tie_policy {tie_policy!r}")


def evaluate(score_fn, split, target, k=10, tie_policy="pessimistic", batch_size=1024):
    """Evaluate a scorer on 'validation' or 'test'. Returns (summary dict, ranks)."""
    if target not in ("validation", "test"):
        raise ValueError("target must be 'validation' or 'test'")
    targets = target_items(split, target)
    excluded = exclusion_lists(split, target)
    ranks = np.empty(split["n_users"], dtype=np.int64)
    for start in range(0, split["n_users"], batch_size):
        users = np.arange(start, min(start + batch_size, split["n_users"]))
        scores = score_fn(users)
        if scores.shape != (len(users), split["n_items"]):
            raise ValueError(f"score_fn returned shape {scores.shape}, "
                             f"expected {(len(users), split['n_items'])}")
        ranks[users] = rank_targets(scores, targets[users], [excluded[u] for u in users], tie_policy)

    per_user = {
        f"ndcg@{k}": metrics.ndcg_from_rank(ranks, k),
        f"hr@{k}": metrics.hit_rate_from_rank(ranks, k),
        f"recall@{k}": metrics.recall_from_rank(ranks, k),
        f"mrr@{k}": metrics.mrr_from_rank(ranks, k),
    }
    n = len(ranks)
    summary = {"split": target, "n_users": n}
    for name, values in per_user.items():
        summary[name] = float(values.mean())
        summary[f"{name}_se"] = float(values.std(ddof=1) / np.sqrt(n))
    summary["mean_rank"] = float(ranks.mean())
    summary["median_rank"] = float(np.median(ranks))
    return summary, ranks
