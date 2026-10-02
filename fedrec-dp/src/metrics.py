"""Top-K ranking metrics.

Two forms are provided:
  * rank-based, vectorised (`*_from_rank`): for the main protocol with exactly one
    held-out relevant item per user. `rank` is 1-indexed.
  * list-based, generic (`*_at_k`): a ranked list of item ids and a set of relevant
    items; supports any number of relevant items.

With exactly one relevant item, Recall@K == HR@K by definition. They must not be
reported as two independent pieces of evidence.
"""

import numpy as np


# ------------------------------------------------------------ single relevant item

def ndcg_from_rank(rank, k=10):
    """1 / log2(rank + 1) if rank <= k else 0 (IDCG = 1 for one relevant item)."""
    rank = _as_ranks(rank)
    return np.where(rank <= k, 1.0 / np.log2(rank + 1.0), 0.0)


def hit_rate_from_rank(rank, k=10):
    rank = _as_ranks(rank)
    return (rank <= k).astype(float)


def recall_from_rank(rank, k=10):
    """Recall@k with one relevant item; identical to HR@k."""
    return hit_rate_from_rank(rank, k)


def mrr_from_rank(rank, k=10):
    rank = _as_ranks(rank)
    return np.where(rank <= k, 1.0 / rank, 0.0)


def _as_ranks(rank):
    rank = np.asarray(rank, dtype=float)
    if np.any(rank < 1) or np.any(np.isnan(rank)):
        raise ValueError("ranks are 1-indexed and must be >= 1")
    return rank


# ------------------------------------------------------------ generic, any #relevant

def recall_at_k(ranked_items, relevant, k=10):
    """|top-k ∩ relevant| / |relevant|."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("recall is undefined with no relevant items")
    return len(relevant.intersection(list(ranked_items)[:k])) / len(relevant)


def hit_rate_at_k(ranked_items, relevant, k=10):
    """1 if any relevant item is in the top-k."""
    return float(bool(set(relevant).intersection(list(ranked_items)[:k])))


def ndcg_at_k(ranked_items, relevant, k=10):
    """Binary-relevance NDCG@k; IDCG places min(|relevant|, k) hits at the top."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("NDCG is undefined with no relevant items")
    top = list(ranked_items)[:k]
    dcg = sum(1.0 / np.log2(i + 2.0) for i, item in enumerate(top) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(min(len(relevant), k)))
    return dcg / idcg


def mrr_at_k(ranked_items, relevant, k=10):
    """Reciprocal rank of the first relevant item within the top-k, else 0."""
    relevant = set(relevant)
    for i, item in enumerate(list(ranked_items)[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0
