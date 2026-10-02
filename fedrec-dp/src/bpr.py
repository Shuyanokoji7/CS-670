"""BPR matrix factorisation: model, loss and negative sampler (B0).

score(u, i) = p_u . q_i          (no biases, no side information)
loss        = mean_b softplus(-(s_ui+ - s_ui-))
              + reg * mean_b (||p_u||^2 + ||q_i+||^2 + ||q_i-||^2)

softplus(-x) == -log(sigmoid(x)), computed stably.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim, init_std=0.01):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=init_std)
        nn.init.normal_(self.item_emb.weight, std=init_std)

    def score(self, users, items):
        """Scores for aligned (user, item) pairs."""
        return (self.user_emb(users) * self.item_emb(items)).sum(-1)

    def forward(self, users, pos_items, neg_items):
        p = self.user_emb(users)
        q_pos = self.item_emb(pos_items)
        q_neg = self.item_emb(neg_items)
        pos_scores = (p * q_pos).sum(-1)
        neg_scores = (p * q_neg).sum(-1)
        sq_norms = p.pow(2).sum(-1) + q_pos.pow(2).sum(-1) + q_neg.pow(2).sum(-1)
        return pos_scores, neg_scores, sq_norms

    @torch.no_grad()
    def full_scores(self, users):
        """(len(users), n_items) score matrix as float64 numpy, for src.evaluate."""
        users = torch.as_tensor(np.asarray(users), dtype=torch.long)
        return (self.user_emb(users) @ self.item_emb.weight.T).double().numpy()


def bpr_loss(pos_scores, neg_scores, sq_norms, reg):
    """Returns (total loss, pure BPR term) — both batch means."""
    bpr = F.softplus(neg_scores - pos_scores).mean()   # = -logsigmoid(pos - neg)
    return bpr + reg * sq_norms.mean(), bpr


class NegativeSampler:
    """Uniform negatives: any catalog item that is NOT a TRAIN positive of the user.

    Built from the training positives only. It never sees validation/test targets
    or low (< threshold) ratings, so those items are sampled like any other
    non-positive item (their labels are never used).
    """

    def __init__(self, train_users, train_items, n_users, n_items, rng):
        self.n_items = n_items
        self.rng = rng
        self.positive = np.zeros((n_users, n_items), dtype=bool)
        self.positive[train_users, train_items] = True
        full = self.positive.all(axis=1)
        if full.any():
            raise ValueError(f"users {np.flatnonzero(full)} have no possible negative item")

    def sample(self, users):
        neg = self.rng.integers(0, self.n_items, size=len(users))
        bad = self.positive[users, neg]
        while bad.any():  # rejection sampling; positives are ~3% of the catalog on average
            neg[bad] = self.rng.integers(0, self.n_items, size=int(bad.sum()))
            bad = self.positive[users, neg]
        return neg
