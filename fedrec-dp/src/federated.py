"""B1: cross-device federated BPR-MF simulator (no differential privacy).

Local to client u (never aggregated, never sent):   p_u in R^d, u's training positives.
Global / shared (held by the server):               Q in R^{M x d}.
Score (identical to B0):                             score(u, i) = p_u . q_i

One round t:
  1. Poisson sampling: every client joins independently with probability q.
     An empty round is SKIPPED (Q unchanged), never resampled, to keep the
     sampling distribution exact for later DP accounting.
  2. Each selected client downloads Q_t (dense), copies it to Q_local, and runs
     `local_epochs` full-batch SGD steps of B0's BPR loss on its own positives,
     updating p_u and Q_local together. Fresh uniform negatives every step,
     drawn from items that are not the client's training positives (B0's rule).
  3. The client uploads the dense ΔQ_u = Q_local - Q_t and keeps p_u.
  4. The server applies equal-user FedAvg:
        Q_{t+1} = Q_t + eta_server * (1/|S_t|) * Σ_{u in S_t} ΔQ_u

Sparse rows: a client only touches the rows of its positives and sampled negatives;
all other rows of ΔQ_u are exactly zero. They add nothing, so an item nobody touched
keeps its embedding. L2 is applied only to rows used in the loss (as in B0), never to
the whole Q, so untouched items are not shrunk by regularisation. An item touched by
k of |S_t| clients moves by (k/|S_t|) × (mean of those k updates): this "dilution" is
the exact gradient of the equal-user average objective, not an artefact.
`aggregation="item_mean"` (divide each row by the number of clients that touched it)
is available only as a diagnostic: it changes the weighting and, under DP, would
require private per-item counts.

Everything is float32 (payload accounting assumes 4-byte floats).
"""

import numpy as np
import torch

from src.bpr import BPRMF

FLOAT_BYTES = 4


# --------------------------------------------------------------------------- state

def init_state(n_users, n_items, dim, init_std, seed):
    """Same initialisation as B0 for the same seed: P from user_emb, Q from item_emb."""
    torch.manual_seed(seed)
    m = BPRMF(n_users, n_items, dim, init_std)
    P = m.user_emb.weight.detach().numpy().astype(np.float32).copy()
    Q = m.item_emb.weight.detach().numpy().astype(np.float32).copy()
    return P, Q


def client_positives(train, n_users, n_items):
    """Per-client arrays of training positive items and a boolean positive mask."""
    users = train["user"].to_numpy()
    items = train["item"].to_numpy()
    mask = np.zeros((n_users, n_items), dtype=bool)
    mask[users, items] = True
    order = np.argsort(users, kind="stable")
    bounds = np.searchsorted(users[order], np.arange(n_users + 1))
    pos = [items[order[bounds[u]:bounds[u + 1]]] for u in range(n_users)]
    return pos, mask


# --------------------------------------------------------------------------- sampling

def sample_clients(rng, n_clients, q):
    """Poisson (independent Bernoulli(q)) client sampling. May return an empty array."""
    return np.flatnonzero(rng.random(n_clients) < q)


def sample_negatives(rng, pos_mask_row, n):
    """Uniform negatives among items that are not this client's training positives."""
    n_items = len(pos_mask_row)
    neg = rng.integers(0, n_items, size=n)
    bad = pos_mask_row[neg]
    while bad.any():
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
        bad = pos_mask_row[neg]
    return neg


# --------------------------------------------------------------------------- local training

def bpr_grads(p, Q, pos, neg, reg):
    """Loss and gradients of B0's per-batch BPR objective for one client's triplets.

    L = mean_k softplus(-(p.q_pos_k - p.q_neg_k))
        + reg * mean_k(||p||^2 + ||q_pos_k||^2 + ||q_neg_k||^2)
    Returns (loss, bpr_term, grad_p, grad rows for pos, grad rows for neg).
    """
    n = len(pos)
    qi, qj = Q[pos], Q[neg]
    diff = qi - qj
    x = diff @ p
    bpr = np.logaddexp(0.0, -x).mean()
    sq = (p @ p) + (qi * qi).sum(1) + (qj * qj).sum(1)
    loss = bpr + reg * sq.mean()
    sig_neg = 0.5 * (1.0 - np.tanh(0.5 * x))                     # sigmoid(-x), overflow-safe
    g = (-sig_neg / n).astype(np.float32)                        # d bpr / d x_k = -sigmoid(-x_k) / n
    grad_p = (g[:, None] * diff).sum(0) + 2.0 * reg * p
    grad_qi = g[:, None] * p[None, :] + (2.0 * reg / n) * qi
    grad_qj = -g[:, None] * p[None, :] + (2.0 * reg / n) * qj
    return float(loss), float(bpr), grad_p.astype(np.float32), grad_qi.astype(np.float32), grad_qj.astype(np.float32)


def local_update(p_u, Q_global, pos, pos_mask_row, lr, local_epochs, reg, rng):
    """Client-side training. Returns (new p_u, dense ΔQ_u, mean local loss, touched-row mask).

    Q_global is never modified; the client works on its own copy.
    """
    p = p_u.copy()
    Q_local = Q_global.copy()
    touched = np.zeros(len(Q_global), dtype=bool)
    losses = []
    for _ in range(local_epochs):
        neg = sample_negatives(rng, pos_mask_row, len(pos))
        loss, _, gp, gqi, gqj = bpr_grads(p, Q_local, pos, neg, reg)
        p = p - lr * gp
        np.add.at(Q_local, pos, -lr * gqi)
        np.add.at(Q_local, neg, -lr * gqj)
        touched[pos] = True
        touched[neg] = True
        losses.append(loss)
    return p, Q_local - Q_global, float(np.mean(losses)), touched


# --------------------------------------------------------------------------- server

def aggregate(deltas, touched=None, aggregation="fedavg"):
    """Combine client ΔQ_u (list of dense arrays). Returns the server update ΔQ_t."""
    if not deltas:
        raise ValueError("no client updates to aggregate")
    total = np.sum(deltas, axis=0, dtype=np.float32)
    if aggregation == "fedavg":                    # equal-user weighting, main B1 rule
        return total / np.float32(len(deltas))
    if aggregation == "item_mean":                 # diagnostic only
        counts = np.sum(touched, axis=0).astype(np.float32)
        return total / np.maximum(counts, 1.0)[:, None]
    raise ValueError(f"unknown aggregation {aggregation!r}")


def round_communication(n_selected, n_items, dim, touched_rows_per_client=None):
    """Simulated tensor payload (float32) for one round; dense protocol + sparse diagnostic."""
    dense = n_items * dim * FLOAT_BYTES
    out = {
        "clients": int(n_selected),
        "download_bytes": int(n_selected * dense),        # Q_t to every selected client
        "upload_bytes": int(n_selected * dense),          # dense ΔQ_u from every selected client
        "bytes_per_client": int(2 * dense),
    }
    if touched_rows_per_client is not None:               # rows (d floats + 4-byte index) if sent sparsely
        out["sparse_upload_bytes_diag"] = int(sum(r * (dim * FLOAT_BYTES + 4) for r in touched_rows_per_client))
    return out


# --------------------------------------------------------------------------- simulator

class FederatedBPR:
    """Holds global Q (server) and the per-client p_u (conceptually on each device)."""

    def __init__(self, train, n_users, n_items, cfg, seed):
        m, fl = cfg["model"], cfg["federated"]
        self.n_users, self.n_items, self.dim = n_users, n_items, m["dim"]
        self.cfg, self.seed = cfg, seed
        self.P, self.Q = init_state(n_users, n_items, m["dim"], m["init_std"], seed)
        self.pos, self.pos_mask = client_positives(train, n_users, n_items)
        self.sampling_rng = np.random.default_rng([seed, 0])
        self.client_rng = np.random.default_rng([seed, 1])
        self.q = fl["client_sampling_q"]
        self.lr = fl["local_lr"]
        self.local_epochs = fl["local_epochs"]
        self.reg = fl["l2_reg"]
        self.server_lr = fl["server_lr"]
        self.aggregation = fl.get("aggregation", "fedavg")
        self.round = 0
        self.ever_sampled = np.zeros(n_users, dtype=bool)
        self.comm_totals = {"download_bytes": 0, "upload_bytes": 0}

    def run_round(self):
        self.round += 1
        selected = sample_clients(self.sampling_rng, self.n_users, self.q)
        info = {"round": self.round, "clients": len(selected), "skipped_empty": len(selected) == 0}
        if len(selected) == 0:
            info.update(round_communication(0, self.n_items, self.dim, []))
            return info
        Q_t = self.Q                      # what every selected client downloads this round
        deltas, touched, losses = [], [], []
        for u in selected:                # sorted order -> deterministic RNG consumption
            p_new, dQ, loss, t = local_update(self.P[u], Q_t, self.pos[u], self.pos_mask[u],
                                              self.lr, self.local_epochs, self.reg, self.client_rng)
            self.P[u] = p_new             # stays on the client
            deltas.append(dQ)
            touched.append(t)
            losses.append(loss)
        update = aggregate(deltas, touched, self.aggregation)
        self.Q = Q_t + np.float32(self.server_lr) * update
        if not (np.isfinite(self.Q).all() and np.isfinite(self.P[selected]).all()):
            raise FloatingPointError(f"training diverged at round {self.round}")
        self.ever_sampled[selected] = True
        comm = round_communication(len(selected), self.n_items, self.dim, [int(t.sum()) for t in touched])
        self.comm_totals["download_bytes"] += comm["download_bytes"]
        self.comm_totals["upload_bytes"] += comm["upload_bytes"]
        n_pos = np.array([len(self.pos[u]) for u in selected])
        info.update(comm)
        info.update({
            "local_loss": float(np.mean(losses)),
            "update_norm": float(np.linalg.norm(update)),
            "mean_touched_rows": float(np.mean([t.sum() for t in touched])),
            "client_pos_mean": float(n_pos.mean()), "client_pos_min": int(n_pos.min()),
            "client_pos_max": int(n_pos.max()),
        })
        return info

    def full_scores(self, users):
        users = np.asarray(users)
        return (self.P[users] @ self.Q.T).astype(np.float64)

    def state(self):
        return {"P": self.P.copy(), "Q": self.Q.copy()}

    def load_state(self, state):
        self.P, self.Q = state["P"].copy(), state["Q"].copy()
