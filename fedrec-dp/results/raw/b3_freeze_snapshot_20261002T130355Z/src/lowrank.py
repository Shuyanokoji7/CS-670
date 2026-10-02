"""B3/B4: federated BPR with a LOW-RANK shared item matrix Q = A B (A: M x r, B: r x d).

Objective: exactly B1's per-client BPR loss and regulariser on p_u and the *effective* touched item rows q_i = A_i B
(src.federated.bpr_grads). Its gradient g_Q = dL/dq (which already contains B1's L2 term on the touched q rows) is
chained through the factors:  dL/dA_i = g_Q,i B^T  (row-sparse: touched items only),  dL/dB = sum_i A_i^T g_Q,i (dense).
There is NO extra factor-only ||A|| / ||B|| penalty. No L2 term is applied directly to unseen effective rows; their
scores can still change locally and globally through the shared B (by design). p_u stays local (never clipped / uploaded).

Balancing: A, B are re-parameterised by QR + core SVD so that A B is unchanged (to float tolerance) and the
factors are balanced (A^T A = B B^T = S). Done once at initialisation and after every server step; in B4 this is
post-processing of the released noisy A, B.

Round t (B3 = no privacy block; B4 = cfg["privacy"]):
  B3: Poisson sampling; each selected client trains locally and uploads (dA_u, dB_u); server applies
      equal-user FedAvg / realised |S_t| (empty rounds skipped); balance.
  B4: one whole-client L2 clip of the concatenated [vec dA_u, vec dB_u] to C; sum; + N(0, sigma^2 C^2 I) on ALL
      D_r = r (M + d) shared coordinates; / fixed q N; * eta_s; balance. Empty rounds are noised and accounted.
With privacy {clip_norm: None, noise_multiplier: 0, denominator: realised} B4 reproduces B3 bit-for-bit (tested).
"""

import numpy as np

from src.federated import (FLOAT_BYTES, bpr_grads, client_positives, init_state, sample_clients,
                           sample_negatives)
from src.privacy import clip_update, gaussian_noise, private_update, update_norm


# --------------------------------------------------------------------------- factors

def init_scale(rank, target_var=1e-4):
    """Per-entry std s_r with Var(q_ij) = r * s_r^4 = target_var (B1/B2's N(0, 0.01^2) for q)."""
    return (target_var / rank) ** 0.25


def balance(A, B):
    """Return (A', B') with A' B' == A B (float tolerance) and A'^T A' == B' B'^T (balanced gauge)."""
    A64, B64 = A.astype(np.float64), B.astype(np.float64)
    Qa, Ra = np.linalg.qr(A64)                 # A = Qa Ra
    Qb, Rb = np.linalg.qr(B64.T)               # B^T = Qb Rb  ->  B = Rb^T Qb^T
    U, S, Vt = np.linalg.svd(Ra @ Rb.T)        # core K = Ra Rb^T = U S V^T
    root = np.sqrt(S)
    A2 = (Qa @ U) * root[None, :]
    B2 = (root[:, None] * Vt) @ Qb.T
    return A2.astype(np.float32), B2.astype(np.float32)


def init_lowrank_state(n_users, n_items, dim, rank, seed, init_std_p=0.01):
    """P exactly as B1/B2 for the same seed; A, B ~ N(0, s_r^2) from their own stream, then balanced."""
    P, _ = init_state(n_users, n_items, dim, init_std_p, seed)
    rng = np.random.default_rng([seed, 3])
    s = init_scale(rank)
    A = (rng.standard_normal((n_items, rank)) * s).astype(np.float32)
    B = (rng.standard_normal((rank, dim)) * s).astype(np.float32)
    A, B = balance(A, B)
    return P, A, B


def shared_coordinates(n_items, dim, rank):
    return rank * (n_items + dim)


# --------------------------------------------------------------------------- local training

def lowrank_local_update(p_u, A, B, pos, pos_mask_row, lr, local_epochs, reg, rng):
    """Client-side training of (p_u, A, B) on B1's objective. Returns (p, dA, dB, mean loss, touched-row mask).

    A and B (global) are not modified. Each step uses q_i = A_i B for the touched rows only.
    """
    p = p_u.copy()
    A_loc, B_loc = A.copy(), B.copy()
    touched = np.zeros(len(A), dtype=bool)
    losses = []
    for _ in range(local_epochs):
        neg = sample_negatives(rng, pos_mask_row, len(pos))
        items = np.unique(np.concatenate([pos, neg]))
        Qs = A_loc[items] @ B_loc                               # effective rows used by this step
        lp, ln = np.searchsorted(items, pos), np.searchsorted(items, neg)
        loss, _, gp, gqi, gqj = bpr_grads(p, Qs, lp, ln, reg)   # B1's loss + L2 on p and effective q rows
        gQ = np.zeros_like(Qs)
        np.add.at(gQ, lp, gqi)
        np.add.at(gQ, ln, gqj)
        grad_A_rows = gQ @ B_loc.T                              # dL/dA_i = g_Q,i B^T
        grad_B = A_loc[items].T @ gQ                            # dL/dB = sum_i A_i^T g_Q,i
        p = p - lr * gp
        A_loc[items] -= lr * grad_A_rows
        B_loc = B_loc - lr * grad_B
        touched[items] = True
        losses.append(loss)
    return p, A_loc - A, B_loc - B, float(np.mean(losses)), touched


def lowrank_communication(n_selected, n_items, dim, rank):
    dense = shared_coordinates(n_items, dim, rank) * FLOAT_BYTES
    return {"clients": int(n_selected), "download_bytes": int(n_selected * dense),
            "upload_bytes": int(n_selected * dense), "bytes_per_client": int(2 * dense)}


# --------------------------------------------------------------------------- simulator

class LowRankFederatedBPR:
    """B3 (no cfg["privacy"]) or B4 (with cfg["privacy"]). Interface mirrors FederatedBPR / DPFederatedBPR."""

    def __init__(self, train, n_users, n_items, cfg, seed):
        m, fl = cfg["model"], cfg["federated"]
        self.n_users, self.n_items, self.dim, self.rank = n_users, n_items, m["dim"], int(m["rank"])
        self.cfg, self.seed = cfg, seed
        self.P, self.A, self.B = init_lowrank_state(n_users, n_items, m["dim"], self.rank, seed, m["init_std"])
        self.pos, self.pos_mask = client_positives(train, n_users, n_items)
        self.sampling_rng = np.random.default_rng([seed, 0])
        self.client_rng = np.random.default_rng([seed, 1])
        self.q = fl["client_sampling_q"]
        self.lr, self.local_epochs, self.reg = fl["local_lr"], fl["local_epochs"], fl["l2_reg"]
        self.server_lr = fl["server_lr"]
        self.round = 0
        self.ever_sampled = np.zeros(n_users, dtype=bool)
        self.comm_totals = {"download_bytes": 0, "upload_bytes": 0}
        pc = cfg.get("privacy")
        self.private = pc is not None
        if self.private:
            self.C = float(pc["clip_norm"]) if pc.get("clip_norm") is not None else float("inf")
            self.sigma = float(pc["noise_multiplier"])
            self.denominator_mode = pc["denominator"]
            if self.sigma > 0 and not np.isfinite(self.C):
                raise ValueError("Gaussian noise requires a finite clipping norm C")
            self.noise_seed = int(pc.get("noise_seed", seed))
            self.noise_rng = np.random.default_rng([self.noise_seed, 2])
        else:
            self.C, self.sigma, self.denominator_mode, self.noise_seed = float("inf"), 0.0, "realised", None
        self.expected_clients = self.q * n_users

    @property
    def D(self):
        return shared_coordinates(self.n_items, self.dim, self.rank)

    @property
    def Q(self):
        return self.A @ self.B

    def _flat(self, dA, dB):
        return np.concatenate([dA.ravel(), dB.ravel()])

    def _split(self, v):
        k = self.n_items * self.rank
        return v[:k].reshape(self.n_items, self.rank), v[k:].reshape(self.rank, self.dim)

    def run_round(self):
        self.round += 1
        selected = sample_clients(self.sampling_rng, self.n_users, self.q)
        info = {"round": self.round, "clients": len(selected), "skipped_empty": False}
        A_t, B_t = self.A, self.B
        flats, norms, factors, touched, losses = [], [], [], [], []
        for u in selected:
            p_new, dA, dB, loss, t = lowrank_local_update(self.P[u], A_t, B_t, self.pos[u], self.pos_mask[u],
                                                          self.lr, self.local_epochs, self.reg, self.client_rng)
            self.P[u] = p_new                                   # local; never clipped, never uploaded
            v, n, f = clip_update(self._flat(dA, dB), self.C)   # ONE norm over [dA, dB]; identity if C = inf
            flats.append(v)
            norms.append(n)
            factors.append(f)
            touched.append(t)
            losses.append(loss)
        if len(selected) == 0 and self.sigma == 0 and self.denominator_mode == "realised":
            info["skipped_empty"] = True                        # B1/B3 semantics
            info.update(lowrank_communication(0, self.n_items, self.dim, self.rank))
            return info
        denom = self.expected_clients if self.denominator_mode == "expected" else len(selected)
        update, total, noise = private_update(flats, (self.D,), self.sigma, self.C, self.noise_rng
                                              if self.private else None, denom)
        uA, uB = self._split(update)
        A_new = A_t + np.float32(self.server_lr) * uA
        B_new = B_t + np.float32(self.server_lr) * uB
        if not (np.isfinite(A_new).all() and np.isfinite(B_new).all() and np.isfinite(self.P[selected]).all()):
            raise FloatingPointError(f"training diverged at round {self.round}")
        self.A, self.B = balance(A_new, B_new)                  # post-processing; AB unchanged
        self.ever_sampled[selected] = True
        comm = lowrank_communication(len(selected), self.n_items, self.dim, self.rank)
        self.comm_totals["download_bytes"] += comm["download_bytes"]
        self.comm_totals["upload_bytes"] += comm["upload_bytes"]
        info.update(comm)
        na = np.array(norms) if norms else np.array([np.nan])
        agg_norm, noise_norm = update_norm(total), update_norm(noise)
        info.update({
            "local_loss": float(np.mean(losses)) if losses else np.nan,
            "update_norm": update_norm(update), "denominator": float(denom),
            "pre_clip_norm_mean": float(np.nanmean(na)), "pre_clip_norm_median": float(np.nanmedian(na)),
            "pre_clip_norm_p90": float(np.nanpercentile(na, 90)), "pre_clip_norm_p95": float(np.nanpercentile(na, 95)),
            "pre_clip_norm_max": float(np.nanmax(na)),
            "frac_clipped": float(np.mean([f < 1.0 for f in factors])) if factors else np.nan,
            "post_clip_norm_mean": float(np.mean([n * f for n, f in zip(norms, factors)])) if norms else np.nan,
            "shrinkage_mean": float(np.mean(factors)) if factors else np.nan,
            "noise_std": float(self.sigma * self.C) if self.sigma > 0 else 0.0,
            "noise_norm": noise_norm, "clipped_aggregate_norm": agg_norm,
            "signal_to_noise": agg_norm / noise_norm if noise_norm > 0 else np.inf,
            "A_fro": float(np.linalg.norm(self.A)), "B_fro": float(np.linalg.norm(self.B)),
            "mean_touched_rows": float(np.mean([t.sum() for t in touched])) if touched else 0.0,
            "client_norms": norms,
        })
        return info

    def full_scores(self, users):
        users = np.asarray(users)
        return (self.P[users] @ self.Q.T).astype(np.float64)

    def state(self):
        return {"P": self.P.copy(), "A": self.A.copy(), "B": self.B.copy()}

    def load_state(self, state):
        self.P, self.A, self.B = state["P"].copy(), state["A"].copy(), state["B"].copy()
