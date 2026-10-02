"""E1 extension: effective Gaussian noise, fixed-public-B BPR and DP popularity.

Frozen B0–B4 modules are reused without modification. Diagnostics contain private
local state and noiseless aggregates: they are research artifacts, not DP releases.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from src.federated import (bpr_grads, client_positives, init_state, local_update,
                           sample_clients, sample_negatives)
from src.lowrank import balance, lowrank_local_update
from src.privacy import clip_update, private_update, update_norm


def public_basis(dim, rank, seed=314159):
    """Nested, data-independent orthonormal rows; signs fixed for reproducibility."""
    if not 1 <= rank <= dim:
        raise ValueError("rank must be between 1 and dim")
    orth, triangular = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    orth *= np.where(np.diag(triangular) < 0, -1, 1)[None, :]
    return orth[:, :rank].T.astype(np.float32).copy()


def fixed_local_update(p_u, A, B, pos, pos_mask_row, lr, local_epochs, reg, rng):
    """Train p and A simultaneously on the existing effective-row BPR loss."""
    p, Al = p_u.copy(), A.copy()
    losses = []
    for _ in range(local_epochs):
        neg = sample_negatives(rng, pos_mask_row, len(pos))
        items = np.unique(np.concatenate([pos, neg]))
        lp, ln = np.searchsorted(items, pos), np.searchsorted(items, neg)
        loss, _, gp, gi, gj = bpr_grads(p, Al[items] @ B, lp, ln, reg)
        gq = np.zeros((len(items), B.shape[1]), dtype=np.float32)
        np.add.at(gq, lp, gi)
        np.add.at(gq, ln, gj)
        ga = gq @ B.T
        p -= np.float32(lr) * gp
        Al[items] -= np.float32(lr) * ga
        losses.append(loss)
    return p, Al - A, float(np.mean(losses))


def expected_energy(A, B, P, tau, method="two"):
    """Conditional E||delta Q||² and E||P delta Qᵀ||² for independent iid noise.

    A/B/P are held fixed *after* the signal/local updates; tau includes server
    learning rate and denominator. Cross expectations vanish, not realized terms.
    """
    A, B, P = (np.asarray(v, dtype=np.float64) for v in (A, B, P))
    M, r = A.shape
    d = B.shape[1]
    a2, b2, p2 = np.sum(A*A), np.sum(B*B), np.sum(P*P)
    pb2 = np.sum((P @ B.T)**2)
    q_a, s_a = tau**2 * M * b2, tau**2 * M * pb2
    q_b = tau**2 * d * a2 if method == "two" else 0.0
    s_b = tau**2 * a2 * p2 if method == "two" else 0.0
    q_cross = M*d*r*tau**4 if method == "two" else 0.0
    s_cross = M*r*tau**4*p2 if method == "two" else 0.0
    return {"q_a": float(q_a), "q_b": float(q_b), "q_cross": float(q_cross),
            "q_total": float(q_a+q_b+q_cross), "score_a": float(s_a),
            "score_b": float(s_b), "score_cross": float(s_cross),
            "score_total": float(s_a+s_b+s_cross)}


def perturbation_terms(A, B, ZA, ZB=None):
    """Noise-only shock around signal-updated factors, calculated in float64."""
    A, B, ZA = (np.asarray(v, dtype=np.float64) for v in (A, B, ZA))
    first = ZA @ B
    if ZB is None:
        second, cross = np.zeros_like(first), np.zeros_like(first)
    else:
        ZB = np.asarray(ZB, dtype=np.float64)
        second, cross = A @ ZB, ZA @ ZB
    return first, second, cross


def gaussian_delta(epsilon, sigma):
    """Exact delta of the Gaussian mechanism with L2 sensitivity 1, std sigma."""
    if epsilon <= 0 or sigma <= 0:
        raise ValueError("epsilon and sigma must be positive")
    return float(ndtr(1/(2*sigma)-epsilon*sigma)
                 - np.exp(epsilon)*ndtr(-1/(2*sigma)-epsilon*sigma))


def analytic_gaussian_sigma(epsilon, delta):
    if not 0 < delta < 1 or epsilon <= 0:
        raise ValueError("need epsilon > 0 and 0 < delta < 1")
    return float(brentq(lambda s: gaussian_delta(epsilon, s)-delta, .001, 100., xtol=1e-12))


def bounded_popularity(train, n_users, n_items, bound=np.sqrt(20)):
    """Unique binary item vectors, clipped in L2 per entire user, summed in float64."""
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("bound must be finite and positive")
    pos, _ = client_positives(train.drop_duplicates(["user", "item"]), n_users, n_items)
    scores = np.zeros(n_items, dtype=np.float64)
    for items in pos:
        if len(items):
            scores[items] += min(1., bound/np.sqrt(len(items)))
    return scores


class EffectiveNoiseBPR:
    """E1 Full/Two/FixedB with fixed-qN aggregation and exact instantaneous shocks."""

    def __init__(self, train, n_users, n_items, settings):
        c = settings
        self.settings = c
        self.method, self.rank, self.dim = c["method"], c["rank"], c["dim"]
        self.n_users, self.n_items = n_users, n_items
        self.P, Q = init_state(n_users, n_items, self.dim, c["init_std"], c["seed"])
        if self.method == "full":
            self._Q = Q
        elif self.method in ("two", "fixed"):
            self.B = public_basis(self.dim, self.rank, c["public_basis_seed"])
            rng = np.random.default_rng([c["seed"], 3])
            self.A = (rng.standard_normal((n_items, self.rank))*c["init_std"]
                      *np.sqrt(self.dim/self.rank)).astype(np.float32)
            if self.method == "two":
                self.A, self.B = balance(self.A, self.B)
        else:
            raise ValueError("method must be full, two or fixed")
        self.pos, self.pos_mask = client_positives(train, n_users, n_items)
        self.sampling_rng = np.random.default_rng([c["seed"], 0])
        self.client_rng = np.random.default_rng([c["seed"], 1])
        self.noise_rng = np.random.default_rng([c["seed"], 2])
        self.round, self.total_bytes = 0, 0
        self.C = float("inf") if c["clip_norm"] is None else c["clip_norm"]
        if c["sigma"] > 0 and (not np.isfinite(self.C) or self.C <= 0):
            raise ValueError("noise requires a finite positive clip norm")
        if not 0 < c["q"] <= 1:
            raise ValueError("q must be in (0, 1]")
        self.groups = np.array([0 if len(x) <= 22 else 1 if len(x) <= 61 else 2 for x in self.pos])
        self.cold = self.pos_mask.sum(axis=0) == 0
        rng = np.random.default_rng(271828)
        self.pairs = rng.integers(0, n_items, size=(c["pair_count"], 2))

    @property
    def Q(self):
        return self._Q if self.method == "full" else self.A @ self.B

    @property
    def D(self):
        return self.n_items*self.dim if self.method == "full" else (
            self.n_items*self.rank + (self.rank*self.dim if self.method == "two" else 0))

    def full_scores(self, users):
        return (self.P[np.asarray(users)] @ self.Q.T).astype(np.float64)

    def state(self):
        return {"P": self.P.copy(), **({"Q": self.Q.copy()} if self.method == "full"
                                      else {"A": self.A.copy(), "B": self.B.copy()})}

    def load_state(self, state):
        self.P = state["P"].copy()
        if self.method == "full":
            self._Q = state["Q"].copy()
        else:
            self.A, self.B = state["A"].copy(), state["B"].copy()
            if self.method == "fixed" and not np.array_equal(
                    self.B, public_basis(self.dim, self.rank, self.settings["public_basis_seed"])):
                raise ValueError("fixed public B changed")

    def run_round(self):
        c = self.settings
        self.round += 1
        selected = sample_clients(self.sampling_rng, self.n_users, c["q"])
        Q0 = self.Q.astype(np.float64)
        flats, norms, factors, losses = [], [], [], []
        for u in selected:
            args = (self.pos[u], self.pos_mask[u], c["local_lr"], c["local_epochs"], c["reg"], self.client_rng)
            if self.method == "full":
                p, delta, loss, _ = local_update(self.P[u], self._Q, *args)
                v = delta.ravel()
            elif self.method == "two":
                p, da, db, loss, _ = lowrank_local_update(self.P[u], self.A, self.B, *args)
                v = np.concatenate([da.ravel(), db.ravel()])
            else:
                p, da, loss = fixed_local_update(self.P[u], self.A, self.B, *args)
                v = da.ravel()
            self.P[u] = p
            clipped, norm, factor = clip_update(v, self.C)
            if not np.isfinite(clipped).all() or not np.isfinite(p).all():
                raise FloatingPointError(f"non-finite local state at round {self.round}")
            flats.append(clipped)
            norms.append(norm)
            factors.append(factor)
            losses.append(loss)
        denom = c["q"]*self.n_users
        _, total, noise = private_update(flats, (self.D,), c["sigma"], self.C, self.noise_rng, denom)
        step = np.float32(c["server_lr"])/np.float32(denom)
        # Preserve the B2/B4 arithmetic order for the actual noisy update.
        noisy_step = np.float32(c["server_lr"])*((total+noise)/np.float32(denom))
        signal_step = np.float32(c["server_lr"])*(total/np.float32(denom))
        Z = noise.astype(np.float64)*float(step)
        tau = c["server_lr"]*c["sigma"]*(self.C if c["sigma"] else 0)/denom
        if self.method == "full":
            Qs = (self._Q+signal_step.reshape(self._Q.shape)).astype(np.float64)
            self._Q = self._Q+noisy_step.reshape(self._Q.shape)
            first = Z.reshape(self._Q.shape)
            second, cross = np.zeros_like(first), np.zeros_like(first)
            expected = {"q_a": self.D*tau**2, "q_b": 0., "q_cross": 0., "q_total": self.D*tau**2,
                        "score_total": self.n_items*tau**2*float(np.sum(self.P.astype(np.float64)**2))}
        else:
            boundary = self.A.size
            As = self.A+signal_step[:boundary].reshape(self.A.shape)
            Bs = self.B+signal_step[boundary:].reshape(self.B.shape) if self.method == "two" else self.B
            Qs = As.astype(np.float64) @ Bs.astype(np.float64)
            ZA = Z[:boundary].reshape(self.A.shape)
            ZB = Z[boundary:].reshape(self.B.shape) if self.method == "two" else None
            first, second, cross = perturbation_terms(As, Bs, ZA, ZB)
            expected = expected_energy(As, Bs, self.P, tau, self.method)
            An = self.A+noisy_step[:boundary].reshape(self.A.shape)
            Bn = self.B+noisy_step[boundary:].reshape(self.B.shape) if self.method == "two" else self.B
            if not np.isfinite(An).all() or not np.isfinite(Bn).all():
                raise FloatingPointError(f"non-finite server factors at round {self.round}")
            self.A, self.B = balance(An, Bn) if self.method == "two" else (An, Bn)
        Qn = self.Q.astype(np.float64)
        if not np.isfinite(Qn).all():
            raise FloatingPointError(f"non-finite Q at round {self.round}")
        dq = first+second+cross
        self.total_bytes += 8*self.D*len(selected)
        info = {"round": self.round, "clients": len(selected), "D": self.D,
                "total_bytes": self.total_bytes, "tau": tau,
                "pre_clip_norm_median": float(np.median(norms)) if norms else np.nan,
                "frac_clipped": float(np.mean(np.array(factors) < 1.)) if factors else np.nan,
                "local_loss": float(np.mean(losses)) if losses else np.nan,
                "coordinate_noise_norm": update_norm(Z),
                "effective_signal_norm": update_norm(Qs-Q0),
                "q_noise_a_energy": float(np.sum(first**2)),
                "q_noise_b_energy": float(np.sum(second**2)),
                "q_noise_cross_energy": float(np.sum(cross**2)),
                "q_noise_energy": float(np.sum(dq**2)),
                "q_noise_expected_energy": expected["q_total"],
                "q_noise_expected_cross_energy": expected["q_cross"],
                "float32_residual_norm": update_norm(Qn-Qs-dq),
                "q_noise_cold_row_rms": float(np.sqrt(np.mean(dq[self.cold]**2))) if self.cold.any() else np.nan,
                "mean_user_norm": float(np.linalg.norm(self.P, axis=1).mean())}
        if self.round in c["score_rounds"]:
            P = self.P.astype(np.float64)
            ds, ss = P @ dq.T, P @ Qs.T
            pair_shock = ds[:, self.pairs[:, 0]]-ds[:, self.pairs[:, 1]]
            pair_signal = ss[:, self.pairs[:, 0]]-ss[:, self.pairs[:, 1]]
            info.update(score_noise_energy=float(np.sum(ds**2)),
                        score_noise_expected_energy=expected["score_total"],
                        score_noise_rmse=float(np.sqrt(np.mean(ds**2))),
                        score_noise_relative=float(np.linalg.norm(ds)/max(np.linalg.norm(ss), 1e-30)),
                        margin_noise_rmse=float(np.sqrt(np.mean(pair_shock**2))),
                        margin_sign_flip_fraction=float(np.mean((pair_signal > 0) != (pair_signal+pair_shock > 0))))
            for group, name in enumerate(("low", "medium", "high")):
                mask = self.groups == group
                info[f"score_noise_rmse_{name}"] = float(np.sqrt(np.mean(ds[mask]**2))) if mask.any() else np.nan
        return info
