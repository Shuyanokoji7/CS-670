"""B2: user-level DP for the federated BPR simulator (clipping, Gaussian aggregate noise, accounting).

Privacy unit: one complete user/client. Neighbouring datasets differ by adding or removing one user
together with all of their interactions (add/remove-one-user adjacency).

Per private round t (B1's round, plus the four marked steps):
    S_t      = {u : U_u < q}                          Poisson sampling (empty rounds NOT resampled)
    ΔQ_u     = Q_u,local − Q_t                        B1's local training, unchanged
  * ΔQbar_u  = ΔQ_u · min(1, C / ||ΔQ_u||_F)          one Frobenius norm over the whole shared update
  * Σ_t      = Σ_{u∈S_t} ΔQbar_u                       (secure aggregation ASSUMED: server sees only Σ_t)
  * Σ̃_t      = Σ_t + Z_t,   Z_t ~ N(0, σ²C² I_D)       D = n_items × d coordinates
  * Q_{t+1}  = Q_t + η_s · Σ̃_t / (q N)                fixed denominator: post-processing of Σ̃_t
p_u is local, never clipped, never uploaded.

Adding or removing one user changes Σ_t by at most C in L2 norm, so each round is a Poisson-subsampled
Gaussian mechanism with noise multiplier σ and sampling rate q. T rounds compose T such mechanisms,
which is exactly what the Opacus RDP/PRV accountants assume (history = [(σ, q, T)]).

The noise RNG is a seeded numpy PCG64 stream: reproducible for research, NOT cryptographically secure.
"""

import numpy as np

from src.federated import FederatedBPR, local_update, round_communication, sample_clients

ACCOUNTANTS = ("prv", "rdp")


# --------------------------------------------------------------------------- mechanism pieces

def update_norm(delta):
    """Frobenius norm of one client's complete shared update (computed in float64)."""
    return float(np.sqrt(np.sum(np.square(delta, dtype=np.float64))))


def clip_update(delta, C):
    """Scale the WHOLE update so that its Frobenius norm is <= C. Returns (clipped, norm, factor)."""
    norm = update_norm(delta)
    if C is None or not np.isfinite(C) or norm <= C:
        return delta, norm, 1.0
    factor = (C / norm) * (1.0 - 1e-6)          # small margin: float32 rounding can never push ||.|| above C
    clipped = (delta * np.float32(factor)).astype(np.float32)
    if update_norm(clipped) > C:
        raise FloatingPointError("clipped update exceeds C; sensitivity bound would be violated")
    return clipped, norm, factor


def gaussian_noise(rng, shape, sigma, C):
    """Z ~ N(0, (sigma*C)^2 I) for the clipped SUM (not divided by the number of clients)."""
    if sigma == 0:
        return np.zeros(shape, dtype=np.float32)
    return (rng.standard_normal(shape, dtype=np.float32) * np.float32(sigma * C)).astype(np.float32)


def private_update(clipped, shape, sigma, C, noise_rng, denominator):
    """(Σ clipped + Z) / denominator. `clipped` may be empty (empty Poisson round)."""
    total = np.sum(clipped, axis=0, dtype=np.float32) if clipped else np.zeros(shape, dtype=np.float32)
    noise = gaussian_noise(noise_rng, shape, sigma, C)
    return (total + noise) / np.float32(denominator), total, noise


# --------------------------------------------------------------------------- accounting

def compute_epsilon(sigma, q, T, delta, accountant="prv"):
    """User-level epsilon of T Poisson-subsampled Gaussian rounds. sigma = 0 -> inf (no DP)."""
    if sigma <= 0:
        return float("inf")
    from opacus.accountants import PRVAccountant, RDPAccountant
    acc = {"prv": PRVAccountant, "rdp": RDPAccountant}[accountant]()
    acc.history = [(float(sigma), float(q), int(T))]
    return float(acc.get_epsilon(delta=delta))


def solve_sigma(target_eps, q, T, delta, accountant="prv", tol=0.01, lo=0.3, hi=100.0):
    """Smallest noise multiplier with epsilon <= target_eps (bisection on the monotone eps(sigma)).

    Returns (sigma, achieved_eps). Achieved eps is <= target and within `tol` (relative) of it.
    """
    if compute_epsilon(hi, q, T, delta, accountant) > target_eps:
        raise ValueError(f"target eps {target_eps} not reachable with sigma <= {hi}")
    while compute_epsilon(lo, q, T, delta, accountant) <= target_eps:
        lo /= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if compute_epsilon(mid, q, T, delta, accountant) > target_eps:
            lo = mid
        else:
            hi = mid
        eps_hi = compute_epsilon(hi, q, T, delta, accountant)
        if target_eps - eps_hi <= tol * target_eps:
            return hi, eps_hi
    return hi, compute_epsilon(hi, q, T, delta, accountant)


def privacy_record(privacy_cfg, q, T, primary="prv", cross_check="rdp"):
    """Everything needed to state the guarantee of one run. C does not enter epsilon (sigma is relative to C)."""
    sigma = float(privacy_cfg["noise_multiplier"])
    delta = float(privacy_cfg["delta"])
    rec = {"dp": sigma > 0, "q": q, "T": int(T), "clip_norm": privacy_cfg.get("clip_norm"),
           "noise_multiplier": sigma, "delta": delta, "target_epsilon": privacy_cfg.get("target_epsilon"),
           "accountant": primary, "epsilon": compute_epsilon(sigma, q, T, delta, primary),
           "cross_check_accountant": cross_check,
           "epsilon_cross_check": compute_epsilon(sigma, q, T, delta, cross_check),
           "denominator": privacy_cfg["denominator"]}
    return rec


# --------------------------------------------------------------------------- simulator

class DPFederatedBPR(FederatedBPR):
    """B1's simulator with whole-client clipping, Gaussian aggregate noise and a fixed denominator.

    privacy config (cfg["privacy"]):
      clip_norm C (None/inf = no clipping), noise_multiplier sigma (0 = no noise),
      denominator "expected" (q*N, B2) or "realised" (|S_t|, B1), noise_seed_offset.
    With C=inf, sigma=0 and denominator="realised" this reproduces B1 bit-for-bit (tested).
    """

    def __init__(self, train, n_users, n_items, cfg, seed):
        super().__init__(train, n_users, n_items, cfg, seed)
        pc = cfg["privacy"]
        self.C = float(pc["clip_norm"]) if pc.get("clip_norm") is not None else float("inf")
        self.sigma = float(pc["noise_multiplier"])
        self.denominator_mode = pc["denominator"]
        if self.sigma > 0 and not np.isfinite(self.C):
            raise ValueError("Gaussian noise requires a finite clipping norm C")
        self.noise_seed = int(pc.get("noise_seed", seed))
        self.noise_rng = np.random.default_rng([self.noise_seed, 2])   # own stream (sampling: 0, clients: 1)
        self.expected_clients = self.q * n_users

    def denominator(self, n_selected):
        if self.denominator_mode == "expected":
            return self.expected_clients
        if self.denominator_mode == "realised":
            return n_selected
        raise ValueError(f"unknown denominator {self.denominator_mode!r}")

    def run_round(self):
        self.round += 1
        selected = sample_clients(self.sampling_rng, self.n_users, self.q)
        info = {"round": self.round, "clients": len(selected), "skipped_empty": False}
        Q_t = self.Q
        clipped, norms, factors, touched, losses = [], [], [], [], []
        for u in selected:
            p_new, dQ, loss, t = local_update(self.P[u], Q_t, self.pos[u], self.pos_mask[u],
                                              self.lr, self.local_epochs, self.reg, self.client_rng)
            self.P[u] = p_new                       # local; never clipped, never uploaded
            c, n, f = clip_update(dQ, self.C)
            clipped.append(c)
            norms.append(n)
            factors.append(f)
            touched.append(t)
            losses.append(loss)
        if len(selected) == 0 and self.sigma == 0 and self.denominator_mode == "realised":
            info["skipped_empty"] = True            # B1 semantics: nothing to aggregate, nothing to add
            info.update(round_communication(0, self.n_items, self.dim, []))
            return info
        denom = self.denominator(len(selected))
        update, total, noise = private_update(clipped, self.Q.shape, self.sigma, self.C, self.noise_rng, denom)
        self.Q = Q_t + np.float32(self.server_lr) * update
        if not (np.isfinite(self.Q).all() and np.isfinite(self.P[selected]).all()):
            raise FloatingPointError(f"training diverged at round {self.round}")
        self.ever_sampled[selected] = True
        comm = round_communication(len(selected), self.n_items, self.dim, [int(t.sum()) for t in touched])
        self.comm_totals["download_bytes"] += comm["download_bytes"]
        self.comm_totals["upload_bytes"] += comm["upload_bytes"]
        info.update(comm)
        norms_a = np.array(norms) if norms else np.array([np.nan])
        agg_norm, noise_norm = update_norm(total), update_norm(noise)
        info.update({
            "local_loss": float(np.mean(losses)) if losses else np.nan,
            "update_norm": update_norm(update),
            "denominator": float(denom),
            "pre_clip_norm_mean": float(np.nanmean(norms_a)), "pre_clip_norm_median": float(np.nanmedian(norms_a)),
            "pre_clip_norm_p90": float(np.nanpercentile(norms_a, 90)),
            "pre_clip_norm_p95": float(np.nanpercentile(norms_a, 95)),
            "pre_clip_norm_max": float(np.nanmax(norms_a)),
            "frac_clipped": float(np.mean([f < 1.0 for f in factors])) if factors else np.nan,
            "post_clip_norm_mean": float(np.mean([n * f for n, f in zip(norms, factors)])) if norms else np.nan,
            "shrinkage_mean": float(np.mean(factors)) if factors else np.nan,
            "noise_std": float(self.sigma * self.C) if self.sigma > 0 else 0.0,
            "noise_norm": noise_norm, "clipped_aggregate_norm": agg_norm,
            "signal_to_noise": agg_norm / noise_norm if noise_norm > 0 else np.inf,
            "client_norms": norms,                  # per-client list (dropped before CSV export)
        })
        return info
