"""Training loops for the low-rank simulator.

train_lowrank_converge: B3 — B1's convergence protocol (validation every eval_every rounds, patience in evaluations,
    max_rounds, best-validation state restored; divergence recorded).
train_lowrank_fixed:    B4 — exactly T rounds, no early stopping, final model = round-T model; validation is a
    diagnostic only.
Validation uses the frozen evaluator through src.train_federated.validation_metrics. Test is never touched here.
"""

import time

import numpy as np

from src.lowrank import LowRankFederatedBPR
from src.train_dp_federated import ROUND_FIELDS
from src.train_federated import METRICS, validation_metrics

LOWRANK_ROUND_FIELDS = ROUND_FIELDS + ["A_fro", "B_fro", "mean_touched_rows"]


def _eval_row(sim, t0, split, k, validate):
    row = {"round": sim.round, "wall_time_s": time.perf_counter() - t0,
           "mean_user_norm": float(np.linalg.norm(sim.P, axis=1).mean()),
           "item_norm_mean": float(np.linalg.norm(sim.Q, axis=1).mean()),
           "A_fro": float(np.linalg.norm(sim.A)), "B_fro": float(np.linalg.norm(sim.B)),
           "frac_clients_ever_sampled": float(sim.ever_sampled.mean()),
           "cum_download_bytes": sim.comm_totals["download_bytes"],
           "cum_upload_bytes": sim.comm_totals["upload_bytes"]}
    if validate:
        try:
            m = validation_metrics(sim, split, k)[0]
        except ValueError as e:
            # Only a CONFIRMED numerical failure is relabelled as divergence: the evaluator's exact non-finite-scores
            # message AND a direct check that the score matrix really is non-finite (e.g. float32 overflow of
            # P (A B)^T with finite factors). Any other ValueError (shape/programming errors) is re-raised unchanged.
            if str(e) == "scores must be finite" and not scores_all_finite(sim):
                raise FloatingPointError(f"non-finite scores at round {sim.round}") from e
            raise
        row.update({f"val_{x}@{k}": m[f"{x}@{k}"] for x in METRICS})
    return row


def scores_all_finite(sim, batch=256):
    for start in range(0, sim.n_users, batch):
        if not np.isfinite(sim.full_scores(np.arange(start, min(start + batch, sim.n_users)))).all():
            return False
    return True


def train_lowrank_converge(split, cfg, seed, log=print):
    """B3. Returns (sim with best-validation state, eval history, per-round log, best dict incl. stopped_by)."""
    fl, k = cfg["federated"], cfg["evaluation"]["k"]
    key = f"ndcg@{k}"
    sim = LowRankFederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, seed)
    t0 = time.perf_counter()
    history, rounds = [_eval_row(sim, t0, split, k, True)], []
    best = {"round": 0, key: history[0][f"val_{key}"], "state": sim.state()}
    bad, interval, diverged = 0, [], False
    for _ in range(fl["max_rounds"]):
        try:
            info = sim.run_round()
        except FloatingPointError as e:               # non-finite parameters
            log(str(e))
            diverged = True
            break
        info.pop("client_norms", None)
        rounds.append({f: info.get(f, np.nan) for f in LOWRANK_ROUND_FIELDS})
        interval.append(info)
        if sim.round % fl["eval_every"]:
            continue
        try:
            row = _eval_row(sim, t0, split, k, True)
        except FloatingPointError as e:               # non-finite scores at evaluation
            log(str(e))
            diverged = True
            break
        row["local_loss"] = float(np.nanmean([r.get("local_loss", np.nan) for r in interval]))
        row["mean_clients_per_round"] = float(np.mean([r["clients"] for r in interval]))
        interval = []
        history.append(row)
        if row[f"val_{key}"] > best[key]:
            best = {"round": sim.round, key: row[f"val_{key}"], "state": sim.state()}
            bad = 0
        else:
            bad += 1
        if sim.round % (fl["eval_every"] * 50) == 0:
            log(f"round {sim.round:5d}  val {key} {row[f'val_{key}']:.4f}  best {best[key]:.4f}@{best['round']}")
        if bad >= fl["patience_evals"]:
            break
    best["stopped_by"] = ("diverged" if diverged else
                          "early_stopping" if bad >= fl["patience_evals"] else "max_rounds")
    best["rounds_run"] = sim.round
    sim.load_state(best["state"])
    return sim, history, rounds, best


def train_lowrank_fixed(split, cfg, seed, T, eval_every=10, validate=True, keep_client_norms=False, log=print):
    """B4. Exactly T rounds. Returns (sim after round T, eval history, per-round diagnostics, client norms)."""
    k = cfg["evaluation"]["k"]
    sim = LowRankFederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, seed)
    t0 = time.perf_counter()
    history, rounds, norms = [_eval_row(sim, t0, split, k, validate)], [], []
    for _ in range(T):
        info = sim.run_round()
        if keep_client_norms:
            norms.extend((sim.round, n) for n in info.get("client_norms", []))
        info.pop("client_norms", None)
        rounds.append({f: info.get(f, np.nan) for f in LOWRANK_ROUND_FIELDS})
        if sim.round % eval_every == 0 or sim.round == T:
            history.append(_eval_row(sim, t0, split, k, validate))
    assert sim.round == T
    return sim, history, rounds, norms
