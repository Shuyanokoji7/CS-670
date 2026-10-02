"""B1 federated training loop with validation-based early stopping.

Validation uses the frozen evaluator (src/evaluate.py) on the current global Q and every
client's current (persisted) p_u. The test split is never touched here.
"""

import time

import numpy as np

from src.evaluate import evaluate
from src.federated import FederatedBPR

METRICS = ("ndcg", "hr", "recall", "mrr")


def validation_metrics(sim, split, k):
    summary, ranks = evaluate(sim.full_scores, split, "validation", k=k)
    return summary, ranks


def train_federated(train, n_users, n_items, cfg, seed, validate=None, log=print):
    """Run federated rounds. validate: callable(sim) -> metrics dict with 'ndcg@K'.

    Returns (sim with best-validation state restored, eval history, per-round log, best dict).
    """
    fl = cfg["federated"]
    k = cfg["evaluation"]["k"]
    key = f"ndcg@{k}"
    sim = FederatedBPR(train, n_users, n_items, cfg, seed)
    history, rounds_log = [], []
    t0 = time.perf_counter()

    def record(interval):
        row = {"round": sim.round}
        if interval:
            row.update({
                "clients_last_round": interval[-1]["clients"],
                "mean_clients_per_round": float(np.mean([r["clients"] for r in interval])),
                "local_loss": float(np.nanmean([r.get("local_loss", np.nan) for r in interval])),
                "update_norm_last_round": interval[-1].get("update_norm", 0.0),
                "client_pos_mean": float(np.nanmean([r.get("client_pos_mean", np.nan) for r in interval])),
                "client_pos_min": int(min(r.get("client_pos_min", 10**9) for r in interval)),
                "client_pos_max": int(max(r.get("client_pos_max", 0) for r in interval)),
            })
        row.update({
            "mean_user_norm": float(np.linalg.norm(sim.P, axis=1).mean()),
            "item_norm_mean": float(np.linalg.norm(sim.Q, axis=1).mean()),
            "frac_clients_ever_sampled": float(sim.ever_sampled.mean()),
            "cum_download_bytes": sim.comm_totals["download_bytes"],
            "cum_upload_bytes": sim.comm_totals["upload_bytes"],
            "wall_time_s": time.perf_counter() - t0,
        })
        if validate is not None:
            m = validate(sim)
            row.update({f"val_{x}@{k}": m[f"{x}@{k}"] for x in METRICS})
        return row

    best = {"round": 0, key: -np.inf, "state": sim.state()}
    first = record([])
    history.append(first)
    if validate is not None:
        best[key] = first[f"val_{key}"]
    bad_evals, interval = 0, []
    diverged = False
    for _ in range(fl["max_rounds"]):
        try:
            info = sim.run_round()
        except FloatingPointError as e:
            log(str(e))
            diverged = True
            break
        rounds_log.append(info)
        interval.append(info)
        if sim.round % fl["eval_every"]:
            continue
        row = record(interval)
        interval = []
        history.append(row)
        msg = f"round {sim.round:5d}  clients {row['mean_clients_per_round']:.1f}  loss {row['local_loss']:.4f}"
        if validate is not None:
            msg += f"  val ndcg@{k} {row[f'val_{key}']:.4f}"
            if row[f"val_{key}"] > best[key]:
                best = {"round": sim.round, key: row[f"val_{key}"], "state": sim.state()}
                bad_evals = 0
            else:
                bad_evals += 1
        log(msg)
        if validate is not None and bad_evals >= fl["patience_evals"]:
            log(f"early stop: no validation improvement for {fl['patience_evals']} evaluations "
                f"({fl['patience_evals'] * fl['eval_every']} rounds); best round {best['round']}")
            break

    if diverged:
        stopped_by = "diverged"
    elif validate is not None and bad_evals >= fl["patience_evals"]:
        stopped_by = "early_stopping"
    else:
        stopped_by = "max_rounds"
    best["stopped_by"] = stopped_by
    best["rounds_run"] = sim.round
    if validate is not None:
        sim.load_state(best["state"])
    return sim, history, rounds_log, best
