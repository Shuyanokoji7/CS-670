"""B2 training loop: exactly T private rounds, no early stopping, final model = model after round T.

Validation (frozen evaluator) is logged every `eval_every` rounds as a DIAGNOSTIC only; it never
selects a checkpoint, so the reported model is always the one whose privacy cost is T rounds.
The test split is never touched here.
"""

import time

import numpy as np

from src.privacy import DPFederatedBPR
from src.train_federated import METRICS, validation_metrics

ROUND_FIELDS = ["round", "clients", "denominator", "local_loss", "update_norm", "pre_clip_norm_mean",
                "pre_clip_norm_median", "pre_clip_norm_p90", "pre_clip_norm_p95", "pre_clip_norm_max",
                "frac_clipped", "post_clip_norm_mean", "shrinkage_mean", "noise_std", "noise_norm",
                "clipped_aggregate_norm", "signal_to_noise", "download_bytes", "upload_bytes",
                "bytes_per_client", "sparse_upload_bytes_diag", "skipped_empty"]


def train_dp_federated(split, cfg, seed, T, eval_every=10, validate=True, keep_client_norms=False, log=print):
    """Returns (sim after exactly T rounds, eval history, per-round diagnostics, all client norms)."""
    k = cfg["evaluation"]["k"]
    sim = DPFederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, seed)
    rounds, history, all_norms = [], [], []
    t0 = time.perf_counter()

    def evaluate_now():
        row = {"round": sim.round, "wall_time_s": time.perf_counter() - t0,
               "mean_user_norm": float(np.linalg.norm(sim.P, axis=1).mean()),
               "item_norm_mean": float(np.linalg.norm(sim.Q, axis=1).mean()),
               "cum_download_bytes": sim.comm_totals["download_bytes"],
               "cum_upload_bytes": sim.comm_totals["upload_bytes"]}
        if validate:
            m = validation_metrics(sim, split, k)[0]
            row.update({f"val_{x}@{k}": m[f"{x}@{k}"] for x in METRICS})
        return row

    history.append(evaluate_now())
    for _ in range(T):
        info = sim.run_round()
        if keep_client_norms:
            all_norms.extend((sim.round, n) for n in info.get("client_norms", []))
        info.pop("client_norms", None)
        rounds.append({f: info.get(f, np.nan) for f in ROUND_FIELDS})
        if sim.round % eval_every == 0 or sim.round == T:
            row = evaluate_now()
            history.append(row)
            if validate and (sim.round % (eval_every * 20) == 0 or sim.round == T):
                log(f"round {sim.round:5d}  val ndcg@{k} {row[f'val_ndcg@{k}']:.4f}  "
                    f"clipped {np.nanmean([r['frac_clipped'] for r in rounds[-eval_every:]]):.2f}")
    assert sim.round == T
    return sim, history, rounds, all_norms
