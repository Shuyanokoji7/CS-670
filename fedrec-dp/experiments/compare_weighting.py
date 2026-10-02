"""Weighting control: separate the weighting change from the federation change in B0 -> B1.

    B0     = centralised, interaction-weighted objective
    B0-UW  = centralised, user-weighted objective   (diagnostic control, configs/b0_uw.yaml)
    B1     = federated,   user-weighted objective

    B0 -> B0-UW  ~ weighting effect
    B0-UW -> B1  ~ federation effect (same objective weighting)

All comparisons are paired per user on test NDCG@10, each model's per-user score averaged over
its training seeds, with a 10,000-resample user bootstrap.

    python experiments/compare_weighting.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_processed  # noqa: E402

RESULTS, RAW = ROOT / "results", ROOT / "results" / "raw"
SEEDS = [42, 123, 2026]
MODELS = {"B0": "b0", "B0-UW": "b0uw", "B1": "b1"}
K = 10


def per_user(prefix, metric=f"ndcg@{K}"):
    return np.mean([pd.read_csv(RAW / f"{prefix}_per_user_test_seed{s}.csv").sort_values("user")[metric].to_numpy()
                    for s in SEEDS], axis=0)


def paired(d, seed=1, n_boot=10000):
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_improved": float((d > 0).mean()), "frac_degraded": float((d < 0).mean()),
            "frac_tied": float((d == 0).mean())}


def main():
    split = load_processed()
    scores = {name: per_user(prefix) for name, prefix in MODELS.items()}

    summary = []
    for name, prefix in MODELS.items():
        s = pd.read_csv(RESULTS / f"{prefix}_summary.csv").set_index("split")
        summary.append({"model": name, **{f"test_{m}@{K}": s.loc["test", f"{m}@{K}_mean"] for m in ("ndcg", "hr", "mrr")},
                        f"test_ndcg@{K}_std": s.loc["test", f"ndcg@{K}_std"],
                        f"val_ndcg@{K}": s.loc["validation", f"ndcg@{K}_mean"]})
    summary = pd.DataFrame(summary)

    comparisons = [("B0-UW - B0", "weighting effect (centralised)", "B0-UW", "B0"),
                   ("B1 - B0-UW", "federation effect (same user weighting)", "B1", "B0-UW"),
                   ("B1 - B0", "total B0 -> B1 (previously reported)", "B1", "B0")]
    rows = [{"comparison": c, "interpretation": what, **paired(scores[a] - scores[b])} for c, what, a, b in comparisons]
    paired_df = pd.DataFrame(rows)

    n_train = split["train"].groupby("user").size().reindex(range(split["n_users"])).to_numpy()
    cuts = np.quantile(n_train, [1 / 3, 2 / 3])
    group = np.where(n_train <= cuts[0], "low", np.where(n_train <= cuts[1], "medium", "high"))
    act = []
    for g in ("low", "medium", "high"):
        m = group == g
        row = {"group": g, "n_users": int(m.sum()), "train_pos_min": int(n_train[m].min()),
               "train_pos_max": int(n_train[m].max())}
        row.update({f"{name}_ndcg@{K}": float(v[m].mean()) for name, v in scores.items()})
        for c, _, a, b in comparisons[:2]:
            p = paired(scores[a][m] - scores[b][m])
            row.update({f"{c}: mean": p["mean_diff"], f"{c}: ci_low": p["ci_low"], f"{c}: ci_high": p["ci_high"]})
        act.append(row)
    act = pd.DataFrame(act)

    summary.to_csv(RESULTS / "weighting_control_summary.csv", index=False, float_format="%.6f")
    paired_df.to_csv(RESULTS / "weighting_control_paired.csv", index=False, float_format="%.6f")
    act.to_csv(RESULTS / "weighting_control_activity.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    for df in (summary, paired_df, act):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), "\n")


if __name__ == "__main__":
    main()
