"""E1 analysis from retained artifacts. Validation ablations never select settings.

--validation: mechanism summaries, validation personalization and activity groups.
--test: predeclared paired bootstrap, seed differences and descriptive subgroups.
--plots: standalone PNG/SVG figures from retained CSVs; no retraining/rescoring.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.threads import force_env, enforce_and_report
force_env()

import argparse
import json

import numpy as np
import pandas as pd

import experiments.run_effective_noise as runner
from src.effective_noise import EffectiveNoiseBPR
from src.evaluate import evaluate, target_items
from src.metrics import ndcg_from_rank

OUT = runner.OUT
LEVELS = runner.LEVELS


def model_key(unit, level):
    return f"{unit}_{level}"


def validation_analysis():
    cfg, split, fp = runner.load_all()
    rows = json.loads((OUT/"final_records.json").read_text())
    runner.check_inventory(cfg, rows)
    geometry, personalization, subgroups = [], [], []
    pos_counts = np.bincount(split["train"].user, minlength=split["n_users"])
    train_item_counts = np.bincount(split["train"].item, minlength=split["n_items"])
    target = target_items(split, "validation")
    groups = {"low_activity": pos_counts <= 22, "medium_activity": (pos_counts > 22)&(pos_counts <= 61),
              "high_activity": pos_counts > 61, "cold_target": train_item_counts[target] == 0,
              "warm_target": train_item_counts[target] > 0}
    for row in rows:
        c = row["settings"]
        rounds = pd.read_csv(OUT/"runs"/f"{row['tag']}_rounds.csv")
        score = rounds.dropna(subset=["score_noise_energy"])
        expected, measured = rounds.q_noise_expected_energy.sum(), rounds.q_noise_energy.sum()
        sexpected, smeasured = score.score_noise_expected_energy.sum(), score.score_noise_energy.sum()
        record = {"unit": row["unit"], "method": row["method"], "rank": row["rank"],
                  "level": row["level"], "seed": row["seed"], "local_lr": row["local_lr"],
                  "server_lr": row["server_lr"], "D": row["D"], "total_bytes": row["total_bytes"],
                  "q_noise_energy_mean": rounds.q_noise_energy.mean(),
                  "q_noise_norm_rms": np.sqrt(rounds.q_noise_energy.mean()),
                  "q_noise_expected_energy_mean": rounds.q_noise_expected_energy.mean(),
                  "q_measured_expected_ratio": measured/expected if expected else np.nan,
                  "expected_cross_fraction": rounds.q_noise_expected_cross_energy.sum()/expected if expected else 0.,
                  "score_measured_expected_ratio": smeasured/sexpected if sexpected else np.nan,
                  "score_noise_rmse_mean": score.score_noise_rmse.mean(),
                  "score_noise_relative_mean": score.score_noise_relative.mean(),
                  "margin_noise_rmse_mean": score.margin_noise_rmse.mean(),
                  "margin_sign_flip_fraction_mean": score.margin_sign_flip_fraction.mean(),
                  "frac_clipped_mean": rounds.frac_clipped.mean(),
                  "pre_clip_norm_median_mean": rounds.pre_clip_norm_median.mean(),
                  "effective_signal_norm_mean": rounds.effective_signal_norm.mean(),
                  "coordinate_noise_norm_mean": rounds.coordinate_noise_norm.mean(),
                  "float32_residual_norm_max": rounds.float32_residual_norm.max(),
                  "cold_row_noise_rms_mean": rounds.q_noise_cold_row_rms.mean(),
                  "final_mean_user_norm": rounds.mean_user_norm.iloc[-1],
                  "val_ndcg": row["validation"]["ndcg@10"]}
        for name in ("low", "medium", "high"):
            record[f"score_noise_rmse_{name}"] = score[f"score_noise_rmse_{name}"].mean()
        geometry.append(record)
        sim = EffectiveNoiseBPR(split["train"], split["n_users"], split["n_items"], c)
        sim.load_state(dict(np.load(ROOT/row["checkpoint"], allow_pickle=False)))
        Q, P = sim.Q, sim.P
        perm = np.random.default_rng(2026).permutation(len(P))
        for mode, p in (("own", P), ("permuted", P[perm]), ("mean", np.broadcast_to(P.mean(0), P.shape))):
            summary, ranks = evaluate(lambda users: (p[users] @ Q.T).astype(np.float64), split, "validation", k=10)
            personalization.append({"unit": row["unit"], "level": row["level"], "seed": row["seed"],
                                    "mode": mode, "ndcg": summary["ndcg@10"]})
            if mode == "own":
                values = ndcg_from_rank(ranks, 10)
                for group, mask in groups.items():
                    subgroups.append({"unit": row["unit"], "level": row["level"], "seed": row["seed"],
                                      "group": group, "n_users": int(mask.sum()), "ndcg": values[mask].mean()})
    pd.DataFrame(geometry).to_csv(OUT/"geometry_per_run.csv", index=False, float_format="%.12g")
    g = pd.DataFrame(geometry).groupby(["unit", "method", "rank", "level"], sort=False)
    g.mean(numeric_only=True).drop(columns="seed").to_csv(OUT/"geometry_means.csv", float_format="%.12g")
    pd.DataFrame(personalization).to_csv(OUT/"personalization_validation.csv", index=False, float_format="%.12g")
    pd.DataFrame(subgroups).to_csv(OUT/"subgroups_validation.csv", index=False, float_format="%.12g")
    print("validation geometry, personalization and descriptive groups saved")


def test_analysis():
    cfg, split, _ = runner.load_all()
    runner.require_frozen(cfg)
    records = json.loads((OUT/"test_records.json").read_text())
    tables = {}
    for r in records:
        path = ROOT/r["users_path"]
        if runner.sha(path) != r["users_sha256"]:
            raise RuntimeError("test per-user artifact changed")
        table = pd.read_csv(path).sort_values("user")
        if not np.array_equal(table.user, np.arange(split["n_users"])) or not np.isfinite(table[["ndcg@10", "hr@10", "mrr@10"]]).all().all():
            raise RuntimeError("invalid per-user test table")
        tables.setdefault(model_key(r["unit"], r["level"]), {})[r["seed"]] = table
    means, models = [], {}
    for key, seeds in tables.items():
        if len(seeds) != 5 and key != "pop_bounded_nodp":
            raise RuntimeError(f"incomplete five-seed model: {key}")
        tabs = [seeds[s] for s in sorted(seeds)]
        models[key] = np.stack([t["ndcg@10"].to_numpy() for t in tabs]).mean(0)
        row = {"model": key, "n_seeds": len(tabs)}
        for metric, name in (("ndcg@10", "ndcg"), ("hr@10", "hr"), ("mrr@10", "mrr")):
            sm = [t[metric].mean() for t in tabs]
            row[f"{name}_mean"] = float(np.mean(sm))
            row[f"{name}_seed_sd"] = float(np.std(sm, ddof=1)) if len(sm) > 1 else 0.
        means.append(row)
    # Historical frozen B4 is an explicitly secondary contextual comparison.
    # Read saved per-user outputs; never retrain or rescore its checkpoints.
    historical = pd.read_csv(ROOT/"results/b4_final_results.csv")
    for (rank, level), cells in historical[historical.role == "main"].groupby(["rank", "privacy_level"]):
        tabs = []
        if set(cells.seed) != set(cfg["seeds"]):
            raise RuntimeError("historical B4 seed inventory mismatch")
        for cell in cells.itertuples():
            table = pd.read_csv(ROOT/"results/raw/b4_runs"/f"per_user_test_{cell.tag}.csv").sort_values("user")
            if not np.array_equal(table.user, np.arange(split["n_users"])) or not np.isfinite(table["ndcg@10"]).all():
                raise RuntimeError("historical B4 per-user input mismatch")
            tabs.append(table["ndcg@10"].to_numpy())
        models[f"historical_B4_r{rank}_{level}"] = np.mean(tabs, axis=0)
    contrasts = []
    for r in cfg["model"]["ranks"]:
        for level in LEVELS[2:]:
            a, b = f"fixed_r{r}_{level}", f"two_r{r}_{level}"
            contrasts.append(("primary_fixed_vs_two", a, b))
            contrasts.append(("secondary_fixed_vs_full", a, f"full_{level}"))
            contrasts.append(("secondary_fixed_vs_pop", a, f"pop_{level}"))
            contrasts.append(("secondary_fixed_vs_historical_B4", a, f"historical_B4_r{r}_{level}"))
    for method, rank in runner.units(cfg):
        name = runner.unit(method, rank)
        contrasts.append(("secondary_clipping", f"{name}_clip", f"{name}_nodp"))
        for level in LEVELS[2:]:
            contrasts.append(("secondary_noise", f"{name}_{level}", f"{name}_clip"))
    for level in LEVELS[2:]:
        contrasts.append(("secondary_pop_noise", f"pop_{level}", "pop_bounded_nodp"))
    selection = pd.read_csv(OUT/"rank_selection.csv")
    for level in LEVELS[2:]:
        f = int(selection[(selection.method == "fixed")&(selection.level == level)].iloc[0]["rank"])
        t = int(selection[(selection.method == "two")&(selection.level == level)].iloc[0]["rank"])
        contrasts.append(("secondary_selected_ranks", f"fixed_r{f}_{level}", f"two_r{t}_{level}"))
    D = np.stack([models[a]-models[b] for _, a, b in contrasts])
    n, B = D.shape[1], cfg["statistics"]["bootstrap_resamples"]
    rng = np.random.default_rng(cfg["statistics"]["bootstrap_seed"])
    boots = np.empty((len(contrasts), B))
    for start in range(0, B, 2000):
        size = min(2000, B-start)
        idx = rng.integers(0, n, size=(size, n))
        offsets = np.arange(size)[:, None]*n
        counts = np.bincount((idx+offsets).ravel(), minlength=size*n).reshape(size, n).astype(np.float64)
        boots[:, start:start+size] = (counts @ D.T).T/n
    results, seed_diffs = [], []
    for (family, a, b), d, bt in zip(contrasts, D, boots):
        row = {"family": family, "a": a, "b": b, "diff": float(d.mean()),
               "ci95_lo": float(np.quantile(bt, .025)), "ci95_hi": float(np.quantile(bt, .975))}
        if family == "primary_fixed_vs_two":
            tail = .05/(2*cfg["statistics"]["primary_contrasts"])
            row.update(sim_level=1-2*tail, sim_lo=float(np.quantile(bt, tail)), sim_hi=float(np.quantile(bt, 1-tail)))
        results.append(row)
        if family == "primary_fixed_vs_two":
            for seed in cfg["seeds"]:
                seed_diffs.append({"a": a, "b": b, "seed": seed,
                                   "diff": float(tables[a][seed]["ndcg@10"].mean()-tables[b][seed]["ndcg@10"].mean())})
    pd.DataFrame(means).to_csv(OUT/"test_means.csv", index=False, float_format="%.12g")
    pd.DataFrame(results).to_csv(OUT/"contrasts.csv", index=False, float_format="%.12g")
    pd.DataFrame(seed_diffs).to_csv(OUT/"primary_seed_differences.csv", index=False, float_format="%.12g")
    counts = np.bincount(split["train"].user, minlength=n)
    ic = np.bincount(split["train"].item, minlength=split["n_items"])
    targets = target_items(split, "test")
    masks = {"low_activity": counts <= 22, "medium_activity": (counts > 22)&(counts <= 61),
             "high_activity": counts > 61, "cold_target": ic[targets] == 0, "warm_target": ic[targets] > 0}
    group_rows = [{"model": key, "group": group, "n_users": int(mask.sum()), "ndcg_mean": float(v[mask].mean())}
                  for key, v in models.items() for group, mask in masks.items()]
    pd.DataFrame(group_rows).to_csv(OUT/"subgroups_test.csv", index=False, float_format="%.12g")
    primary = pd.DataFrame(results).query("family == 'primary_fixed_vs_two'")
    print(primary[["a", "diff", "sim_lo", "sim_hi"]].to_string(index=False))


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest = OUT/"plots"
    dest.mkdir(exist_ok=True)
    data = pd.read_csv(OUT/"test_means.csv").set_index("model")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, r in zip(axes, [4, 8, 16, 32]):
        for label, name, color in [("Full", "full", "black"), ("Two factors", f"two_r{r}", "#d96536"),
                                    ("Fixed B", f"fixed_r{r}", "#2879b9"), ("DP popularity", "pop", "#739b40")]:
            records = [data.loc[f"{name}_eps{e}"] for e in [1, 2, 4, 8]]
            ax.errorbar([1, 2, 4, 8], [x.ndcg_mean for x in records], yerr=[x.ndcg_seed_sd for x in records],
                        label=label, color=color, marker="o", capsize=2, linewidth=1.4)
        ax.axhline(.044292, color="gray", linestyle=":", label="Raw popularity reference")
        ax.set_title(f"Rank {r}")
        ax.set_xlabel("Target epsilon")
        ax.set_xticks([1, 2, 4, 8])
    axes[0].set_ylabel("Test NDCG@10 (mean ± seed SD)")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(dest/f"utility.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    g = pd.read_csv(OUT/"geometry_means.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    for ax, col, label in zip(axes, ["coordinate_noise_norm_mean", "q_noise_norm_rms", "score_noise_rmse_mean"],
                              ["Coordinate noise norm", "Effective Q shock RMS norm", "Score shock RMSE"]):
        for method, color in [("two", "#d96536"), ("fixed", "#2879b9")]:
            x = g[(g.method == method)&(g.level == "eps1")].sort_values("rank")
            ax.plot(x["rank"], x[col], "o-", label=method, color=color)
        full = g[(g.method == "full")&(g.level == "eps1")].iloc[0]
        ax.axhline(full[col], color="black", linestyle="--", label="full")
        ax.set_xlabel("Rank")
        ax.set_xticks([4, 8, 16, 32])
        ax.set_ylabel(label)
    axes[0].legend()
    fig.suptitle("Instantaneous shocks at epsilon ≈1, selected settings")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(dest/f"noise_geometry.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    enforce_and_report()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["validation", "test", "plots"])
    args = parser.parse_args()
    {"validation": validation_analysis, "test": test_analysis, "plots": plots}[args.stage]()


if __name__ == "__main__":
    main()
