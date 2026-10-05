"""Generate report figures solely from the curated aggregate/seed tables."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cs670-matplotlib")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
COLORS = {"Full": "#444444", "Two": "#2166ac", "FixedB": "#b35806", "DP popularity": "#1b7837"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def main():
    old = pd.read_csv(ROOT / "results/baselines/final_5seed_means.csv")
    weighting = pd.read_csv(ROOT / "results/baselines/centralized_federated_3seeds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].bar(weighting.model, weighting["test_ndcg@10"], yerr=weighting["test_ndcg@10_std"],
                color=["#888888", "#9999bb", "#2166ac"], capsize=3)
    axes[0].set(title="Centralized / federated controls (3 seeds)", ylabel="Test NDCG@10", ylim=(0, .11))
    data = old[old.family.isin(["B1", "B3"])]
    axes[1].bar(data.model.str.replace("B3_", ""), data.test_ndcg_mean,
                yerr=data.test_ndcg_seed_sd, color=["#444444"] + ["#2166ac"] * 4, capsize=3)
    axes[1].set(title="Full / low-rank federated models (5 seeds)", ylabel="Test NDCG@10", ylim=(0, .11))
    save(fig, "01_nonprivate_controls")

    fig, ax = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    for family, rank, label, color in [("B2", None, "B2 Full", "#333333")] + [
        ("B4_main", r, f"B4 rank {r}", c) for r, c in zip([4, 8, 16, 32], ["#7b3294", "#2166ac", "#1b7837", "#b35806"])]:
        d = old[(old.family == family) & old.level.fillna("").str.startswith("eps")].copy()
        if rank is not None:
            d = d[d["rank"] == rank]
        d["eps"] = d.level.str[3:].astype(int)
        d = d.sort_values("eps")
        ax.errorbar(d.eps, d.test_ndcg_mean, yerr=d.test_ndcg_seed_sd, marker="o", capsize=3, label=label, color=color)
    ax.axhline(.044292, color="#777777", linestyle=":", label="Historical nonprivate popularity")
    ax.set(xlabel="Target epsilon (larger = weaker privacy)", ylabel="Test NDCG@10", xticks=[1, 2, 4, 8], ylim=(0, .065), title="Historical B2/B4, 50 rounds")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "02_historical_private_utility")

    e1 = pd.read_csv(ROOT / "results/effective_noise/test_means.csv").set_index("model")
    ranks = pd.read_csv(ROOT / "results/effective_noise/rank_selection.csv")
    selected = []
    fig, ax = plt.subplots(figsize=(6.8, 4.2), layout="constrained")
    for name, method in [("Full", "full"), ("Two", "two"), ("FixedB", "fixed"), ("DP popularity", "pop")]:
        ys, sds = [], []
        for eps in (1, 2, 4, 8):
            level = f"eps{eps}"
            if method in ("two", "fixed"):
                rank = int(ranks[(ranks.method == method) & (ranks.level == level)].iloc[0]["rank"])
                key = f"{method}_r{rank}_{level}"
            else:
                rank = None
                key = f"{method}_{level}"
            row = e1.loc[key]
            ys.append(row.ndcg_mean)
            sds.append(row.ndcg_seed_sd)
            selected.append({"method": name, "epsilon_target": eps, "rank": rank, "model": key,
                             "ndcg_mean": row.ndcg_mean, "ndcg_seed_sd": row.ndcg_seed_sd})
        ax.errorbar([1, 2, 4, 8], ys, yerr=sds, label=name, color=COLORS[name], marker="o", capsize=3)
    ax.set(xlabel="Target epsilon (larger = weaker privacy)", ylabel="Test NDCG@10", xticks=[1, 2, 4, 8], ylim=(0, .065), title="E1: ranks selected on validation, 50 rounds")
    ax.legend(fontsize=9, ncol=2)
    pd.DataFrame(selected).to_csv(ROOT / "results/effective_noise/selected_rank_utility.csv", index=False)
    save(fig, "03_effective_noise_utility")

    geo = pd.read_csv(ROOT / "results/effective_noise/geometry_means.csv")
    d = geo[geo.level == "eps1"].copy()
    order = ["full"] + [f"{m}_r{r}" for r in (4, 8, 16, 32) for m in ("two", "fixed")]
    d = d.set_index("unit").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), layout="constrained")
    names = ["Full"] + [f"{m} {r}" for r in (4, 8, 16, 32) for m in ("Two", "FixedB")]
    colors = [COLORS["Full"]] + [COLORS[m] for r in (4, 8, 16, 32) for m in ("Two", "FixedB")]
    for ax, col, title in zip(axes, ["q_noise_norm_rms", "score_noise_rmse_mean", "frac_clipped_mean"],
                              ["Item-matrix shock RMS", "Score shock RMSE (log scale)", "Fraction of clipped clients"]):
        ax.barh(names, d[col], color=colors)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
    axes[1].set_xscale("log")
    fig.suptitle("E1 at epsilon 1: all ranks, including unstable finite trajectories", fontsize=11)
    save(fig, "04_noise_and_clipping")

    path = ROOT / "results/noise_memory/by_seed.csv"
    if path.exists():
        memory = pd.read_csv(path)
        memory = memory[(memory.alpha == 1) & (memory.level == "eps1") & (memory.phase == "after_reset")]
        aligned = pd.read_csv(ROOT / "results/noise_memory_aligned/by_seed.csv")
        aligned = aligned[(aligned.alpha == 1) & (aligned.level == "eps1") & (aligned.phase == "after_reset")]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        panels = [("full", memory, "Full"), ("fixed_r8", memory, "FixedB rank 8"),
                  ("two_r8", memory, "Two rank 8: raw-coordinate coupling"),
                  ("two_r8", aligned, "Two rank 8: aligned coupling")]
        for ax, (unit, data, title) in zip(axes.ravel(), panels):
            for name, label, color in [("retained", "Pulse retained", "#444444"),
                                       ("shared_reset", "Shared state reset", "#2166ac"),
                                       ("local_reset", "Local state reset", "#b35806")]:
                z = data[(data.unit == unit) & (data.branch == name)].groupby("lag").top10_churn_all
                mean, sd = z.mean(), z.std()
                ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", capsize=3, color=color, label=label)
            ax.set(title=title, xlabel="Rounds after reset", ylabel="Top-10 set disagreement", ylim=(0, .65))
        axes[0, 1].legend(fontsize=8)
        fig.suptitle("E2 exploratory probe: disturbance, not recommendation relevance", fontsize=11)
        save(fig, "05_local_state_memory")
        raw = pd.read_csv(ROOT / "results/noise_memory/by_seed.csv")
        ali = pd.read_csv(ROOT / "results/noise_memory_aligned/by_seed.csv")
        fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
        for i, (name, data, color) in enumerate([("Raw-coordinate coupling", raw, "#b35806"),
                                                ("Aligned coupling", ali, "#2166ac")]):
            d = data[(data.unit == "two_r8") & (data.alpha == 1) & (data.phase == "after_reset")
                     & (data.lag == 10) & (data.branch == "shared_reset")]
            g = d.groupby("level").top10_churn_all
            centers = np.array([0, 1]) + (i - .5) * .34
            ax.errorbar(centers, g.mean().values * 100, yerr=g.std().values * 100,
                        linestyle="none", marker="D", color=color, capsize=4, label=name)
            for center, level in zip(centers, ("eps1", "eps2")):
                values = d[d.level == level].sort_values("seed").top10_churn_all.values * 100
                ax.scatter(center + np.linspace(-.04, .04, len(values)), values,
                           facecolors="none", edgecolors=color, s=25, alpha=.8)
            for x, y in zip(np.array([0, 1]) + (i - .5) * .34, g.mean().values * 100):
                ax.annotate(f"{y:.2f}%", (x, y), xytext=(8, 0), textcoords="offset points", fontsize=9)
        ax.set(xticks=[0, 1], xticklabels=["epsilon 1", "epsilon 2"], yscale="log", ylim=(.05, 80),
               ylabel="Top-10 disagreement (%) — log scale", title="Two-r8 shared-reset probe after 10 rounds")
        ax.set_xlim(-.4, 1.65)
        ax.legend(fontsize=9)
        save(fig, "06_coupling_control")
    print("Figures generated from curated tables")


if __name__ == "__main__":
    main()
