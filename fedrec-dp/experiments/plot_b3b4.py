"""Figures from results/b34_*.csv only (no training/evaluation). Error bars = training-seed SD (5 seeds)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "results" / "plots"
EPS = [("eps1", 1), ("eps2", 2), ("eps4", 4), ("eps8", 8)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(RES / "b34_test_means.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    series = [("B2 full rank (d=64)", "B2_{}", "k", "o")] + [(f"B4 rank {r}", f"B4_main_r{r}_{{}}", c, "s")
                                                             for r, c in ((4, "tab:blue"), (8, "tab:orange"),
                                                                          (16, "tab:green"), (32, "tab:red"))]
    for i, (lab, pat, col, mk) in enumerate(series):
        xs = [e * (1 + 0.03 * (i - 2)) for _, e in EPS]
        ys = [m.loc[pat.format(l), "test_ndcg_mean"] for l, _ in EPS]
        sd = [m.loc[pat.format(l), "test_ndcg_seed_sd"] for l, _ in EPS]
        ax.errorbar(xs, ys, yerr=sd, color=col, marker=mk, capsize=3, label=lab)
        ax.errorbar([16 * (1 + 0.03 * (i - 2))], [m.loc[pat.format("nodp"), "test_ndcg_mean"]],
                    yerr=[m.loc[pat.format("nodp"), "test_ndcg_seed_sd"]], color=col, marker=mk, capsize=3,
                    linestyle="none", markerfacecolor="white")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "no DP"])
    ax.set_xlabel("target ε (δ = 1e-5, T = 50, q = 0.1)")
    ax.set_ylabel("test NDCG@10")
    ax.set_title("B4 vs frozen B2: utility vs privacy\nerror bars = training-seed SD (5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"b4_utility_vs_epsilon.{ext}", dpi=150)
    plt.close(fig)

    c = pd.read_csv(RES / "b34_communication.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, lab, col in (("B1", "B1 full rank (d=64)", "k"), ("B3_r4", "B3 rank 4", "tab:blue"),
                           ("B3_r8", "B3 rank 8", "tab:orange"), ("B3_r16", "B3 rank 16", "tab:green"),
                           ("B3_r32", "B3 rank 32", "tab:red")):
        ax.errorbar(c.loc[name, "GB_to_best_mean"], m.loc[name, "test_ndcg_mean"], yerr=m.loc[name, "test_ndcg_seed_sd"],
                    color=col, marker="o", capsize=3, linestyle="none", label=lab)
    ax.set_xscale("log")
    ticks = [10, 20, 50, 100, 200]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("total GB to selected checkpoint (mean, 5 seeds; log scale)")
    ax.set_ylabel("test NDCG@10")
    ax.set_title("B3 (no DP): utility vs total communication\nerror bars = training-seed SD (5 seeds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"b3_utility_vs_total_comm.{ext}", dpi=150)
    plt.close(fig)
    print("wrote results/plots/b4_utility_vs_epsilon.{png,svg}, results/plots/b3_utility_vs_total_comm.{png,svg}")


if __name__ == "__main__":
    main()
