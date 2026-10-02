"""B2 — user-level DP federated BPR. Experiment runner.

    python experiments/run_b2.py --accounting                 # solve sigma per target eps (+ cross-check)
    python experiments/run_b2.py --norm-stats                 # B1 client-update norms (no test, no DP)
    python experiments/run_b2.py --sanity --clip C            # one eps≈4 run, mechanism checks (validation only)
    python experiments/run_b2.py --clip-search                # pre-declared C grid at eps≈4 (validation only)
    python experiments/run_b2.py --server-lr-check            # only if the declared failure criterion is met
    python experiments/run_b2.py --run --epsilon 4 --seed 42  # one run (use --epsilon none for the no-DP control)
    python experiments/run_b2.py --sweep --seeds 42 123 2026  # control + all eps, then analysis and plots
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from experiments.run_b1 import per_user_table  # noqa: E402
from src.data import load_processed, split_fingerprint  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.privacy import DPFederatedBPR, privacy_record, solve_sigma  # noqa: E402
from src.train_dp_federated import train_dp_federated  # noqa: E402
from src.train_federated import METRICS, validation_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS, RAW, CHECKPOINTS = ROOT / "results", ROOT / "results" / "raw", ROOT / "checkpoints"
CONFIG = "configs/b2.yaml"
B1_SERVER_LR = 1.0


def load_all():
    cfg = load_config(ROOT / CONFIG)
    data_cfg = load_config(ROOT / cfg["dataset_config"])
    split = load_processed(data_cfg)
    return cfg, data_cfg, split, split_fingerprint(split)


def sigma_table(cfg):
    """Noise multipliers for the declared target epsilons (cached in results/b2_accounting.csv)."""
    path = RESULTS / "b2_accounting.csv"
    if not path.exists():
        raise FileNotFoundError("run --accounting first")
    return pd.read_csv(path).set_index("target_epsilon")


def run_cfg(cfg, epsilon, clip=None, server_lr=None):
    """Privacy config for one run. epsilon=None -> matched no-DP control (no clipping, no noise)."""
    c = copy.deepcopy(cfg)
    pc = c["privacy"]
    if epsilon is None:
        pc.update(clip_norm=None, noise_multiplier=0.0, target_epsilon=None)
    else:
        tab = sigma_table(cfg)
        pc.update(clip_norm=float(clip if clip is not None else pc["clip_norm"]),
                  noise_multiplier=float(tab.loc[float(epsilon), "noise_multiplier"]),
                  target_epsilon=float(epsilon))
    if server_lr is not None:
        c["federated"]["server_lr"] = float(server_lr)
    return c


def label(epsilon):
    return "nodp" if epsilon is None else f"eps{epsilon:g}"


# --------------------------------------------------------------------------- accounting

def accounting():
    cfg = load_config(ROOT / CONFIG)
    pc, q, T = cfg["privacy"], cfg["federated"]["client_sampling_q"], cfg["privacy"]["T"]
    rows = []
    for eps in pc["target_epsilons"]:
        sigma, achieved = solve_sigma(eps, q, T, pc["delta"], pc["accountant"], tol=pc["solver_tolerance"])
        rec = privacy_record({"noise_multiplier": sigma, "delta": pc["delta"], "clip_norm": None,
                              "denominator": pc["denominator"]}, q, T, pc["accountant"], pc["cross_check_accountant"])
        rows.append({"target_epsilon": eps, "noise_multiplier": sigma, "epsilon": rec["epsilon"],
                     "accountant": pc["accountant"], "epsilon_cross_check": rec["epsilon_cross_check"],
                     "cross_check_accountant": pc["cross_check_accountant"], "delta": pc["delta"], "q": q, "T": T})
        print(f"target eps {eps}: sigma {sigma:.4f} -> {pc['accountant']} eps {rec['epsilon']:.4f}, "
              f"{pc['cross_check_accountant']} eps {rec['epsilon_cross_check']:.4f}")
    pd.DataFrame(rows).to_csv(RESULTS / "b2_accounting.csv", index=False, float_format="%.12g")   # full sigma precision


# --------------------------------------------------------------------------- B1 norm statistics

def norm_stats(seed=42):
    """Exact B1 mechanics (no clip, no noise, realised denominator) for T rounds; record ||ΔQ_u||_F."""
    cfg, _, split, _ = load_all()
    c = copy.deepcopy(cfg)
    c["privacy"].update(clip_norm=None, noise_multiplier=0.0, denominator="realised")
    T = cfg["privacy"]["T"]
    _, _, _, norms = train_dp_federated(split, c, seed, T, eval_every=T, validate=False, keep_client_norms=True,
                                        log=lambda *_: None)
    df = pd.DataFrame(norms, columns=["round", "update_norm"])
    df.to_csv(RAW / f"b2_b1_client_update_norms_seed{seed}.csv", index=False, float_format="%.6f")
    qs = [0.25, 0.5, 0.75, 0.9, 0.95]
    rows = []
    for name, sub in [("all rounds 1-%d" % T, df), ("rounds 1-100", df[df["round"] <= 100]),
                      ("rounds 101-500", df[(df["round"] > 100) & (df["round"] <= 500)]),
                      ("rounds 501-%d" % T, df[df["round"] > 500])]:
        x = sub["update_norm"]
        rows.append({"rounds": name, "n_client_updates": len(x), "mean": x.mean(), "min": x.min(),
                     **{f"p{int(p * 100)}": x.quantile(p) for p in qs}, "max": x.max()})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "b2_b1_update_norm_stats.csv", index=False, float_format="%.6f")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- training a run

def train_one(c, split, seed, log=lambda *_: None):
    T = c["privacy"]["T"]
    return train_dp_federated(split, c, seed, T, eval_every=c["federated"]["eval_every"], log=log)


def save_checkpoint(path, sim, cfg, data_cfg, seed, fp, rec, val_summary, split):
    torch.save({
        "P": torch.from_numpy(sim.P.copy()), "Q": torch.from_numpy(sim.Q.copy()),
        "model": "DPFederatedBPRMF", "config": cfg, "dataset_config": data_cfg,
        "seed": seed, "noise_seed": sim.noise_seed, "round": sim.round, "split_fingerprint": fp,
        "q": rec["q"], "T": rec["T"], "clip_norm": rec["clip_norm"], "noise_multiplier": rec["noise_multiplier"],
        "delta": rec["delta"], "epsilon": rec["epsilon"], "accountant": rec["accountant"],
        "epsilon_cross_check": rec["epsilon_cross_check"], "denominator": rec["denominator"],
        "target_epsilon": rec["target_epsilon"], "validation": val_summary,
        "n_users": split["n_users"], "n_items": split["n_items"], "torch_version": str(torch.__version__),
    }, path)


PRIVACY_KEYS = ("q", "T", "clip_norm", "noise_multiplier", "delta", "denominator")


def load_checkpoint(path, split, expected_privacy=None):
    """Safe weights-only load; rejects a different split or (optionally) a different privacy config."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt["split_fingerprint"] != split_fingerprint(split):
        raise ValueError(f"{path} was trained on a different data split")
    if expected_privacy is not None:
        bad = [k for k in PRIVACY_KEYS if ckpt[k] != expected_privacy[k]]
        if bad:
            raise ValueError(f"{path}: privacy configuration mismatch in {bad}")
    sim = DPFederatedBPR(split["train"], split["n_users"], split["n_items"], ckpt["config"], ckpt["seed"])
    sim.load_state({"P": ckpt["P"].numpy(), "Q": ckpt["Q"].numpy()})
    sim.round = ckpt["round"]
    return sim, ckpt


def _diag_summary(rounds):
    r = pd.DataFrame(rounds)
    return {"mean_clients_per_round": r["clients"].mean(), "frac_clipped_mean": r["frac_clipped"].mean(),
            "pre_clip_norm_median_mean": r["pre_clip_norm_median"].mean(),
            "shrinkage_mean": r["shrinkage_mean"].mean(), "noise_norm_mean": r["noise_norm"].mean(),
            "clipped_aggregate_norm_mean": r["clipped_aggregate_norm"].mean(),
            "signal_to_noise_mean": r["signal_to_noise"].replace(np.inf, np.nan).mean(),
            "comm_bytes_total": int((r["download_bytes"].fillna(0) + r["upload_bytes"].fillna(0)).sum()),
            "bytes_per_client_per_round": int(r["bytes_per_client"].max())}


def run_one(args):
    """Train one (epsilon, seed); evaluate the round-T model on validation and ONCE on test."""
    epsilon, seed, clip, server_lr, evaluate_test = args
    cfg, data_cfg, split, fp = load_all()
    c = run_cfg(cfg, epsilon, clip, server_lr)
    k = c["evaluation"]["k"]
    rec = privacy_record(c["privacy"], c["federated"]["client_sampling_q"], c["privacy"]["T"],
                         c["privacy"]["accountant"], c["privacy"]["cross_check_accountant"])
    sim, history, rounds, _ = train_one(c, split, seed)
    val_summary, val_ranks = validation_metrics(sim, split, k)
    D = split["n_items"] * c["model"]["dim"]
    sc = rec["noise_multiplier"] * (rec["clip_norm"] or 0.0)
    meta = {"model": "b2_dp_federated_bpr" if rec["dp"] else "b1_dpready_nodp", "privacy_level": label(epsilon),
            "seed": seed, "noise_seed": sim.noise_seed, "rounds": sim.round, **rec,
            "server_lr": c["federated"]["server_lr"], "shared_coordinates_D": D,
            "expected_noise_sq_norm": D * sc ** 2, "typical_noise_norm": sc * np.sqrt(D),
            **_diag_summary(rounds), "split_fingerprint": fp}
    tag = f"{label(epsilon)}_C{rec['clip_norm']}_slr{c['federated']['server_lr']}_seed{seed}"
    out = RAW / "b2_runs"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / f"history_{tag}.csv", index=False, float_format="%.6f")
    pd.DataFrame(rounds).to_csv(out / f"rounds_{tag}.csv", index=False, float_format="%.6f")
    rows = [{**meta, **val_summary}]
    if evaluate_test:
        test_summary, test_ranks = evaluate(sim.full_scores, split, "test", k=k)   # the only test evaluation
        rows.append({**meta, **test_summary})
        per_user_table(split, "test", test_ranks, k).to_csv(out / f"per_user_test_{tag}.csv", index=False,
                                                            float_format="%.6f")
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        save_checkpoint(CHECKPOINTS / f"b2_{label(epsilon)}_seed{seed}.pt", sim, c, data_cfg, seed, fp, rec,
                        val_summary, split)
    per_user_table(split, "validation", val_ranks, k).to_csv(out / f"per_user_validation_{tag}.csv", index=False,
                                                             float_format="%.6f")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"result_{tag}.csv", index=False, float_format="%.6f")
    print(f"{tag}: val ndcg@{k} {val_summary[f'ndcg@{k}']:.4f}"
          + (f"  test ndcg@{k} {rows[1][f'ndcg@{k}']:.4f}" if evaluate_test else "")
          + f"  eps {rec['epsilon']:.3f}  clipped {meta['frac_clipped_mean']:.2f}  snr {meta['signal_to_noise_mean']:.3f}",
          flush=True)
    return df


def run_many(jobs, workers):
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return pd.concat(list(ex.map(run_one, jobs)), ignore_index=True)


# --------------------------------------------------------------------------- validation-only studies

def sanity(clip):
    cfg, _, split, _ = load_all()
    eps = cfg["privacy"]["reference_epsilon"]
    c = run_cfg(cfg, eps, clip)
    rec = privacy_record(c["privacy"], c["federated"]["client_sampling_q"], c["privacy"]["T"])
    tab = sigma_table(cfg)
    sim, history, rounds, _ = train_one(c, split, cfg["seed"], log=print)
    r = pd.DataFrame(rounds)
    h = pd.DataFrame(history)
    n_items, d = split["n_items"], c["model"]["dim"]
    checks = {
        "scores finite": bool(np.isfinite(sim.full_scores(np.arange(split["n_users"]))).all()),
        "validation NDCG@10 above initial (round 0)": bool(h["val_ndcg@10"].iloc[-1] > h["val_ndcg@10"].iloc[0]),
        "clipping active (some clients clipped)": bool(r["frac_clipped"].max() > 0),
        "post-clip norms <= C": bool((r["post_clip_norm_mean"] <= rec["clip_norm"] * (1 + 1e-6)).all()),
        "noise non-zero, std = sigma*C": bool((r["noise_norm"] > 0).all() and
                                              np.allclose(r["noise_std"], rec["noise_multiplier"] * rec["clip_norm"])),
        "accountant eps matches table": bool(abs(rec["epsilon"] - tab.loc[float(eps), "epsilon"]) < 1e-6),
        "exactly T rounds": bool(sim.round == c["privacy"]["T"] and len(r) == c["privacy"]["T"]),
        "comm = clients x 2 x M x d x 4 bytes": bool(((r["download_bytes"] + r["upload_bytes"]) ==
                                                      r["clients"] * 2 * n_items * d * 4).all()),
    }
    pd.DataFrame([{"check": k, "passed": v} for k, v in checks.items()]).to_csv(
        RESULTS / "b2_sanity_checks.csv", index=False)
    pd.DataFrame(history).to_csv(RAW / "b2_sanity_history.csv", index=False, float_format="%.6f")
    for k, v in checks.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")
    print(f"final val ndcg@10 {h['val_ndcg@10'].iloc[-1]:.4f} (round 0: {h['val_ndcg@10'].iloc[0]:.4f}); "
          f"eps {rec['epsilon']:.4f} sigma {rec['noise_multiplier']:.4f} C {rec['clip_norm']}")
    if not all(checks.values()):
        raise SystemExit("sanity checks failed")


def clip_search(workers):
    cfg = load_config(ROOT / CONFIG)
    eps, seed = cfg["privacy"]["reference_epsilon"], cfg["seed"]
    df = run_many([(eps, seed, C, None, False) for C in cfg["privacy"]["clip_grid"]], workers)
    df = df.sort_values("ndcg@10", ascending=False, kind="mergesort")
    keep = ["clip_norm", "noise_multiplier", "epsilon", "ndcg@10", "hr@10", "recall@10", "mrr@10",
            "frac_clipped_mean", "shrinkage_mean", "signal_to_noise_mean"]
    df[keep].to_csv(RESULTS / "b2_clip_search.csv", index=False, float_format="%.6f")
    print(df[keep].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"selected C (highest final-round validation NDCG@10 at eps≈{eps}): {df.iloc[0]['clip_norm']}")


def server_lr_check(workers):
    cfg = load_config(ROOT / CONFIG)
    eps, seed = cfg["privacy"]["reference_epsilon"], cfg["seed"]
    df = run_many([(eps, seed, None, s, False) for s in cfg["privacy"]["server_lr_grid"]], workers)
    keep = ["server_lr", "clip_norm", "ndcg@10", "hr@10", "mrr@10", "signal_to_noise_mean"]
    df[keep].sort_values("ndcg@10", ascending=False).to_csv(RESULTS / "b2_server_lr_check.csv", index=False,
                                                              float_format="%.6f")
    print(df[keep].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- sweep + analysis

def sweep(seeds, workers):
    cfg = load_config(ROOT / CONFIG)
    levels = [None] + list(cfg["privacy"]["target_epsilons"])
    jobs = [(e, s, None, None, True) for s in seeds for e in levels]
    if float(cfg["federated"]["server_lr"]) != B1_SERVER_LR:      # B1-DPReady: the spec's control, B1's eta_s
        jobs += [(None, s, None, B1_SERVER_LR, True) for s in seeds]
    run_many(jobs, workers)
    analyse(seeds)


def paired(d, seed=1, n_boot=10000):
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_improved": float((d > 0).mean()), "frac_degraded": float((d < 0).mean()),
            "frac_tied": float((d == 0).mean())}


def analyse(seeds):
    cfg = load_config(ROOT / CONFIG)
    C = cfg["privacy"]["clip_norm"]
    slr = float(cfg["federated"]["server_lr"])
    levels = [None] + list(cfg["privacy"]["target_epsilons"])
    k = cfg["evaluation"]["k"]
    tags = {label(e): [f"{label(e)}_C{None if e is None else float(C)}_slr{slr}_seed{s}" for s in seeds] for e in levels}
    if slr != B1_SERVER_LR:   # the spec's B1-DPReady control (B1's eta_s) is reported separately
        tags["nodp_b1dpready"] = [f"nodp_CNone_slr{B1_SERVER_LR}_seed{s}" for s in seeds]
    runs = pd.concat([pd.read_csv(RAW / "b2_runs" / f"result_{t}.csv").assign(privacy_level=lvl)
                      for lvl, ts in tags.items() for t in ts], ignore_index=True)
    runs.to_csv(RESULTS / "b2_privacy_sweep.csv", index=False, float_format="%.6f")

    rows = []
    for (lvl, sp), g in runs.groupby(["privacy_level", "split"], sort=False):
        row = {"privacy_level": lvl, "split": sp, "n_seeds": len(g), "server_lr": g["server_lr"].iloc[0],
               "target_epsilon": g["target_epsilon"].iloc[0], "epsilon": g["epsilon"].iloc[0],
               "epsilon_cross_check": g["epsilon_cross_check"].iloc[0],
               "noise_multiplier": g["noise_multiplier"].iloc[0], "clip_norm": g["clip_norm"].iloc[0],
               "typical_noise_norm": g["typical_noise_norm"].iloc[0], "shared_coordinates_D": g["shared_coordinates_D"].iloc[0]}
        for m in [f"{x}@{k}" for x in METRICS] + ["frac_clipped_mean", "signal_to_noise_mean", "noise_norm_mean",
                                                  "clipped_aggregate_norm_mean", "shrinkage_mean",
                                                  "pre_clip_norm_median_mean", "comm_bytes_total"]:
            row[f"{m}_mean"], row[f"{m}_std"] = g[m].mean(), g[m].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    ctrl = summary[summary.privacy_level == "nodp"].set_index("split")[f"ndcg@{k}_mean"]
    summary["ndcg_loss_vs_matched_nodp"] = summary.apply(lambda r: ctrl[r["split"]] - r[f"ndcg@{k}_mean"], axis=1)
    summary["ndcg_retention_vs_matched_nodp"] = summary.apply(lambda r: r[f"ndcg@{k}_mean"] / ctrl[r["split"]], axis=1)
    summary.to_csv(RESULTS / "b2_summary.csv", index=False, float_format="%.6f")

    per = lambda lvl: np.mean([pd.read_csv(RAW / "b2_runs" / f"per_user_test_{t}.csv").sort_values("user")[f"ndcg@{k}"]
                               .to_numpy() for t in tags[lvl]], axis=0)
    ref = per("nodp")                                 # matched control: B2 protocol (incl. eta_s) without DP
    b1 = np.mean([pd.read_csv(RAW / f"b1_per_user_test_seed{s}.csv").sort_values("user")[f"ndcg@{k}"].to_numpy()
                  for s in seeds], axis=0)
    pairs, cols = [], {"user": np.arange(len(ref)), "b1_frozen": b1, "nodp_matched": ref}
    if "nodp_b1dpready" in tags:
        dpready = per("nodp_b1dpready")
        cols["nodp_b1dpready"] = dpready
        pairs += [{"comparison": "B1-DPReady (eta_s=1, qN, T=1000) - B1 frozen  [protocol effect]", **paired(dpready - b1)},
                  {"comparison": f"matched no-DP (eta_s={slr:g}) - B1-DPReady  [server-lr effect]", **paired(ref - dpready)}]
    else:
        pairs += [{"comparison": "matched no-DP (B1-DPReady) - B1 frozen  [protocol effect]", **paired(ref - b1)}]
    for e in levels[1:]:
        cols[label(e)] = per(label(e))
        pairs.append({"comparison": f"B2 {label(e)} - matched no-DP  [DP effect]", **paired(cols[label(e)] - ref)})
    pd.DataFrame(pairs).to_csv(RESULTS / "b2_paired.csv", index=False, float_format="%.6f")
    pd.DataFrame(cols).to_csv(RESULTS / "b2_per_user_test.csv", index=False, float_format="%.6f")

    diag = pd.concat([pd.read_csv(RAW / "b2_runs" / f"rounds_{t}.csv").assign(privacy_level=lvl,
                                                                             seed=int(t.rsplit("seed", 1)[1]))
                      for lvl, ts in tags.items() for t in ts], ignore_index=True)
    diag.to_csv(RESULTS / "b2_diagnostics.csv", index=False, float_format="%.6f")
    plots(summary, diag, k)
    pd.set_option("display.width", 250)
    print(summary[summary.split == "test"][["privacy_level", "server_lr", "epsilon", "noise_multiplier",
                                             f"ndcg@{k}_mean", f"ndcg@{k}_std", f"hr@{k}_mean", f"mrr@{k}_mean",
                                             "ndcg_retention_vs_matched_nodp", "frac_clipped_mean_mean",
                                             "signal_to_noise_mean_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(pairs).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def plots(summary, diag, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    t = summary[summary.split == "test"].copy()
    dp = t[~t.privacy_level.str.startswith("nodp")].sort_values("epsilon")
    nodp = t[t.privacy_level == "nodp"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.errorbar(dp["epsilon"], dp[f"ndcg@{k}_mean"], yerr=dp[f"ndcg@{k}_std"], marker="o", capsize=3, label="B2 (DP)")
    ax.axhline(nodp[f"ndcg@{k}_mean"], color="C2", linestyle="--", label="matched no-DP control")
    ax.axhline(0.0443, color="C3", linestyle=":", label="popularity")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dp["epsilon"])
    ax.set_xticklabels([f"{e:.2g}" for e in dp["epsilon"]])
    ax.set_xlabel("ε (user-level, δ=1e-5; smaller = stronger privacy)")
    ax.set_ylabel(f"test NDCG@{k} (mean ± std over seeds)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(dp["epsilon"], dp["ndcg_retention_vs_matched_nodp"], marker="o")
    ax.axhline(1.0, color="C2", linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dp["epsilon"])
    ax.set_xticklabels([f"{e:.2g}" for e in dp["epsilon"]])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("ε")
    ax.set_ylabel("NDCG@10 retention vs matched no-DP")
    fig.suptitle("B2: privacy–utility trade-off (q=0.1, T=1000, fixed C)")
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "b2_privacy_utility.png", dpi=120)
    plt.close(fig)

    d = diag[~diag.privacy_level.str.startswith("nodp")]
    order = list(dp["privacy_level"])
    for col, ylabel, fname in [("frac_clipped", "fraction of selected clients clipped", "b2_clipping.png"),
                               ("signal_to_noise", "||clipped aggregate|| / ||noise||", "b2_signal_noise.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for lvl in order:
            g = d[d.privacy_level == lvl].groupby("round")[col].mean().rolling(10, min_periods=1).mean()
            ax.plot(g.index, g.values, label=lvl)
        ax.set_xlabel("private round (mean over seeds, 10-round moving average)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
        ax.set_title(f"B2: {ylabel}")
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / fname, dpi=120)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--accounting", action="store_true")
    p.add_argument("--norm-stats", action="store_true")
    p.add_argument("--sanity", action="store_true")
    p.add_argument("--clip", type=float)
    p.add_argument("--clip-search", action="store_true")
    p.add_argument("--server-lr-check", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--epsilon", default=None, help="target epsilon, or 'none' for the no-DP control")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--analyse", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2026])
    p.add_argument("--workers", type=int, default=15)
    a = p.parse_args()
    if a.accounting:
        accounting()
    elif a.norm_stats:
        norm_stats()
    elif a.sanity:
        sanity(a.clip)
    elif a.clip_search:
        clip_search(a.workers)
    elif a.server_lr_check:
        server_lr_check(a.workers)
    elif a.run:
        eps = None if a.epsilon in (None, "none") else float(a.epsilon)
        run_one((eps, a.seed, None, None, True))
    elif a.sweep:
        sweep(a.seeds, a.workers)
    elif a.analyse:
        analyse(a.seeds)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
