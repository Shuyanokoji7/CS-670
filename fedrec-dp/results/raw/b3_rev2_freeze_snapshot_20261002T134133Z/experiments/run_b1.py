"""B1 — federated BPR-MF (no DP) experiment runner.

    # validation-only staged search (never touches test)
    python experiments/run_b1.py --config configs/b1.yaml --search

    # validation-only convergence check of one configuration with a larger round budget
    python experiments/run_b1.py --config configs/b1.yaml --check Q LR EPOCHS --max-rounds N

    # train the frozen config; test evaluated ONCE on the best-validation checkpoint
    python experiments/run_b1.py --config configs/b1.yaml --seed 42

    # several seeds (+ summary, B0 comparison, paired analysis, personalisation check)
    python experiments/run_b1.py --config configs/b1.yaml --seeds 42 123 2026

    # aggregation diagnostic (item-wise mean instead of equal-user FedAvg), validation only
    python experiments/run_b1.py --config configs/b1.yaml --aggregation-diagnostic
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # single-threaded BLAS: deterministic sums, parallel runs don't fight

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import metrics  # noqa: E402
from src.data import load_processed, previously_rated, split_fingerprint  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.federated import FederatedBPR  # noqa: E402
from src.train_federated import METRICS, train_federated, validation_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
CHECKPOINTS = ROOT / "checkpoints"


def load_all(config_path):
    cfg = load_config(ROOT / config_path)
    data_cfg = load_config(ROOT / cfg["dataset_config"])
    split = load_processed(data_cfg)
    return cfg, data_cfg, split, split_fingerprint(split)


def with_overrides(cfg, **fl):
    c = copy.deepcopy(cfg)
    c["federated"].update(fl)
    return c


def run_training(cfg, split, seed, log=print):
    k = cfg["evaluation"]["k"]
    return train_federated(split["train"], split["n_users"], split["n_items"], cfg, seed,
                           lambda sim: validation_metrics(sim, split, k)[0], log)


def fl_name(fl):
    return f"q{fl['client_sampling_q']}_lr{fl['local_lr']}_E{fl['local_epochs']}_{fl.get('aggregation', 'fedavg')}"


# --------------------------------------------------------------------------- search

def _search_worker(args):
    config_path, overrides, stage, seed, out_dir = args
    cfg, _, split, _ = load_all(config_path)
    c = with_overrides(cfg, **overrides)
    k = c["evaluation"]["k"]
    _, history, rounds_log, best = run_training(c, split, seed, log=lambda *_: None)
    fl = c["federated"]
    pd.DataFrame(history).to_csv(Path(out_dir) / f"history_{fl_name(fl)}.csv", index=False, float_format="%.6f")
    at_best = next((h for h in history if h["round"] == best["round"]), history[0])
    return {
        "stage": stage, "client_sampling_q": fl["client_sampling_q"], "local_lr": fl["local_lr"],
        "local_epochs": fl["local_epochs"], "l2_reg": fl["l2_reg"], "aggregation": fl["aggregation"],
        "seed": seed, "best_round": best["round"], "rounds_run": best["rounds_run"],
        "stopped_by": best["stopped_by"],
        **{f"val_{m}@{k}": (at_best.get(f"val_{m}@{k}", np.nan) if best["stopped_by"] != "diverged" or
                            best["round"] > 0 else np.nan) for m in METRICS},
        "mean_clients_per_round": float(np.mean([r["clients"] for r in rounds_log])) if rounds_log else np.nan,
        "total_comm_bytes": sum(r.get("download_bytes", 0) + r.get("upload_bytes", 0) for r in rounds_log),
    }


def _run_stage(config_path, stage, grid, seed, workers):
    out_dir = RAW / "b1_search"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(config_path, g, stage, seed, str(out_dir)) for g in grid]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        rows = list(ex.map(_search_worker, jobs))
    for r in rows:
        print(f"{stage}  q={r['client_sampling_q']:<5} lr={r['local_lr']:<5} E={r['local_epochs']}  "
              f"best round {r['best_round']:5d}/{r['rounds_run']:5d} ({r['stopped_by']})  "
              f"val ndcg {r['val_ndcg@10']:.4f}")
    return rows


def _best(rows, k=10):
    ok = [r for r in rows if r["stopped_by"] != "diverged"]
    return max(ok, key=lambda r: (r[f"val_ndcg@{k}"], -r["local_epochs"]))


def search(config_path, workers):
    cfg = load_config(ROOT / config_path)
    sc = cfg["search"]
    seed = sc["seed"]
    rows, done = [], set()

    def run(stage, grid):
        grid = [g for g in grid if (g["client_sampling_q"], g["local_lr"], g["local_epochs"]) not in done]
        for g in grid:
            done.add((g["client_sampling_q"], g["local_lr"], g["local_epochs"]))
        if grid:
            rows.extend(_run_stage(config_path, stage, grid, seed, workers))

    s1 = sc["stage1"]
    run("stage1", [{"client_sampling_q": q, "local_lr": lr, "local_epochs": 1}
                   for q, lr in itertools.product(s1["client_sampling_q"], s1["local_lr"])])
    b1 = _best(rows)
    s2 = sc["stage2"]
    run("stage2", [{"client_sampling_q": b1["client_sampling_q"], "local_lr": b1["local_lr"] * f, "local_epochs": e}
                   for e, f in itertools.product(s2["local_epochs"], s2["lr_factors"])])
    b2 = _best(rows)
    lrs = sorted({r["local_lr"] for r in rows if r["client_sampling_q"] == b2["client_sampling_q"]
                  and r["local_epochs"] == b2["local_epochs"]})
    if b2["local_lr"] in (lrs[0], lrs[-1]):                  # stage 3 only if the best lr is at a grid edge
        f = sc["stage3"]["edge_lr_factor"]
        nxt = b2["local_lr"] * f if b2["local_lr"] == lrs[-1] else b2["local_lr"] / f
        run("stage3", [{"client_sampling_q": b2["client_sampling_q"], "local_lr": nxt,
                        "local_epochs": b2["local_epochs"]}])

    k = cfg["evaluation"]["k"]
    df = pd.DataFrame(rows).sort_values([f"val_ndcg@{k}", "local_epochs"], ascending=[False, True],
                                        kind="mergesort", na_position="last")
    df.insert(0, "rank_by_val_ndcg", range(1, len(df) + 1))
    top = _best(rows)
    df.insert(1, "selected", [(r.client_sampling_q, r.local_lr, r.local_epochs) ==
                              (top["client_sampling_q"], top["local_lr"], top["local_epochs"])
                              for r in df.itertuples()])
    df.to_csv(RESULTS / "b1_hyperparameter_search.csv", index=False, float_format="%.6f")
    print("\n" + df.drop(columns=["seed", "l2_reg", "aggregation"]).to_string(index=False))
    print(f"\nSelected (highest validation NDCG@{k}): q={top['client_sampling_q']} lr={top['local_lr']} "
          f"E={top['local_epochs']}")
    if top["stopped_by"] != "early_stopping":
        raise RuntimeError("the best configuration did not converge by early stopping; raise max_rounds and "
                           "re-run (or use --check) instead of selecting a censored run")


def check(config_path, q, lr, epochs, max_rounds):
    cfg, _, split, _ = load_all(config_path)
    c = with_overrides(cfg, client_sampling_q=q, local_lr=lr, local_epochs=epochs, max_rounds=max_rounds)
    _, history, _, best = run_training(c, split, cfg["search"]["seed"], log=lambda *_: None)
    out = RAW / "b1_search" / f"convergence_check_{fl_name(c['federated'])}_max{max_rounds}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out, index=False, float_format="%.6f")
    print(f"q={q} lr={lr} E={epochs} max_rounds={max_rounds}: best round {best['round']}, rounds run "
          f"{best['rounds_run']}, stopped by {best['stopped_by']}, best val ndcg@10 {best['ndcg@10']:.6f}\n-> {out}")


# --------------------------------------------------------------------------- single run

def per_user_table(split, target, ranks, k):
    t = split[target].sort_values("user").reset_index(drop=True)
    return pd.DataFrame({
        "user": t["user"], "user_id": t["user_id"], "item": t["item"], "item_id": t["item_id"],
        "n_candidates": [split["n_items"] - len(x) for x in previously_rated(split, target)],
        "rank": ranks,
        f"ndcg@{k}": metrics.ndcg_from_rank(ranks, k), f"hr@{k}": metrics.hit_rate_from_rank(ranks, k),
        f"recall@{k}": metrics.recall_from_rank(ranks, k), f"mrr@{k}": metrics.mrr_from_rank(ranks, k),
    })


def save_checkpoint(path, sim, cfg, data_cfg, seed, fp, best, val_summary, split):
    fl = cfg["federated"]
    torch.save({
        "P": torch.from_numpy(sim.P.copy()),       # every client's local p_u (simulation only)
        "Q": torch.from_numpy(sim.Q.copy()),       # global item model
        "model": "FederatedBPRMF",
        "config": cfg, "dataset_config": data_cfg, "seed": seed, "split_fingerprint": fp,
        "round": best["round"], "rounds_run": best["rounds_run"], "stopped_by": best["stopped_by"],
        "validation": val_summary,
        "client_sampling": {"scheme": "poisson", "q": fl["client_sampling_q"], "empty_rounds": "skipped"},
        "n_users": split["n_users"], "n_items": split["n_items"],
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
    }, path)


def load_checkpoint(path, split):
    """Load a B1 checkpoint (safe weights-only), refusing it if trained on a different split."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt["split_fingerprint"] != split_fingerprint(split):
        raise ValueError(f"{path} was trained on a different data split")
    ckpt["P"], ckpt["Q"] = ckpt["P"].numpy(), ckpt["Q"].numpy()
    sim = FederatedBPR(split["train"], split["n_users"], split["n_items"], ckpt["config"], ckpt["seed"])
    sim.load_state({"P": ckpt["P"], "Q": ckpt["Q"]})
    sim.round = ckpt["round"]
    return sim, ckpt


def plot_run(history, rounds_log, best_round, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(history)
    b0 = pd.read_csv(RESULTS / "b0_summary.csv").set_index("split")
    pop = pd.read_csv(RESULTS / "popularity_baseline.csv").set_index("split")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h["round"], h[f"val_ndcg@{k}"], marker=".", markersize=3, label="B1 federated")
    ax.axhline(b0.loc["validation", f"ndcg@{k}_mean"], color="C2", linestyle=":", label="B0 centralised (3-seed mean)")
    ax.axhline(pop.loc["validation", f"ndcg@{k}"], color="C3", linestyle=":", label="popularity")
    ax.axvline(best_round, color="gray", linestyle="--", label=f"best round ({best_round})")
    ax.set_xlabel("federated round")
    ax.set_ylabel(f"validation NDCG@{k}")
    ax.set_title(f"B1 federated BPR-MF: validation NDCG@{k}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "b1_validation_ndcg.png", dpi=120)
    plt.close(fig)

    r = pd.DataFrame(rounds_log)
    cum = (r["download_bytes"] + r["upload_bytes"]).cumsum() / 1e9
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r["round"], cum)
    ax.axvline(best_round, color="gray", linestyle="--", label=f"best round ({best_round})")
    ax.set_xlabel("federated round")
    ax.set_ylabel("cumulative simulated tensor volume (GB)")
    ax.set_title("B1: simulated communication (download + upload, float32)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "plots" / "b1_communication.png", dpi=120)
    plt.close(fig)


def run(config_path, seed, primary):
    cfg, data_cfg, split, fp = load_all(config_path)
    k = cfg["evaluation"]["k"]
    sim, history, rounds_log, best = run_training(cfg, split, seed, log=lambda *_: None)
    if best["stopped_by"] != "early_stopping":
        raise RuntimeError(f"seed {seed}: run ended by {best['stopped_by']}, not early stopping")

    val_summary, val_ranks = validation_metrics(sim, split, k)
    test_summary, test_ranks = evaluate(sim.full_scores, split, "test", k=k)    # the only test evaluation

    comm = pd.DataFrame(rounds_log)
    comm["cum_download_bytes"] = comm["download_bytes"].cumsum()
    comm["cum_upload_bytes"] = comm["upload_bytes"].cumsum()
    to_best = comm[comm["round"] <= best["round"]]
    fl = cfg["federated"]
    meta = {"model": "b1_federated_bpr", "seed": seed, "best_round": best["round"], "rounds_run": best["rounds_run"],
            "client_sampling_q": fl["client_sampling_q"], "local_lr": fl["local_lr"],
            "local_epochs": fl["local_epochs"], "l2_reg": fl["l2_reg"], "dim": cfg["model"]["dim"],
            "mean_clients_per_round": float(comm["clients"].mean()),
            "empty_rounds": int((comm["clients"] == 0).sum()),
            "bytes_per_client_per_round": int(comm["bytes_per_client"].max()),
            "comm_bytes_to_best_round": int((to_best["download_bytes"] + to_best["upload_bytes"]).sum()),
            "comm_bytes_total_run": int((comm["download_bytes"] + comm["upload_bytes"]).sum()),
            "split_fingerprint": fp}
    both = pd.DataFrame([{**meta, **val_summary}, {**meta, **test_summary}])

    RAW.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    both.to_csv(RAW / f"b1_seed{seed}.csv", index=False, float_format="%.6f")
    pd.DataFrame(history).to_csv(RAW / f"b1_training_history_seed{seed}.csv", index=False, float_format="%.6f")
    comm.to_csv(RAW / f"b1_communication_seed{seed}.csv", index=False, float_format="%.6f")
    per_user = per_user_table(split, "test", test_ranks, k)
    per_user.to_csv(RAW / f"b1_per_user_test_seed{seed}.csv", index=False, float_format="%.6f")
    per_user_table(split, "validation", val_ranks, k).to_csv(
        RAW / f"b1_per_user_validation_seed{seed}.csv", index=False, float_format="%.6f")
    save_checkpoint(CHECKPOINTS / f"b1_seed{seed}.pt", sim, cfg, data_cfg, seed, fp, best, val_summary, split)
    if primary:
        both.iloc[[0]].to_csv(RESULTS / "b1_validation_results.csv", index=False, float_format="%.6f")
        both.iloc[[1]].to_csv(RESULTS / "b1_test_results.csv", index=False, float_format="%.6f")
        pd.DataFrame(history).to_csv(RESULTS / "b1_training_history.csv", index=False, float_format="%.6f")
        comm.to_csv(RESULTS / "b1_communication.csv", index=False, float_format="%.6f")
        per_user.to_csv(RESULTS / "b1_per_user_test.csv", index=False, float_format="%.6f")
        save_checkpoint(CHECKPOINTS / "b1_best.pt", sim, cfg, data_cfg, seed, fp, best, val_summary, split)
        plot_run(history, rounds_log, best["round"], k)
    cols = ["split", f"ndcg@{k}", f"hr@{k}", f"recall@{k}", f"mrr@{k}"]
    print(f"seed {seed}: best round {best['round']} of {best['rounds_run']}")
    print(both[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return both


def _run_worker(args):
    return run(*args)


# --------------------------------------------------------------------------- analysis

def paired(d, rng, n_boot=10000):
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "frac_improved": float((d > 0).mean()), "frac_degraded": float((d < 0).mean()),
            "frac_tied": float((d == 0).mean())}


def analyse(seeds, k=10):
    runs = pd.concat([pd.read_csv(RAW / f"b1_seed{s}.csv") for s in seeds], ignore_index=True)
    cols = [f"{m}@{k}" for m in METRICS] + ["best_round", "mean_clients_per_round", "comm_bytes_to_best_round"]
    rows = []
    for split_name, g in runs.groupby("split", sort=False):
        row = {"model": "b1_federated_bpr", "split": split_name, "n_seeds": len(g),
               "seeds": " ".join(str(s) for s in g["seed"])}
        for c in cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "b1_summary.csv", index=False, float_format="%.6f")

    b0 = pd.read_csv(RESULTS / "b0_summary.csv").set_index("split")
    comp = []
    for name, f in [("random", "random_baseline.csv"), ("popularity", "popularity_baseline.csv")]:
        for _, r in pd.read_csv(RESULTS / f).iterrows():
            comp.append({"model": name, "split": r["split"], "n_seeds": 1,
                         **{f"{m}@{k}": r[f"{m}@{k}"] for m in METRICS}})
    for model, s in [("b0_bpr", b0.reset_index()), ("b1_federated_bpr", summary)]:
        for _, r in s.iterrows():
            comp.append({"model": model, "split": r["split"], "n_seeds": r["n_seeds"],
                         **{f"{m}@{k}": r[f"{m}@{k}_mean"] for m in METRICS},
                         **{f"{m}@{k}_std": r[f"{m}@{k}_std"] for m in METRICS}})
    comp = pd.DataFrame(comp)
    for sp in ("validation", "test"):
        b0v = b0.loc[sp, f"ndcg@{k}_mean"]
        b1v = summary.set_index("split").loc[sp, f"ndcg@{k}_mean"]
        comp.loc[(comp.model == "b1_federated_bpr") & (comp.split == sp), "ndcg_retention_vs_b0"] = b1v / b0v
        comp.loc[(comp.model == "b1_federated_bpr") & (comp.split == sp), "federation_induced_gap"] = (b0v - b1v) / b0v
    comp.to_csv(RESULTS / "b1_comparison.csv", index=False, float_format="%.6f")

    # paired per-user test comparison, seed-averaged (B0 uses its frozen seeds 42/123/2026)
    per = lambda prefix, ss: np.mean([pd.read_csv(RAW / f"{prefix}_per_user_test_seed{x}.csv")
                                      .sort_values("user")[f"ndcg@{k}"].to_numpy() for x in ss], 0)
    b1u, b0u = per("b1", seeds), per("b0", [42, 123, 2026])
    split = load_processed()
    rng = np.random.default_rng(1)
    pr = paired(b1u - b0u, rng)
    out = [{"comparison": "B1 - B0 (test NDCG@10, seed-averaged per user)", **pr}]

    # activity groups: fixed tercile cut-points of the number of training positives
    n_train = split["train"].groupby("user").size().reindex(range(split["n_users"])).to_numpy()
    cuts = np.quantile(n_train, [1 / 3, 2 / 3])
    group = np.where(n_train <= cuts[0], "low", np.where(n_train <= cuts[1], "medium", "high"))
    act = []
    for gname in ("low", "medium", "high"):
        m = group == gname
        d = paired(b1u[m] - b0u[m], rng)
        act.append({"group": gname, "n_users": int(m.sum()), "train_pos_min": int(n_train[m].min()),
                    "train_pos_max": int(n_train[m].max()), "b0_ndcg@10": float(b0u[m].mean()),
                    "b1_ndcg@10": float(b1u[m].mean()), **d})
    pd.DataFrame(act).to_csv(RESULTS / "b1_activity_groups.csv", index=False, float_format="%.6f")

    # personalisation sanity check on the primary checkpoint
    sim, _ = load_checkpoint(CHECKPOINTS / "b1_best.pt", split)
    P, Q = sim.P, sim.Q
    perm = np.random.default_rng(0).permutation(split["n_users"])
    checks = {
        "own_p_u": evaluate(lambda u: (P[u] @ Q.T).astype(np.float64), split, "test", k=k)[0][f"ndcg@{k}"],
        "random_other_p_v": evaluate(lambda u: (P[perm[u]] @ Q.T).astype(np.float64), split, "test", k=k)[0][f"ndcg@{k}"],
        "mean_user_vector": evaluate(lambda u: np.repeat(P.mean(0, keepdims=True) @ Q.T, len(u), 0).astype(np.float64),
                                     split, "test", k=k)[0][f"ndcg@{k}"],
    }
    pd.DataFrame([{"check": c, f"test_ndcg@{k}": v} for c, v in checks.items()]).to_csv(
        RESULTS / "b1_personalisation_check.csv", index=False, float_format="%.6f")
    pd.DataFrame(out).to_csv(RESULTS / "b1_paired_vs_b0.csv", index=False, float_format="%.6f")

    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + pd.DataFrame(act).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\npersonalisation check (seed of b1_best.pt):", {c: round(v, 4) for c, v in checks.items()})


def aggregation_diagnostic(config_path):
    """Frozen config with item-wise mean aggregation; validation only, not eligible for selection."""
    cfg, _, split, _ = load_all(config_path)
    rows = []
    for agg in ("fedavg", "item_mean"):
        c = with_overrides(cfg, aggregation=agg)
        _, history, _, best = run_training(c, split, cfg["seed"], log=lambda *_: None)
        rows.append({"aggregation": agg, "best_round": best["round"], "rounds_run": best["rounds_run"],
                     "stopped_by": best["stopped_by"], "val_ndcg@10": best["ndcg@10"]})
        pd.DataFrame(history).to_csv(RAW / f"b1_aggregation_diag_{agg}.csv", index=False, float_format="%.6f")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b1_aggregation_diagnostic.csv", index=False, float_format="%.6f")
    print(df.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/b1.yaml")
    p.add_argument("--search", action="store_true")
    p.add_argument("--check", type=float, nargs=3, metavar=("Q", "LR", "EPOCHS"))
    p.add_argument("--max-rounds", type=int, default=16000)
    p.add_argument("--seed", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--aggregation-diagnostic", action="store_true")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    if args.search:
        return search(args.config, args.workers)
    if args.check:
        return check(args.config, args.check[0], args.check[1], int(args.check[2]), args.max_rounds)
    if args.aggregation_diagnostic:
        return aggregation_diagnostic(args.config)
    cfg = load_config(ROOT / args.config)
    if args.seeds:
        jobs = [(args.config, s, s == cfg["seed"]) for s in args.seeds]
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            list(ex.map(_run_worker, jobs))
        analyse(args.seeds, cfg["evaluation"]["k"])
    else:
        seed = args.seed if args.seed is not None else cfg["seed"]
        run(args.config, seed, primary=(seed == cfg["seed"]))


if __name__ == "__main__":
    main()
