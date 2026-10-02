"""B0 — centralised BPR-MF experiment runner.

    # 1. validation-only hyperparameter search (never touches test)
    python experiments/run_b0.py --config configs/b0.yaml --search

    # 2. train with the frozen config; test is evaluated ONCE on the best-validation checkpoint
    python experiments/run_b0.py --config configs/b0.yaml --seed 42

    # 3. several seeds + mean/std summary and comparison with the baselines
    python experiments/run_b0.py --config configs/b0.yaml --seeds 42 123 2026
"""

import argparse
import copy
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import metrics  # noqa: E402
from src.data import load_processed, previously_rated, split_fingerprint  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.train_bpr import build_model, train_bpr, validation_metrics  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
CHECKPOINTS = ROOT / "checkpoints"
METRIC_NAMES = ["ndcg", "hr", "recall", "mrr"]
# Output naming; set from the config (`output_prefix`, `model_name`) in main(). Defaults reproduce B0's files.
PREFIX = "b0"
MODEL_NAME = "b0_bpr"


def load_all(config_path):
    cfg = load_config(ROOT / config_path)
    data_cfg = load_config(ROOT / cfg["dataset_config"])
    split = load_processed(data_cfg)
    fp = split_fingerprint(split)
    return cfg, data_cfg, split, fp


def run_training(cfg, split, seed, log=print):
    k = cfg["evaluation"]["k"]

    def validate(model):
        return validation_metrics(model, split, k)[0]

    return train_bpr(split["train"], split["n_users"], split["n_items"], cfg, seed, validate, log)


# --------------------------------------------------------------------------- search

def _search_one(args):
    config_path, stage_name, lr, reg, seed, prefix, model_name = args
    global PREFIX, MODEL_NAME
    PREFIX, MODEL_NAME = prefix, model_name
    cfg, _, split, _ = load_all(config_path)
    k = cfg["evaluation"]["k"]
    c = copy.deepcopy(cfg)
    c["training"]["learning_rate"], c["training"]["l2_reg"] = lr, reg
    _, history, best = run_training(c, split, seed, log=lambda *_: None)
    pd.DataFrame(history).to_csv(RAW / f"{PREFIX}_search" / f"history_lr{lr}_reg{reg}.csv", index=False)
    at_best = history[best["epoch"] - 1]
    row = {
        "stage": stage_name, "learning_rate": lr, "l2_reg": reg,
        "batch_size": c["training"]["batch_size"], "dim": c["model"]["dim"], "seed": seed,
        "best_epoch": best["epoch"], "epochs_run": len(history),
        "stopped_by": "max_epochs" if len(history) >= c["training"]["max_epochs"] else "early_stopping",
        **{f"val_{m}@{k}": at_best[f"val_{m}@{k}"] for m in METRIC_NAMES},
        "train_bpr_loss_at_best": at_best["bpr_loss"],
    }
    print(f"{stage_name}  lr={lr:<7} l2={reg:<7}  best epoch {best['epoch']:3d}/{len(history):3d}"
          f"  val ndcg@{k} {at_best[f'val_ndcg@{k}']:.6f}", flush=True)
    return row


def search(config_path, workers=1):
    cfg, _, split, _ = load_all(config_path)
    k = cfg["evaluation"]["k"]
    seed = cfg["search"]["seed"]
    (RAW / f"{PREFIX}_search").mkdir(parents=True, exist_ok=True)
    jobs, done = [], set()
    for stage in cfg["search"]["stages"]:
        for lr, reg in itertools.product(stage["learning_rate"], stage["l2_reg"]):
            if (lr, reg) in done:
                continue
            done.add((lr, reg))
            jobs.append((config_path, stage["name"], lr, reg, seed, PREFIX, MODEL_NAME))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_search_one, jobs))
    else:
        rows = [_search_one(j) for j in jobs]
    df = pd.DataFrame(rows).sort_values([f"val_ndcg@{k}", "l2_reg"], ascending=False, kind="mergesort")
    df.insert(0, "rank_by_val_ndcg", range(1, len(df) + 1))
    df.insert(1, "selected", [True] + [False] * (len(df) - 1))
    df.to_csv(RESULTS / f"{PREFIX}_hyperparameter_search.csv", index=False, float_format="%.6f")
    print("\n" + df.to_string(index=False))
    top = df.iloc[0]
    print(f"\nSelected (highest validation NDCG@{k}): lr={top.learning_rate} l2_reg={top.l2_reg}")
    if top["stopped_by"] != "early_stopping":
        raise RuntimeError("the best configuration hit max_epochs (not converged); raise training.max_epochs "
                           "and re-run the search instead of selecting a censored run")


def convergence_check(config_path, lr, reg, max_epochs):
    """Validation-only rerun of one search configuration with a larger epoch budget."""
    cfg, _, split, _ = load_all(config_path)
    k = cfg["evaluation"]["k"]
    c = copy.deepcopy(cfg)
    c["training"].update(learning_rate=lr, l2_reg=reg, max_epochs=max_epochs)
    seed = cfg["search"]["seed"]
    _, history, best = run_training(c, split, seed, log=lambda *_: None)
    out = RAW / f"{PREFIX}_search" / f"convergence_check_lr{lr}_reg{reg}_max{max_epochs}.csv"
    pd.DataFrame(history).to_csv(out, index=False, float_format="%.6f")
    stopped = "max_epochs" if len(history) >= max_epochs else "early_stopping"
    print(f"lr={lr} l2={reg} seed={seed} patience={c['training']['patience']} max_epochs={max_epochs}: "
          f"best epoch {best['epoch']}, epochs run {len(history)}, stopped by {stopped}, "
          f"best val ndcg@{k} {best[f'ndcg@{k}']:.6f}")
    print(f"-> {out}")


# --------------------------------------------------------------------------- single run

def per_user_table(split, target, ranks, k):
    t = split[target].sort_values("user").reset_index(drop=True)
    return pd.DataFrame({
        "user": t["user"], "user_id": t["user_id"], "item": t["item"], "item_id": t["item_id"],
        "n_candidates": [split["n_items"] - len(x) for x in previously_rated(split, target)],
        "rank": ranks,
        f"ndcg@{k}": metrics.ndcg_from_rank(ranks, k),
        f"hr@{k}": metrics.hit_rate_from_rank(ranks, k),
        f"recall@{k}": metrics.recall_from_rank(ranks, k),
        f"mrr@{k}": metrics.mrr_from_rank(ranks, k),
    })


def save_checkpoint(path, model, cfg, data_cfg, seed, fp, best, val_summary, split):
    torch.save({
        "state_dict": model.state_dict(),
        "model": "BPRMF",
        "config": cfg,
        "dataset_config": data_cfg,
        "seed": seed,
        "split_fingerprint": fp,
        "n_users": split["n_users"],
        "n_items": split["n_items"],
        "best_epoch": best["epoch"],
        "validation": val_summary,
        "torch_version": str(torch.__version__),
    }, path)


def load_checkpoint(path, split):
    """Load a B0 checkpoint, refusing it if it was trained on a different split."""
    ckpt = torch.load(path, weights_only=True)
    if ckpt["split_fingerprint"] != split_fingerprint(split):
        raise ValueError(f"{path} was trained on a different data split")
    model = build_model(ckpt["n_users"], ckpt["n_items"], ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def plot_history(history, best_epoch, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (RESULTS / "plots").mkdir(parents=True, exist_ok=True)
    h = pd.DataFrame(history)
    for col, ylabel, fname in [("bpr_loss", "mean BPR loss (train)", f"{PREFIX}_training_loss.png"),
                               (f"val_ndcg@{k}", f"validation NDCG@{k}", f"{PREFIX}_validation_ndcg.png")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h["epoch"], h[col], marker=".")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"B0 BPR-MF: {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "plots" / fname, dpi=120)
        plt.close(fig)


def run(config_path, seed, primary):
    cfg, data_cfg, split, fp = load_all(config_path)
    k = cfg["evaluation"]["k"]
    print(f"B0 seed {seed}  split fingerprint {fp}")
    model, history, best = run_training(cfg, split, seed)
    model.eval()

    # Validation of the selected (best) checkpoint, then ONE test evaluation of it.
    val_summary, val_ranks = validation_metrics(model, split, k)
    test_summary, test_ranks = evaluate(model.full_scores, split, "test", k=k)

    meta = {"model": MODEL_NAME, "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history),
            "learning_rate": cfg["training"]["learning_rate"], "l2_reg": cfg["training"]["l2_reg"],
            "dim": cfg["model"]["dim"], "split_fingerprint": fp}
    both = pd.DataFrame([{**meta, **val_summary}, {**meta, **test_summary}])
    RAW.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    both.to_csv(RAW / f"{PREFIX}_seed{seed}.csv", index=False, float_format="%.6f")
    pd.DataFrame(history).to_csv(RAW / f"{PREFIX}_training_history_seed{seed}.csv", index=False, float_format="%.6f")
    per_user = per_user_table(split, "test", test_ranks, k)
    per_user.to_csv(RAW / f"{PREFIX}_per_user_test_seed{seed}.csv", index=False, float_format="%.6f")
    per_user_table(split, "validation", val_ranks, k).to_csv(
        RAW / f"{PREFIX}_per_user_validation_seed{seed}.csv", index=False, float_format="%.6f")
    save_checkpoint(CHECKPOINTS / f"{PREFIX}_seed{seed}.pt", model, cfg, data_cfg, seed, fp, best, val_summary, split)

    if primary:
        both.iloc[[0]].to_csv(RESULTS / f"{PREFIX}_validation_results.csv", index=False, float_format="%.6f")
        both.iloc[[1]].to_csv(RESULTS / f"{PREFIX}_test_results.csv", index=False, float_format="%.6f")
        pd.DataFrame(history).to_csv(RESULTS / f"{PREFIX}_training_history.csv", index=False, float_format="%.6f")
        per_user.to_csv(RESULTS / f"{PREFIX}_per_user_test.csv", index=False, float_format="%.6f")
        save_checkpoint(CHECKPOINTS / f"{PREFIX}_best.pt", model, cfg, data_cfg, seed, fp, best, val_summary, split)
        plot_history(history, best["epoch"], k)

    cols = ["split", f"ndcg@{k}", f"hr@{k}", f"recall@{k}", f"mrr@{k}"]
    print(f"\nbest epoch {best['epoch']} of {len(history)}")
    print(both[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return both


# --------------------------------------------------------------------------- summary

def summarize(seeds, k=10):
    runs = pd.concat([pd.read_csv(RAW / f"{PREFIX}_seed{s}.csv") for s in seeds], ignore_index=True)
    cols = [f"{m}@{k}" for m in METRIC_NAMES] + ["best_epoch"]
    rows = []
    for split_name, g in runs.groupby("split", sort=False):
        row = {"model": MODEL_NAME, "split": split_name, "n_seeds": len(g),
               "seeds": " ".join(str(s) for s in g["seed"])}
        for c in cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1) if len(g) > 1 else 0.0
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / f"{PREFIX}_summary.csv", index=False, float_format="%.6f")

    comp = []
    for name, f in [("random", "random_baseline.csv"), ("popularity", "popularity_baseline.csv")]:
        for _, r in pd.read_csv(RESULTS / f).iterrows():
            comp.append({"model": name, "split": r["split"], "n_seeds": 1,
                         **{f"{m}@{k}": r[f"{m}@{k}"] for m in METRIC_NAMES},
                         **{f"{m}@{k}_std": np.nan for m in METRIC_NAMES}})
    for _, r in summary.iterrows():
        comp.append({"model": MODEL_NAME, "split": r["split"], "n_seeds": r["n_seeds"],
                     **{f"{m}@{k}": r[f"{m}@{k}_mean"] for m in METRIC_NAMES},
                     **{f"{m}@{k}_std": r[f"{m}@{k}_std"] for m in METRIC_NAMES}})
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS / f"{PREFIX}_comparison.csv", index=False, float_format="%.6f")
    print("\n" + summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n" + comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/b0.yaml")
    p.add_argument("--search", action="store_true", help="validation-only hyperparameter search")
    p.add_argument("--seed", type=int, default=None, help="single training seed (default: config seed)")
    p.add_argument("--seeds", type=int, nargs="+", help="run several seeds, then write b0_summary.csv")
    p.add_argument("--workers", type=int, default=1, help="parallel processes for --search (each uses num_threads)")
    p.add_argument("--check", type=float, nargs=2, metavar=("LR", "L2"),
                   help="validation-only convergence check of one configuration (never touches test)")
    p.add_argument("--max-epochs", type=int, default=600, help="epoch budget for --check")
    args = p.parse_args()

    if args.check:
        convergence_check(args.config, args.check[0], args.check[1], args.max_epochs)
        return
    global PREFIX, MODEL_NAME
    run_cfg = load_config(ROOT / args.config)
    PREFIX = run_cfg.get("output_prefix", "b0")
    MODEL_NAME = run_cfg.get("model_name", "b0_bpr")

    if args.search:
        search(args.config, args.workers)
        return
    cfg = load_config(ROOT / args.config)
    seeds = args.seeds or [args.seed if args.seed is not None else cfg["seed"]]
    for s in seeds:
        run(args.config, s, primary=(s == cfg["seed"]))
    if args.seeds:
        summarize(seeds, cfg["evaluation"]["k"])


if __name__ == "__main__":
    main()
