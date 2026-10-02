"""B3 — low-rank federated BPR (no DP). Runner.

    python experiments/run_b3.py --profile              # runtime / memory per rank (validation-free, 50 rounds)
    python experiments/run_b3.py --search --workers N   # per-rank lr selection (validation only)
    python experiments/run_b3.py --final-train --workers N  # FROZEN: 5 seeds x ranks, validation only, checkpoints
    python experiments/run_b3.py --final-score --workers N  # FROZEN: inventory check, then test once per checkpoint
    python experiments/run_b3.py --b1-supplemental      # FROZEN B1 protocol, seeds 7/99, separate tagged paths
Test scoring is refused unless configs/b3.yaml has protocol_frozen: true.
"""

import os

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.threads import force_env  # noqa: E402

force_env()                              # override inherited BLAS/OpenMP thread variables

import argparse  # noqa: E402
import copy  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_processed, split_fingerprint  # noqa: E402
from src.lowrank import LowRankFederatedBPR  # noqa: E402
import torch  # noqa: E402

from src.evaluate import evaluate  # noqa: E402
from src.train_federated import METRICS, validation_metrics  # noqa: E402
from src.train_lowrank import train_lowrank_converge  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS, RAW, CHECKPOINTS = ROOT / "results", ROOT / "results" / "raw", ROOT / "checkpoints"
CONFIG = "configs/b3.yaml"


def load_all():
    cfg = load_config(ROOT / CONFIG)
    data_cfg = load_config(ROOT / cfg["dataset_config"])
    split = load_processed(data_cfg)
    return cfg, data_cfg, split, split_fingerprint(split)


def require_frozen(cfg):
    if not cfg.get("protocol_frozen", False):
        raise RuntimeError("B3 protocol is not frozen (protocol_frozen: false); test scoring refused")


def run_cfg(cfg, rank, lr):
    c = copy.deepcopy(cfg)
    c["model"]["rank"] = int(rank)
    c["federated"]["local_lr"] = float(lr)
    return c


def final_job(cfg, rank, seed):
    """(job config, tag) for a frozen final job; a logged per-job ceiling override gets its own tagged paths."""
    c = run_cfg(cfg, rank, selected_lr(cfg, rank))
    t = tag(rank, c["federated"]["local_lr"], seed)
    ov = (cfg.get("max_rounds_override") or {}).get(f"{int(rank)}_{int(seed)}")
    if ov is not None:
        c["federated"]["max_rounds"] = int(ov)
        t += f"_max{int(ov)}"
    return c, t


def tag(rank, lr, seed):
    return f"r{int(rank)}_lr{float(lr)}_seed{seed}"


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------- profile

def _profile_one(args):
    rank, lr, rounds = args
    cfg, _, split, _ = load_all()
    c = run_cfg(cfg, rank, lr)
    sim = LowRankFederatedBPR(split["train"], split["n_users"], split["n_items"], c, cfg["search"]["seed"])
    t0, status = time.perf_counter(), "ok"
    try:
        with np.errstate(all="ignore"):
            for _ in range(rounds):
                sim.run_round()
    except FloatingPointError as e:
        status = str(e)
    dt = time.perf_counter() - t0
    from src.train_federated import validation_metrics
    t1 = time.perf_counter()
    if status == "ok":
        validation_metrics(sim, split, c["evaluation"]["k"])
    return {"rank": rank, "local_lr": lr, "rounds_done": sim.round, "status": status,
            "sec_per_round": dt / max(sim.round, 1),
            "sec_per_validation": (time.perf_counter() - t1) if status == "ok" else np.nan,
            "peak_rss_mb": peak_rss_mb()}


def profile(workers):
    cfg = load_config(ROOT / CONFIG)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_profile_one, [(r, lr, 50) for r in cfg["model"]["ranks"] for lr in (2.5, 5.0)]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b3_profile.csv", index=False, float_format="%.4f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------- search

def _search_one(args):
    rank, lr, stage = args
    cfg, _, split, _ = load_all()
    c = run_cfg(cfg, rank, lr)
    k = c["evaluation"]["k"]
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        sim, history, rounds, best = train_lowrank_converge(split, c, cfg["search"]["seed"], log=lambda *_: None)
    out = RAW / "b3_search"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / f"history_{tag(rank, lr, cfg['search']['seed'])}.csv", index=False,
                                 float_format="%.6f")
    pd.DataFrame(rounds).to_csv(out / f"rounds_{tag(rank, lr, cfg['search']['seed'])}.csv", index=False,
                                float_format="%.6f")
    at_best = next((h for h in history if h["round"] == best["round"]), history[0])
    row = {"stage": stage, "rank": rank, "local_lr": lr, "seed": cfg["search"]["seed"], "best_round": best["round"],
           "rounds_run": best["rounds_run"], "stopped_by": best["stopped_by"],
           **{f"val_{m}@{k}": at_best.get(f"val_{m}@{k}", np.nan) for m in METRICS},
           "comm_bytes_to_best_round": int(at_best.get("cum_download_bytes", 0) + at_best.get("cum_upload_bytes", 0)),
           "comm_bytes_total_run": int(sim.comm_totals["download_bytes"] + sim.comm_totals["upload_bytes"]),
           "wall_time_s": time.perf_counter() - t0, "peak_rss_mb": peak_rss_mb(),
           "A_fro_best": at_best.get("A_fro", np.nan), "B_fro_best": at_best.get("B_fro", np.nan),
           "item_norm_mean_best": at_best.get("item_norm_mean", np.nan)}
    if best["stopped_by"] == "diverged":
        row.update({f"val_{m}@{k}": np.nan for m in METRICS})
    pd.DataFrame([row]).to_csv(out / f"result_{tag(rank, lr, cfg['search']['seed'])}.csv", index=False,
                               float_format="%.17g")      # persisted immediately, before pool aggregation
    print(f"{stage} r={rank} lr={lr}: best {best['round']}/{best['rounds_run']} ({best['stopped_by']}) "
          f"val ndcg {row[f'val_ndcg@{k}']:.4f}  {row['wall_time_s']:.0f}s", flush=True)
    return row


def _select(df):
    """Ordinary converged argmax: only runs that ended by early stopping are eligible (diverged and max-cap excluded)."""
    ok = df[df["stopped_by"] == "early_stopping"].copy()
    if not len(ok):
        return None
    ok["tie"] = (np.log(ok["local_lr"] / 5.0)).abs()
    return ok.sort_values(["val_ndcg@10", "tie"], ascending=[False, True], kind="mergesort").iloc[0]


def _cap_hits_above(df, best):
    cap = df[df["stopped_by"] == "max_rounds"]
    if best is None:
        return cap
    return cap[cap["val_ndcg@10"] > best["val_ndcg@10"]]


def search(workers):
    cfg = load_config(ROOT / CONFIG)
    sc, ranks = cfg["search"], cfg["model"]["ranks"]
    jobs = [(r, lr, "grid") for r in ranks for lr in sc["lr_grid"]]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_search_one, jobs))
    df = pd.DataFrame(rows)
    lo, hi = min(sc["lr_grid"]), max(sc["lr_grid"])
    extra = []
    for r in ranks:                                         # single edge rule (applied once per rank)
        best = _select(df[df["rank"] == r])
        if best is not None and best["local_lr"] in (lo, hi):
            extra.append((r, hi * sc["edge_factor"] if best["local_lr"] == hi else lo / sc["edge_factor"], "edge"))
    if extra:
        with ProcessPoolExecutor(max_workers=min(workers, len(extra))) as ex:
            df = pd.concat([df, pd.DataFrame(list(ex.map(_search_one, extra)))], ignore_index=True)
    sel = []
    for r in ranks:
        best = _select(df[df["rank"] == r])
        sel.append({"rank": r, "selected_lr": None if best is None else float(best["local_lr"]),
                    "val_ndcg@10": None if best is None else float(best["val_ndcg@10"]),
                    "best_round": None if best is None else int(best["best_round"]),
                    "stopped_by": None if best is None else best["stopped_by"],
                    "edge_expanded": any(e[0] == r for e in extra),
                    "n_diverged": int((df[df["rank"] == r]["stopped_by"] == "diverged").sum()),
                    "flag": "; ".join(f"cap-hit lr={c.local_lr} val={c['val_ndcg@10']:.4f} exceeds the converged argmax"
                                      " (REPORT before any further search)"
                                      for _, c in _cap_hits_above(df[df["rank"] == r], best).iterrows())})
    df.sort_values(["rank", "local_lr"]).to_csv(RESULTS / "b3_lr_search.csv", index=False, float_format="%.6f")
    pd.DataFrame(sel).to_csv(RESULTS / "b3_selection.csv", index=False)
    print(df.sort_values(["rank", "local_lr"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(sel).to_string(index=False))


def selected_lr(cfg, rank):
    sel = cfg["selected_lr"]
    v = sel.get(rank, sel.get(str(rank)))
    if v is None:
        raise RuntimeError(f"no frozen learning rate for rank {rank}")
    return float(v)


SCIENTIFIC_FIELDS = (("dataset_config",), ("model", "dim"), ("model", "init_std"), ("model", "rank"),
                     ("federated", "client_sampling_q"), ("federated", "local_optimizer"), ("federated", "local_lr"),
                     ("federated", "local_epochs"), ("federated", "l2_reg"), ("federated", "server_lr"),
                     ("federated", "aggregation"), ("federated", "max_rounds"), ("federated", "eval_every"),
                     ("federated", "patience_evals"), ("evaluation",))


def scientific_mismatches(stored, expected):
    """Per-job scientific fields only (other ranks' selected_lr etc. are irrelevant to this job)."""
    def get(c, path):
        for k in path:
            c = c.get(k) if isinstance(c, dict) else None
        return c
    return [".".join(p) for p in SCIENTIFIC_FIELDS if get(stored, p) != get(expected, p)]


def _sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _final_train_one(args):
    """FROZEN B3 job, validation only: train to convergence, retain the best-validation checkpoint (no test)."""
    rank, seed = args
    cfg, data_cfg, split, fp = load_all()
    require_frozen(cfg)
    c, t = final_job(cfg, rank, seed)
    k = c["evaluation"]["k"]
    out = RAW / "b3_runs"
    out.mkdir(parents=True, exist_ok=True)
    ck_path = CHECKPOINTS / "b3" / f"b3_{t}.pt"
    rec_path = out / f"train_result_{t}.csv"
    if rec_path.exists() or ck_path.exists():
        # resume only from fully verified artifacts; anything else is reported, never silently retrained over
        if not rec_path.exists():
            raise RuntimeError(f"{t}: checkpoint exists without a training record; refusing to overwrite")
        prev = pd.read_csv(rec_path).iloc[0].to_dict()
        if prev["stopped_by"] == "diverged" and not ck_path.exists():
            return pd.DataFrame([prev])
        if not ck_path.exists() or prev["checkpoint_sha256"] != _sha(ck_path):
            raise RuntimeError(f"{t}: checkpoint missing or SHA-256 differs from its record; refusing to overwrite")
        _, ck = load_b3_checkpoint(ck_path, split, rank, c["federated"]["local_lr"], seed)
        bad = scientific_mismatches(ck["config"], c)
        if bad:
            raise RuntimeError(f"{t}: stored config differs from the frozen job config ({bad}); refusing to overwrite")
        return pd.DataFrame([prev])                        # resumed: verified, not retrained
    with np.errstate(all="ignore"):
        sim, history, rounds, best = train_lowrank_converge(split, c, seed, log=lambda *_: None)
    pd.DataFrame(history).to_csv(out / f"history_{t}.csv", index=False, float_format="%.6f")
    pd.DataFrame(rounds).to_csv(out / f"rounds_{t}.csv", index=False, float_format="%.6f")
    at_best = next((h for h in history if h["round"] == best["round"]), history[0])
    meta = {"rank": rank, "local_lr": c["federated"]["local_lr"], "seed": seed, "tag": t,
            "best_round": best["round"], "rounds_run": best["rounds_run"], "stopped_by": best["stopped_by"],
            "max_rounds_ceiling": c["federated"]["max_rounds"],
            "comm_bytes_to_best_round": int(at_best["cum_download_bytes"] + at_best["cum_upload_bytes"]),
            "comm_bytes_total_run": int(sim.comm_totals["download_bytes"] + sim.comm_totals["upload_bytes"]),
            "shared_coordinates_D": sim.D, "bytes_per_client_per_round": int(2 * 4 * sim.D), "split_fingerprint": fp}
    if best["stopped_by"] != "diverged":
        val, val_ranks = validation_metrics(sim, split, k)
        meta.update({f"val_{m}": val[m] for m in [f"{x}@{k}" for x in METRICS]})
        from experiments.run_b1 import per_user_table
        per_user_table(split, "validation", val_ranks, k).to_csv(out / f"per_user_validation_{t}.csv", index=False,
                                                                 float_format="%.6f")
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"P": torch.from_numpy(sim.P.copy()), "A": torch.from_numpy(sim.A.copy()),
                    "B": torch.from_numpy(sim.B.copy()), "rank": rank, "config": c, "seed": seed,
                    "split_fingerprint": fp, "validation": val, "torch_version": str(torch.__version__),
                    **{x: meta[x] for x in ("best_round", "rounds_run", "stopped_by", "comm_bytes_to_best_round",
                                            "comm_bytes_total_run")}}, ck_path)
        meta["checkpoint_sha256"] = _sha(ck_path)
    else:
        meta["checkpoint_sha256"] = ""
    df = pd.DataFrame([meta])
    df.to_csv(out / f"train_result_{t}.csv", index=False, float_format="%.17g")   # persisted per run
    print(f"final-train r={rank} seed={seed}: {best['stopped_by']} best {best['round']}/{best['rounds_run']}", flush=True)
    return df


def final_train(workers):
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)
    jobs = [(r, s) for r in cfg["model"]["ranks"] for s in cfg["seeds"]]
    if workers <= 1:
        dfs = [_final_train_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            dfs = list(ex.map(_final_train_one, jobs))
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(RESULTS / "b3_final_validation.csv", index=False, float_format="%.6f")
    print(df[["rank", "seed", "best_round", "rounds_run", "stopped_by", "val_ndcg@10"]].to_string(index=False))


def check_b3_inventory(cfg, df):
    """Every rank x declared seed exactly once and ended by early stopping (cap-hit / diverged are reported, block)."""
    expected = {(int(r), int(s)) for r in cfg["model"]["ranks"] for s in cfg["seeds"]}
    got = list(zip(df["rank"].astype(int), df["seed"].astype(int)))
    problems = []
    if len(got) != len(set(got)):
        problems.append("duplicates")
    if set(got) != expected:
        problems.append(f"missing {sorted(expected - set(got))}; unexpected {sorted(set(got) - expected)}")
    bad = df[df["stopped_by"] != "early_stopping"]
    if len(bad):
        problems.append(f"not converged (report before scoring): {bad[['rank', 'seed', 'stopped_by']].values.tolist()}")
    if problems:
        raise RuntimeError("B3 final inventory not scoreable: " + " | ".join(problems))


def _final_score_one(row):
    cfg, _, split, fp = load_all()
    require_frozen(cfg)
    rank, seed, t = int(row["rank"]), int(row["seed"]), row["tag"]
    lr = selected_lr(cfg, rank)
    c_exp, t_exp = final_job(cfg, rank, seed)
    if t != t_exp:
        raise ValueError(f"recorded tag {t} != expected {t_exp}")
    out = RAW / "b3_runs"
    ck_path = CHECKPOINTS / "b3" / f"b3_{t}.pt"
    sha = _sha(ck_path)
    if sha != row["checkpoint_sha256"]:
        raise RuntimeError(f"{ck_path.name}: checkpoint SHA-256 differs from its training record; refusing")
    sim, ck = load_b3_checkpoint(ck_path, split, rank, lr, seed)          # BEFORE any cache branch
    bad = scientific_mismatches(ck["config"], c_exp)
    if bad:
        raise ValueError(f"{ck_path.name}: stored config differs from the frozen job config ({bad})")
    res_path, pu_path = out / f"test_result_{t}.csv", out / f"per_user_test_{t}.csv"
    if res_path.exists():
        prev = pd.read_csv(res_path).iloc[0].to_dict()
        if prev["checkpoint_sha256"] != sha or not pu_path.exists() or prev["per_user_test_sha256"] != _sha(pu_path):
            raise RuntimeError(f"{res_path.name}: recorded result does not verify (checkpoint/per-user hash); refusing")
        return prev                                        # verified reuse: zero new test evaluations
    if pu_path.exists():
        raise RuntimeError(f"{pu_path.name} exists without a verified result; refusing to overwrite audit history")
    k = cfg["evaluation"]["k"]
    test, ranks = evaluate(sim.full_scores, split, "test", k=k)                # the only test evaluation
    from experiments.run_b1 import per_user_table
    per_user_table(split, "test", ranks, k).to_csv(pu_path, index=False, float_format="%.17g")
    result = {"tag": t, "checkpoint_sha256": sha, "per_user_test_sha256": _sha(pu_path),
              **{f"test_{m}": test[m] for m in [f"{x}@{k}" for x in METRICS]}}
    pd.DataFrame([result]).to_csv(res_path, index=False, float_format="%.17g")
    return result


def final_score(workers):
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)
    df = pd.read_csv(RESULTS / "b3_final_validation.csv")
    check_b3_inventory(cfg, df)
    recs = df.to_dict("records")
    if workers <= 1:
        rows = [_final_score_one(r) for r in recs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(recs))) as ex:
            rows = list(ex.map(_final_score_one, recs))
    res = df.merge(pd.DataFrame(rows), on=["tag", "checkpoint_sha256"], how="left")
    res.to_csv(RESULTS / "b3_final_results.csv", index=False, float_format="%.6f")
    print(res.groupby("rank")[["test_ndcg@10", "test_hr@10", "test_mrr@10", "best_round"]].agg(["mean", "std"]).round(4))


def load_b3_checkpoint(path, split, rank, local_lr, seed):
    """Validated load: refuses any checkpoint whose shapes / rank / lr / seed / split do not match."""
    ckpt = torch.load(path, weights_only=True)
    fp = split_fingerprint(split)
    d = ckpt["config"]["model"]["dim"]
    problems = [f"{k} {ckpt.get(k)!r} != {v!r}" for k, v in (("rank", rank), ("seed", seed), ("split_fingerprint", fp))
                if ckpt.get(k) != v]
    if ckpt["config"]["federated"]["local_lr"] != local_lr:
        problems.append("local_lr mismatch")
    for key, shp in (("A", (split["n_items"], rank)), ("B", (rank, d)), ("P", (split["n_users"], d))):
        if tuple(ckpt[key].shape) != shp:
            problems.append(f"{key} shape {tuple(ckpt[key].shape)} != {shp}")
    if problems:
        raise ValueError("B3 checkpoint does not match: " + "; ".join(problems))
    sim = LowRankFederatedBPR(split["train"], split["n_users"], split["n_items"], ckpt["config"], seed)
    sim.load_state({"P": ckpt["P"].numpy(), "A": ckpt["A"].numpy(), "B": ckpt["B"].numpy()})
    return sim, ckpt


def _b1_supplemental_one(seed):
    import experiments.run_b1 as rb1
    rb1.RAW = RAW / "supplemental_b1"                      # separate tagged paths; frozen B1 outputs untouched
    rb1.CHECKPOINTS = CHECKPOINTS / "supplemental_b1"
    rb1.RAW.mkdir(parents=True, exist_ok=True)
    rb1.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    return rb1.run("configs/b1.yaml", seed, primary=False)


def b1_supplemental():
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)                                    # exact frozen-B1 replication, scored only after B3 freeze
    with ProcessPoolExecutor(max_workers=2) as ex:
        list(ex.map(_b1_supplemental_one, [7, 99]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--search", action="store_true")
    p.add_argument("--final-train", action="store_true")
    p.add_argument("--final-score", action="store_true")
    p.add_argument("--b1-supplemental", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.profile:
        profile(a.workers)
    elif a.search:
        search(a.workers)
    elif a.final_train:
        final_train(a.workers)
    elif a.final_score:
        final_score(a.workers)
    elif a.b1_supplemental:
        b1_supplemental()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
