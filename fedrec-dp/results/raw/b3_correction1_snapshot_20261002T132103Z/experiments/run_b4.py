"""B4 — low-rank user-level DP federated BPR. Runner (protocol: RESEARCH_LOG "B3/B4 protocol", 2026-10-02, as AMENDED
by "B4 PROTOCOL AMENDMENT" the same day; the original B3-lr-inheritance / norm-quantile C grid is superseded).

Validation-only phases (no test calls):
    --norm-stats          DIAGNOSTIC ONLY: B3-lr update norms, rounds 1-50 (defines no grid)
    --lr-c-search         needs frozen B3: eps≈4, eta_s 1, seed 42, T 50; lr in B3's grid (+0.625 only where B3
                          selected it) x C in B2's {1.0, 1.5, 2.4}; argmax final-round val NDCG@10
    --slr-search          eta_s in {0.5, 1, 2} at the selected (lr, C); the eta_s = 1 cell is reused
    --final-train         FROZEN per-rank jobs (control + eps 8/4/2/1 [+ fixed-settings diagnostic]) x seeds,
                          validation only, exact T=50 checkpoints retained
    --select-rank         five-seed validation means -> validation-selected rank per eps (written BEFORE any test)
Test phase (refused unless protocol_frozen AND the rank-selection file exists):
    --score-test          score the SAME retained checkpoints on test once
sigma is never solved here: it is read from B2's T=50 accounting rows (full key), so B4 uses B2's exact noise multipliers.
"""

import os

import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.threads import enforce_and_report, force_env  # noqa: E402

force_env()                              # override inherited BLAS/OpenMP thread variables

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

import experiments.run_b2 as rb2  # noqa: E402
from experiments.run_b1 import per_user_table  # noqa: E402
from src.data import load_processed, split_fingerprint  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.lowrank import LowRankFederatedBPR, shared_coordinates  # noqa: E402
from src.privacy import privacy_record  # noqa: E402
from src.train_federated import METRICS, validation_metrics  # noqa: E402
from src.train_lowrank import train_lowrank_fixed  # noqa: E402
from src.utils import load_config  # noqa: E402

RESULTS, RAW, CHECKPOINTS = ROOT / "results", ROOT / "results" / "raw", ROOT / "checkpoints"
CONFIG = "configs/b4.yaml"
B3_CONFIG = "configs/b3.yaml"
B2_CONFIG = "configs/b2.yaml"


def load_all():
    cfg = load_config(ROOT / CONFIG)
    data_cfg = load_config(ROOT / cfg["dataset_config"])
    split = load_processed(data_cfg)
    return cfg, data_cfg, split, split_fingerprint(split)


def require_frozen(cfg):
    if not cfg.get("protocol_frozen", False):
        raise RuntimeError("B4 protocol is not frozen (protocol_frozen: false); test scoring refused")


def require_rank_selection():
    if not (RESULTS / "b4_rank_selection.csv").exists():
        raise RuntimeError("validation-selected ranks are not recorded (results/b4_rank_selection.csv); test refused")


B2_MATCH_KEYS = ("T", "delta", "accountant", "cross_check_accountant", "solver_tolerance", "denominator",
                 "target_epsilons")


def check_b2_compatible(cfg):
    """B4 must use final B2's exact accounting settings before B2's sigma may be reused."""
    b2 = load_config(ROOT / B2_CONFIG)
    bad = [f"privacy.{k}: B4 {cfg['privacy'][k]!r} != B2 {b2['privacy'][k]!r}" for k in B2_MATCH_KEYS
           if cfg["privacy"][k] != b2["privacy"][k]]
    if cfg["federated"]["client_sampling_q"] != b2["federated"]["client_sampling_q"]:
        bad.append("federated.client_sampling_q differs from B2")
    if not b2["privacy"].get("protocol_frozen", False):
        bad.append("B2 protocol is not frozen")
    if bad:
        raise RuntimeError("B4 config is not compatible with frozen B2: " + "; ".join(bad))
    return b2


def b2_sigma(cfg, epsilon):
    """B2's noise multiplier for (eps, T) under B2's exact accounting key; never solved anew here."""
    b2cfg = check_b2_compatible(cfg)
    T = cfg["privacy"]["T"]
    row = rb2.cached_accounting_row(b2cfg, epsilon, T, pd.read_csv(RESULTS / "b2_accounting.csv"))
    if row is None:
        raise RuntimeError(f"no B2 accounting row for eps={epsilon}, T={T}; B4 must reuse B2's sigma")
    return float(row["noise_multiplier"])


def job(rank, epsilon, seed, clip=None, server_lr=None, local_lr=None, kind="main"):
    return {"rank": int(rank), "epsilon": epsilon, "seed": int(seed), "clip": clip, "server_lr": server_lr,
            "local_lr": local_lr, "kind": kind}


def run_cfg(cfg, j):
    """Run config for one job. epsilon None -> matched no-DP control (no clip, no noise, qN)."""
    c = copy.deepcopy(cfg)
    r = j["rank"]
    per = cfg["per_rank"].get(r, cfg["per_rank"].get(str(r), {}))
    c["model"]["rank"] = r
    c["federated"]["local_lr"] = float(j["local_lr"] if j["local_lr"] is not None else per["local_lr"])
    c["federated"]["server_lr"] = float(j["server_lr"] if j["server_lr"] is not None
                                        else per.get("server_lr", cfg["federated"]["server_lr"]))
    pc = c["privacy"]
    if j["epsilon"] is None:
        pc.update(clip_norm=None, noise_multiplier=0.0, target_epsilon=None)
    else:
        pc.update(clip_norm=float(j["clip"] if j["clip"] is not None else per["clip_norm"]),
                  noise_multiplier=b2_sigma(cfg, float(j["epsilon"])), target_epsilon=float(j["epsilon"]))
    return c


def run_tag(c, j):
    lvl = "nodp" if j["epsilon"] is None else f"eps{float(j['epsilon']):g}"
    return (f"{j['kind']}_r{j['rank']}_{lvl}_T{c['privacy']['T']}_C{c['privacy']['clip_norm']}"
            f"_lr{c['federated']['local_lr']}_slr{c['federated']['server_lr']}_seed{j['seed']}")


def _ckpt_path(tag):
    return CHECKPOINTS / "b4" / f"b4_{tag}.pt"


def train_validate(j, keep_checkpoint=False):
    """Train exactly T rounds; validation only. Optionally retain the exact round-T checkpoint."""
    cfg, data_cfg, split, fp = load_all()
    c = run_cfg(cfg, j)
    k, T = c["evaluation"]["k"], c["privacy"]["T"]
    if keep_checkpoint:
        tag0 = run_tag(c, j)
        res0 = RAW / "b4_runs" / f"result_{tag0}.csv"
        if _ckpt_path(tag0).exists() or res0.exists():
            if not (_ckpt_path(tag0).exists() and res0.exists()):
                raise RuntimeError(f"{tag0}: partial artifacts exist; refusing to overwrite (report and inspect)")
            prev = pd.read_csv(res0).iloc[0].to_dict()
            validate_checkpoint(torch.load(_ckpt_path(tag0), weights_only=True), cfg, j, split, fp)
            if prev.get("checkpoint_sha256") != file_sha256(_ckpt_path(tag0)):
                raise RuntimeError(f"{tag0}: checkpoint does not match its record; refusing to overwrite")
            return prev                                                  # verified, not retrained
    threads = enforce_and_report()
    rec = privacy_record(c["privacy"], c["federated"]["client_sampling_q"], T,
                         c["privacy"]["accountant"], c["privacy"]["cross_check_accountant"])
    status = "ok"
    try:
        with np.errstate(all="ignore"):
            sim, history, rounds, _ = train_lowrank_fixed(split, c, j["seed"], T, c["federated"]["eval_every"],
                                                          validate=True, log=lambda *_: None)
    except FloatingPointError as e:
        status, sim, history, rounds = str(e), None, [], []
    tag = run_tag(c, j)
    out = RAW / "b4_runs"
    out.mkdir(parents=True, exist_ok=True)
    row = {"kind": j["kind"], "rank": j["rank"], "privacy_level": "nodp" if j["epsilon"] is None else f"eps{j['epsilon']:g}",
           "seed": j["seed"], "status": status, **{f"privacy_{x}": rec[x] for x in
                                                  ("T", "clip_norm", "noise_multiplier", "epsilon", "epsilon_cross_check", "q", "delta")},
           "local_lr": c["federated"]["local_lr"], "server_lr": c["federated"]["server_lr"],
           "shared_coordinates_D": shared_coordinates(split["n_items"], c["model"]["dim"], j["rank"]),
           "split_fingerprint": fp, "tag": tag,
           "numpy_openblas_threads": threads["numpy_openblas_threads"], "torch_threads": threads["torch_threads"]}
    if sim is not None:
        val, val_ranks = validation_metrics(sim, split, k)
        r = pd.DataFrame(rounds)
        sc = rec["noise_multiplier"] * (rec["clip_norm"] or 0.0)
        row.update({f"val_{m}": val[m] for m in [f"{x}@{k}" for x in METRICS]})
        row.update({"frac_clipped_mean": r["frac_clipped"].mean(), "pre_clip_norm_median_mean": r["pre_clip_norm_median"].mean(),
                    "shrinkage_mean": r["shrinkage_mean"].mean(), "clipped_aggregate_norm_mean": r["clipped_aggregate_norm"].mean(),
                    "noise_norm_mean": r["noise_norm"].mean(), "typical_noise_norm": sc * np.sqrt(row["shared_coordinates_D"]),
                    "signal_to_noise_mean": r["signal_to_noise"].replace(np.inf, np.nan).mean(),
                    "A_fro_final": float(np.linalg.norm(sim.A)), "B_fro_final": float(np.linalg.norm(sim.B)),
                    "final_item_norm_mean": float(np.linalg.norm(sim.Q, axis=1).mean()),
                    "comm_bytes_total": int((r["download_bytes"].fillna(0) + r["upload_bytes"].fillna(0)).sum()),
                    "bytes_per_client_per_round": int(r["bytes_per_client"].max())})
        pd.DataFrame(history).to_csv(out / f"history_{tag}.csv", index=False, float_format="%.6f")
        r.to_csv(out / f"rounds_{tag}.csv", index=False, float_format="%.6f")
        per_user_table(split, "validation", val_ranks, k).to_csv(out / f"per_user_validation_{tag}.csv", index=False,
                                                                 float_format="%.6f")
        if keep_checkpoint:
            _ckpt_path(tag).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"P": torch.from_numpy(sim.P.copy()), "A": torch.from_numpy(sim.A.copy()),
                        "B": torch.from_numpy(sim.B.copy()), "rank": j["rank"], "config": c, "seed": j["seed"],
                        "noise_seed": sim.noise_seed, "round": sim.round, "split_fingerprint": fp,
                        **{x: rec[x] for x in ("q", "T", "clip_norm", "noise_multiplier", "delta", "epsilon",
                                               "accountant", "epsilon_cross_check", "cross_check_accountant",
                                               "denominator")},
                        "validation": val, "torch_version": str(torch.__version__)}, _ckpt_path(tag))
            row["checkpoint_sha256"] = file_sha256(_ckpt_path(tag))
    pd.DataFrame([row]).to_csv(out / f"result_{tag}.csv", index=False, float_format="%.17g")   # persisted per job
    print(f"{tag}: {status} val ndcg {row.get('val_ndcg@10', float('nan')):.4f}", flush=True)
    return row


def run_many(jobs, workers, keep_checkpoint=False):
    if workers <= 1:                                     # in-process (serial) mode
        return pd.DataFrame([train_validate(j, keep_checkpoint) for j in jobs])
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        return pd.DataFrame(list(ex.map(train_validate, jobs, [keep_checkpoint] * len(jobs))))


def job_from_row(row):
    """Rebuild the expected job of a recorded validation row (kind, rank, level, seed and the per-job overrides)."""
    eps = None if row["privacy_level"] == "nodp" else float(str(row["privacy_level"])[3:])
    fixed = row["kind"] == "fixeddiag"
    return job(int(row["rank"]), eps, int(row["seed"]),
               clip=float(row["privacy_clip_norm"]) if fixed else None,
               server_lr=float(row["server_lr"]) if fixed else None,
               local_lr=float(row["local_lr"]) if fixed else None, kind=row["kind"])


SCIENTIFIC_KEYS = {"model": ("dim", "init_std", "rank"),
                   "federated": ("client_sampling_q", "local_lr", "local_epochs", "l2_reg", "server_lr", "aggregation"),
                   "privacy": ("T", "delta", "denominator", "accountant", "cross_check_accountant", "solver_tolerance",
                               "clip_norm", "noise_multiplier", "target_epsilon")}


def validate_checkpoint(ckpt, cfg, j, split, fp):
    """Refuse any checkpoint that does not match the intended frozen job exactly (no silent mis-scoring)."""
    c = run_cfg(cfg, j)
    rec = privacy_record(c["privacy"], c["federated"]["client_sampling_q"], c["privacy"]["T"],
                         c["privacy"]["accountant"], c["privacy"]["cross_check_accountant"])
    r, d = j["rank"], c["model"]["dim"]
    problems = []
    expect = {"rank": r, "seed": j["seed"], "noise_seed": j["seed"], "round": c["privacy"]["T"],
              "split_fingerprint": fp, "q": rec["q"], "T": rec["T"], "clip_norm": rec["clip_norm"],
              "noise_multiplier": rec["noise_multiplier"], "delta": rec["delta"], "accountant": rec["accountant"],
              "cross_check_accountant": rec["cross_check_accountant"], "denominator": rec["denominator"],
              "epsilon": rec["epsilon"], "epsilon_cross_check": rec["epsilon_cross_check"]}
    for key, val in expect.items():
        if ckpt.get(key) != val:
            problems.append(f"{key}: checkpoint {ckpt.get(key)!r} != expected {val!r}")
    for key, shp in {"A": (split["n_items"], r), "B": (r, d), "P": (split["n_users"], d)}.items():
        t = ckpt[key]
        if tuple(t.shape) != shp:
            problems.append(f"{key} shape {tuple(t.shape)} != {shp}")
        if t.dtype != torch.float32:
            problems.append(f"{key} dtype {t.dtype} != float32")
        elif not torch.isfinite(t).all():
            problems.append(f"{key} has non-finite values")
    cf = ckpt["config"]
    for section, keys in SCIENTIFIC_KEYS.items():
        for key in keys:
            got, want = cf.get(section, {}).get(key), c[section].get(key)
            if got != want:
                problems.append(f"config.{section}.{key} {got!r} != expected {want!r}")
    if problems:
        raise ValueError("checkpoint does not match the frozen job: " + "; ".join(problems))


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_test_from_checkpoint(row):
    """Score one retained checkpoint on test ONCE (frozen + rank-selected phase), after validating it.

    Resumable: a completed job leaves test_result_<tag>.csv (full precision + checkpoint SHA-256); a rerun reuses a
    verified result with ZERO new test evaluations. An existing per-user test file without a verified result is
    never overwritten (refused).
    """
    cfg, _, split, fp = load_all()
    require_frozen(cfg)
    require_rank_selection()
    j = job_from_row(row)
    tag = run_tag(run_cfg(cfg, j), j)
    if tag != row["tag"]:
        raise ValueError(f"recorded tag {row['tag']} != expected {tag}")
    path = _ckpt_path(tag)
    sha = file_sha256(path)
    if "checkpoint_sha256" in row and row["checkpoint_sha256"] != sha:
        raise RuntimeError(f"{path.name}: checkpoint SHA-256 differs from its training record; refusing")
    ckpt = torch.load(path, weights_only=True)
    validate_checkpoint(ckpt, cfg, j, split, fp)                         # BEFORE any cache branch
    out = RAW / "b4_runs"
    res_path, pu_path = out / f"test_result_{tag}.csv", out / f"per_user_test_{tag}.csv"
    if res_path.exists():
        prev = pd.read_csv(res_path).iloc[0].to_dict()
        if (prev["checkpoint_sha256"] != sha or not pu_path.exists()
                or prev["per_user_test_sha256"] != file_sha256(pu_path)):
            raise RuntimeError(f"{res_path.name}: recorded result does not verify (checkpoint/per-user hash); refusing")
        return prev                                                     # verified reuse, no new test evaluation
    if pu_path.exists():
        raise RuntimeError(f"{pu_path.name} exists without a verified result; refusing to overwrite audit history")
    sim = LowRankFederatedBPR(split["train"], split["n_users"], split["n_items"], ckpt["config"], ckpt["seed"])
    sim.load_state({"P": ckpt["P"].numpy(), "A": ckpt["A"].numpy(), "B": ckpt["B"].numpy()})
    test, ranks = evaluate(sim.full_scores, split, "test", k=cfg["evaluation"]["k"])
    per_user_table(split, "test", ranks, cfg["evaluation"]["k"]).to_csv(pu_path, index=False, float_format="%.17g")
    result = {"tag": tag, "checkpoint_sha256": sha, "per_user_test_sha256": file_sha256(pu_path),
              **{f"test_{m}": test[m] for m in [f"{x}@10" for x in METRICS]}}
    pd.DataFrame([result]).to_csv(res_path, index=False, float_format="%.17g")
    return result


def _per_rank(cfg, r):
    return cfg["per_rank"].get(r, cfg["per_rank"].get(str(r), {}))


def norm_stats(workers):
    """DIAGNOSTIC ONLY (amended protocol): pooled ||[dA, dB]|| over rounds 1-50 at each rank's B3 lr (no DP).
    Does NOT define any C grid; the amended C grid is B2's existing {1.0, 1.5, 2.4}."""
    cfg = load_config(ROOT / CONFIG)
    b3 = load_config(ROOT / B3_CONFIG)
    rows = []
    for r in cfg["model"]["ranks"]:
        c = copy.deepcopy(cfg)
        c["model"]["rank"] = r
        c["federated"]["local_lr"] = b3_selected_lr(b3, r)
        c["privacy"].update(clip_norm=None, noise_multiplier=0.0, denominator="realised")
        _, _, split, _ = load_all()
        _, _, _, norms = train_lowrank_fixed(split, c, cfg["seed"], c["privacy"]["T"], eval_every=c["privacy"]["T"],
                                             validate=False, keep_client_norms=True, log=lambda *_: None)
        x = pd.Series([n for _, n in norms])
        q = {f"p{int(v * 100)}": float(x.quantile(v)) for v in (0.25, 0.5, 0.75, 0.9, 0.95)}
        rows.append({"rank": r, "b3_local_lr": c["federated"]["local_lr"], "n_updates": len(x), "mean": x.mean(),
                     **q, "max": x.max()})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "b4_norm_stats_diagnostic.csv", index=False, float_format="%.6f")
    print(df.to_string(index=False))


def b3_selected_lr(b3cfg, rank):
    sel = b3cfg["selected_lr"]
    v = sel.get(rank, sel.get(str(rank)))
    if v is None:
        raise RuntimeError(f"B3 has no frozen learning rate for rank {rank}")
    return float(v)


def require_b3_frozen():
    b3 = load_config(ROOT / B3_CONFIG)
    if not b3.get("protocol_frozen", False):
        raise RuntimeError("B3 is not frozen: the amended B4 lr x C check needs B3's frozen selection (0.625 rule)")
    return b3


def lr_c_grid(cfg, b3cfg, rank):
    """Amended grid: existing B3 lr grid (+ the tested edge value only if B3's frozen selection uses it) x B2's C grid."""
    s = cfg["lr_c_search"]
    lrs = [float(x) for x in s["local_lr_grid"]]
    edge = float(s["edge_lr_if_b3_selected"])
    if b3_selected_lr(b3cfg, rank) == edge:
        lrs = [edge] + lrs
    return [(lr, float(C)) for lr in lrs for C in s["clip_grid"]]


def select_lr_c(g):
    """argmax final-round val NDCG@10 over status-ok cells; ties: smaller C, then lr nearer 5, then smaller lr."""
    ok = g[g["status"] == "ok"].copy()
    if not len(ok):
        return None
    ok["_dist5"] = np.abs(np.log(ok["local_lr"].astype(float) / 5.0)).round(12)
    ok = ok.sort_values(["val_ndcg@10", "privacy_clip_norm", "_dist5", "local_lr"],
                        ascending=[False, True, True, True], kind="mergesort")
    return ok.iloc[0]


def select_slr(g):
    """argmax final-round val NDCG@10 over status-ok cells; ties: eta_s nearer 1, then smaller."""
    ok = g[g["status"] == "ok"].copy()
    if not len(ok):
        return None
    ok["_dist1"] = np.abs(np.log(ok["server_lr"].astype(float))).round(12)
    return ok.sort_values(["val_ndcg@10", "_dist1", "server_lr"], ascending=[False, True, True],
                          kind="mergesort").iloc[0]


def lr_c_search(workers):
    """Amended B4 step 1: eps≈4, eta_s 1.0, seed 42, T 50, validation only."""
    cfg = load_config(ROOT / CONFIG)
    require_b3_frozen()
    b3 = load_config(ROOT / cfg["lr_c_search"]["b3_reference_config"])   # INITIAL B3 freeze; later corrections ignored
    s = cfg["lr_c_search"]
    jobs = [job(r, s["epsilon"], s["seed"], clip=C, server_lr=s["server_lr"], local_lr=lr, kind="lrcsearch")
            for r in cfg["model"]["ranks"] for lr, C in lr_c_grid(cfg, b3, r)]
    df = run_many(jobs, workers)
    df.to_csv(RESULTS / "b4_lr_c_search.csv", index=False, float_format="%.17g")
    sel = []
    for r, g in df.groupby("rank"):
        b = select_lr_c(g)
        sel.append({"rank": r, "n_cells": len(g), "n_failed": int((g.status != "ok").sum()),
                    "local_lr": None if b is None else float(b["local_lr"]),
                    "clip_norm": None if b is None else float(b["privacy_clip_norm"]),
                    "val_ndcg@10": None if b is None else float(b["val_ndcg@10"]), "tag": None if b is None else b["tag"]})
    sel = pd.DataFrame(sel)
    sel.to_csv(RESULTS / "b4_lr_c_selection.csv", index=False, float_format="%.17g")
    print(df[["rank", "local_lr", "privacy_clip_norm", "status", "val_ndcg@10", "frac_clipped_mean",
              "signal_to_noise_mean"]].to_string(index=False))
    print(sel.to_string(index=False))


def check_per_rank_matches_lr_c_selection(cfg):
    sel = pd.read_csv(RESULTS / "b4_lr_c_selection.csv")
    for _, r in sel.iterrows():
        pr = _per_rank(cfg, int(r["rank"]))
        if pd.isna(r["local_lr"]):
            raise RuntimeError(f"rank {r['rank']}: lr x C check had no successful cell; report before continuing")
        if float(pr.get("local_lr", np.nan)) != float(r["local_lr"]) or float(pr.get("clip_norm", np.nan)) != float(r["clip_norm"]):
            raise RuntimeError(f"rank {r['rank']}: configs/b4.yaml per_rank lr/C != b4_lr_c_selection.csv")


def slr_search(workers):
    """Amended B4 step 2: eta_s in {0.5, 1, 2} at the selected (lr, C); the eta_s = 1 cell is the identical lr x C
    search cell and is reused, not retrained."""
    cfg = load_config(ROOT / CONFIG)
    require_b3_frozen()
    check_per_rank_matches_lr_c_selection(cfg)
    s = cfg["lr_c_search"]
    lrc = pd.read_csv(RESULTS / "b4_lr_c_search.csv")
    jobs, reused = [], []
    for r in cfg["model"]["ranks"]:
        pr = _per_rank(cfg, r)
        for eta in cfg["privacy"]["server_lr_grid"]:
            if float(eta) == float(s["server_lr"]):
                m = lrc[(lrc["rank"] == r) & (lrc["local_lr"] == float(pr["local_lr"]))
                        & (lrc["privacy_clip_norm"] == float(pr["clip_norm"]))]
                if len(m) != 1:
                    raise RuntimeError(f"rank {r}: cannot find the unique identical eta_s=1 search cell to reuse")
                reused.append(m.iloc[0].to_dict())
            else:
                jobs.append(job(r, s["epsilon"], s["seed"], clip=float(pr["clip_norm"]), server_lr=float(eta),
                                local_lr=float(pr["local_lr"]), kind="slrsearch"))
    df = pd.concat([pd.DataFrame(reused).assign(reused_from="lrcsearch"), run_many(jobs, workers)], ignore_index=True)
    df.to_csv(RESULTS / "b4_slr_search.csv", index=False, float_format="%.17g")
    sel = []
    for r, g in df.groupby("rank"):
        b = select_slr(g)
        sel.append({"rank": r, "server_lr": None if b is None else float(b["server_lr"]),
                    "val_ndcg@10": None if b is None else float(b["val_ndcg@10"]), "n_failed": int((g.status != "ok").sum())})
    pd.DataFrame(sel).to_csv(RESULTS / "b4_slr_selection.csv", index=False, float_format="%.17g")
    print(df[["rank", "server_lr", "status", "val_ndcg@10"]].to_string(index=False))
    print(pd.DataFrame(sel).to_string(index=False))


def final_jobs(cfg):
    jobs = [job(r, e, s, kind="main") for s in cfg["seeds"] for r in cfg["model"]["ranks"]
            for e in [None] + list(cfg["privacy"]["target_epsilons"])]
    fd = cfg["fixed_settings_diagnostic"]
    for s in cfg["seeds"]:
        for r in cfg["model"]["ranks"]:
            pr = _per_rank(cfg, r)
            same = (float(pr["local_lr"]) == fd["local_lr"] and float(pr["clip_norm"]) == fd["clip_norm"]
                    and float(pr["server_lr"]) == fd["server_lr"])
            if not same:                                   # identical cells are reused, not retrained
                jobs.append(job(r, fd["epsilon"], s, clip=fd["clip_norm"], server_lr=fd["server_lr"],
                                local_lr=fd["local_lr"], kind="fixeddiag"))
    return jobs


def check_main_inventory(cfg, df):
    """Exact main inventory: every rank x {nodp, eps...} x declared seed exactly once, all status ok."""
    levels = ["nodp"] + [f"eps{float(e):g}" for e in cfg["privacy"]["target_epsilons"]]
    main = df[df.kind == "main"]
    expected = {(int(r), l, int(s)) for r in cfg["model"]["ranks"] for l in levels for s in cfg["seeds"]}
    got = list(zip(main["rank"].astype(int), main["privacy_level"], main["seed"].astype(int)))
    problems = []
    dup = {x for x in got if got.count(x) > 1}
    if dup:
        problems.append(f"duplicates: {sorted(dup)}")
    if set(got) != expected:
        problems.append(f"missing: {sorted(expected - set(got))}; unexpected: {sorted(set(got) - expected)}")
    bad = main[main.status != "ok"]
    if len(bad):
        problems.append(f"failed main jobs: {bad[['rank', 'privacy_level', 'seed', 'status']].values.tolist()}")
    if problems:
        raise RuntimeError("incomplete or invalid main inventory: " + " | ".join(problems))


def final_train(workers):
    """FROZEN jobs, validation only, exact T=50 checkpoints retained (no test)."""
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)
    df = run_many(final_jobs(cfg), workers, keep_checkpoint=True)
    df.to_csv(RESULTS / "b4_final_validation.csv", index=False, float_format="%.6f")
    print(df.groupby(["kind", "rank", "privacy_level"])["val_ndcg@10"].agg(["mean", "std", "count"]).round(4))


def select_rank():
    """Validation-selected rank per eps from the five-seed validation means (main jobs only). Written BEFORE test."""
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)
    df = pd.read_csv(RESULTS / "b4_final_validation.csv")
    check_main_inventory(cfg, df)                          # never select from fewer seeds or partial cells
    m = df[(df.kind == "main") & (df.status == "ok")].groupby(["privacy_level", "rank"])["val_ndcg@10"].mean().reset_index()
    sel = m.sort_values(["privacy_level", "val_ndcg@10", "rank"], ascending=[True, False, True]).groupby("privacy_level").head(1)
    sel.to_csv(RESULTS / "b4_rank_selection.csv", index=False, float_format="%.6f")
    print(m.pivot(index="rank", columns="privacy_level", values="val_ndcg@10").round(4))
    print(sel.to_string(index=False))


def score_test(workers):
    cfg = load_config(ROOT / CONFIG)
    require_frozen(cfg)
    require_rank_selection()
    df = pd.read_csv(RESULTS / "b4_final_validation.csv")
    check_main_inventory(cfg, df)
    failed_diag = df[(df.kind == "fixeddiag") & (df.status != "ok")]
    if len(failed_diag):
        print("fixed-settings diagnostic failures (reported separately, not scored):",
              failed_diag[["rank", "seed", "status"]].values.tolist())
    recs = df[df.status == "ok"].to_dict("records")
    if workers <= 1:
        rows = [score_test_from_checkpoint(r) for r in recs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(recs))) as ex:
            rows = list(ex.map(score_test_from_checkpoint, recs))
    out = df.merge(pd.DataFrame(rows), on="tag", how="left")
    out.to_csv(RESULTS / "b4_final_results.csv", index=False, float_format="%.6f")
    print(out.groupby(["kind", "rank", "privacy_level"])["test_ndcg@10"].agg(["mean", "std", "count"]).round(4))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("norm-stats", "lr-c-search", "slr-search", "final-train", "select-rank", "score-test"):
        p.add_argument(f"--{flag}", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args()
    if a.norm_stats:
        norm_stats(a.workers)
    elif a.lr_c_search:
        lr_c_search(a.workers)
    elif a.slr_search:
        slr_search(a.workers)
    elif a.final_train:
        final_train(a.workers)
    elif a.select_rank:
        select_rank()
    elif a.score_test:
        score_test(a.workers)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
