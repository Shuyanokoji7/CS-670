"""E2 pulse/reset diagnostic. Reads E1 checkpoints; never evaluates held-out labels."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.threads import force_env
force_env()

import argparse
import copy
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.effective_noise import EffectiveNoiseBPR
from src.federated import sample_clients
from src.lowrank import balance
from src.threads import enforce_and_report

E1 = ROOT / "results/extensions/effective_noise_v1"
OUT = ROOT / "results/extensions/noise_memory_v1"
PROTOCOL = ROOT / "docs/extensions/NOISE_MEMORY_PROTOCOL.md"
UNITS = ["full", "fixed_r8", "two_r8"]
LEVELS = ["eps1", "eps2"]
SEEDS = [42, 123, 2026, 7, 99]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def shared_copy(target, reference):
    if target.method == "full":
        target._Q = reference._Q.copy()
    else:
        target.A, target.B = reference.A.copy(), reference.B.copy()


def assert_equal(a, b):
    for key, value in a.state().items():
        if not np.array_equal(value, b.state()[key]):
            raise AssertionError(f"both-reset/zero-pulse discrepancy in {key}")
    assert_rng_equal(a, b)


def assert_rng_equal(a, b):
    for key in ("sampling_rng", "client_rng", "noise_rng"):
        if getattr(a, key).bit_generator.state != getattr(b, key).bit_generator.state:
            raise AssertionError(f"uncoupled random stream: {key}")


def add_pulse(model, alpha, replicate):
    c = model.settings
    tau = c["server_lr"] * c["sigma"] * c["clip_norm"] / (c["q"] * model.n_users)
    rng = np.random.default_rng([c["seed"], 6702, replicate])
    z = rng.standard_normal(model.D, dtype=np.float32) * np.float32(alpha * tau)
    if alpha == 0:
        return tau
    if model.method == "full":
        model._Q += z.reshape(model._Q.shape)
    else:
        k = model.A.size
        model.A += z[:k].reshape(model.A.shape)
        if model.method == "two":
            model.B += z[k:].reshape(model.B.shape)
            model.A, model.B = balance(model.A, model.B)
    return tau


def finite(model):
    if not all(np.isfinite(v).all() for v in model.state().values()):
        raise FloatingPointError("nonfinite diagnostic state")


def observe(branch, reference, exposed, phase, lag, pulse_rms=None):
    finite(branch)
    finite(reference)
    p, q = reference.P.astype(np.float64), reference.Q.astype(np.float64)
    dp, dq = branch.P.astype(np.float64) - p, branch.Q.astype(np.float64) - q
    direct, local, cross = p @ dq.T, dp @ q.T, dp @ dq.T
    clean = p @ q.T
    shifted = branch.P.astype(np.float64) @ branch.Q.astype(np.float64).T
    diff = shifted - clean
    e = lambda x: float(np.sum(x * x))
    rms = float(np.sqrt(np.mean(diff * diff)))
    row = {
        "phase": phase, "lag": lag, "q_difference_fro": float(np.linalg.norm(dq)),
        "p_difference_fro": float(np.linalg.norm(dp)), "score_difference_rms": rms,
        "score_reference_rms": float(np.sqrt(np.mean(clean * clean))),
        "score_difference_energy": e(diff), "direct_score_energy": e(direct),
        "local_score_energy": e(local), "interaction_score_energy": e(cross),
        "direct_local_inner": float(np.sum(direct * local)),
        "direct_interaction_inner": float(np.sum(direct * cross)),
        "local_interaction_inner": float(np.sum(local * cross)),
        "closure_max_abs": float(np.max(np.abs(diff - direct - local - cross))),
        "mean_user_norm": float(np.linalg.norm(branch.P.astype(np.float64), axis=1).mean()),
        "max_user_norm": float(np.linalg.norm(branch.P.astype(np.float64), axis=1).max()),
    }
    row["score_ratio_to_pulse"] = rms / pulse_rms if pulse_rms else 0.0
    # Deterministic tie convention: stable descending sort retains item-index order.
    s0, s1 = clean.copy(), shifted.copy()
    s0[reference.pos_mask] = -np.inf
    s1[reference.pos_mask] = -np.inf
    k = min(10, int((~reference.pos_mask).sum(axis=1).min()))
    top0 = np.argsort(-s0, axis=1, kind="stable")[:, :k]
    top1 = np.argsort(-s1, axis=1, kind="stable")[:, :k]
    churn = 1 - (top0[:, :, None] == top1[:, None, :]).any(axis=2).sum(axis=1) / k
    pair_rng = np.random.default_rng(6706)
    first = pair_rng.integers(reference.n_items, size=256)
    second = (first + pair_rng.integers(1, reference.n_items, size=256)) % reference.n_items
    m0, m1 = clean[:, first] - clean[:, second], shifted[:, first] - shifted[:, second]
    flips = ((m0 > 0) != (m1 > 0)).mean(axis=1)
    groups = {"all": np.ones(reference.n_users, dtype=bool), "exposed": exposed,
              "unexposed": ~exposed, "low": reference.groups == 0,
              "medium": reference.groups == 1, "high": reference.groups == 2}
    for name, mask in groups.items():
        row[f"users_{name}"] = int(mask.sum())
        if mask.any():
            row[f"top10_churn_{name}"] = float(churn[mask].mean())
            row[f"pair_flip_{name}"] = float(flips[mask].mean())
            row[f"score_rms_{name}"] = float(np.sqrt(np.mean(diff[mask] ** 2)))
    if row["closure_max_abs"] > 1e-10 * max(1, float(np.max(np.abs(shifted)))):
        raise AssertionError("score decomposition did not close")
    return row


def probe(model, alpha, replicate, checks=False):
    reference = copy.deepcopy(model)
    for name, code in (("sampling_rng", 6703), ("client_rng", 6704), ("noise_rng", 6705)):
        setattr(reference, name, np.random.default_rng([model.settings["seed"], code, replicate]))
    reference.settings["score_rounds"] = []
    reference.round = 0
    retained = copy.deepcopy(reference)
    tau = add_pulse(retained, alpha, replicate)
    selected = sample_clients(copy.deepcopy(reference.sampling_rng), reference.n_users,
                              reference.settings["q"])
    exposed = np.zeros(reference.n_users, dtype=bool)
    exposed[selected] = True
    initial = observe(retained, reference, exposed, "pulse", -1)
    pulse_rms = initial["score_difference_rms"]
    initial.update(branch="retained", tau=tau, score_ratio_to_pulse=1 if pulse_rms else 0)
    rows = [initial]
    reference.run_round()
    retained.run_round()
    assert_rng_equal(reference, retained)
    before = observe(retained, reference, exposed, "before_reset", 0, pulse_rms)
    before.update(branch="retained", tau=tau)
    rows.append(before)
    shared_reset, local_reset, both_reset = (copy.deepcopy(retained) for _ in range(3))
    shared_copy(shared_reset, reference)
    local_reset.P = reference.P.copy()
    shared_copy(both_reset, reference)
    both_reset.P = reference.P.copy()
    assert_equal(both_reset, reference)
    branches = {"retained": retained, "shared_reset": shared_reset, "local_reset": local_reset}
    for lag in range(11):
        if lag:
            reference.run_round()
            for branch in branches.values():
                branch.run_round()
                assert_rng_equal(branch, reference)
            if checks:
                both_reset.run_round()
                assert_equal(both_reset, reference)
        if alpha == 0:
            for branch in branches.values():
                assert_equal(branch, reference)
        if lag in (0, 1, 4, 10):
            for name, branch in branches.items():
                row = observe(branch, reference, exposed, "after_reset", lag, pulse_rms)
                row.update(branch=name, tau=tau)
                if name == "shared_reset" and lag == 0 and row["q_difference_fro"] != 0:
                    raise AssertionError("shared reset failed")
                rows.append(row)
    return rows


def synthetic_checks():
    enforce_and_report()
    train = pd.DataFrame({"user": np.repeat(np.arange(8), 3),
                          "item": np.array([(u * 2 + j) % 20 for u in range(8) for j in range(3)])})
    records = []
    for method in ("full", "fixed", "two"):
        c = dict(method=method, rank=2, dim=4, init_std=.01, seed=42, public_basis_seed=314159,
                 q=.5, local_lr=.1, local_epochs=2, reg=1e-5, server_lr=.1,
                 clip_norm=1., sigma=.2, score_rounds=[], pair_count=32)
        model = EffectiveNoiseBPR(train, 8, 20, c)
        for amplitude in (0., 1.):
            rows = probe(model, amplitude, 0, checks=True)
            records.append({"method": method, "alpha": amplitude, "observations": len(rows),
                            "max_closure": max(x["closure_max_abs"] for x in rows), "passed": True})
    return records


def run_job(job):
    record, alpha, replicate = job
    enforce_and_report()
    start = time.perf_counter()
    tag = f"{record['unit']}_{record['level']}_seed{record['seed']}_rep{replicate}_a{alpha:g}"
    path = OUT / "runs" / f"{tag}.json"
    if path.exists():
        raise FileExistsError(f"probe already exists: {tag}")
    checkpoint = ROOT / record["checkpoint"]
    digest = sha(checkpoint)
    if digest != record["artifact_hashes"][record["checkpoint"]]:
        raise RuntimeError("E1 checkpoint checksum mismatch")
    train = pd.read_csv(ROOT / "data/processed/ml-100k/train.csv")
    n_users = len(pd.read_csv(ROOT / "data/processed/ml-100k/user_map.csv"))
    n_items = len(pd.read_csv(ROOT / "data/processed/ml-100k/item_map.csv"))
    c = copy.deepcopy(record["settings"])
    model = EffectiveNoiseBPR(train, n_users, n_items, c)
    with np.load(checkpoint, allow_pickle=False) as values:
        model.load_state({k: values[k] for k in model.state()})
    result = dict(tag=tag, unit=record["unit"], level=record["level"], seed=record["seed"],
                  replicate=replicate, alpha=alpha, checkpoint=record["checkpoint"],
                  checkpoint_sha256=digest, status="ok", created_utc=stamp())
    try:
        result["observations"] = probe(model, alpha, replicate)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        result.update(status="failed", failure=repr(exc), observations=[])
    result["wall_time_s"] = time.perf_counter() - start
    write_json(path, result)
    return {k: result[k] for k in ("tag", "status", "wall_time_s")}


def summarize():
    rows, inventory = [], []
    for path in sorted((OUT / "runs").glob("*.json")):
        result = json.loads(path.read_text())
        observations = result.pop("observations")
        inventory.append(result)
        for row in observations:
            rows.append({**{k: result[k] for k in ("unit", "level", "seed", "replicate", "alpha")}, **row})
    pd.DataFrame(inventory).to_csv(OUT / "inventory.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "observations.csv", index=False, float_format="%.12g")
    keys = ["unit", "level", "alpha", "phase", "lag", "branch"]
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in keys + ["seed", "replicate"]]
    seeds = df.groupby(keys + ["seed"])[numeric].mean().reset_index()
    seeds.to_csv(OUT / "by_seed.csv", index=False, float_format="%.12g")
    summary = seeds.groupby(keys)[numeric].agg(["mean", "median", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "summary.csv", index=False, float_format="%.12g")
    print(json.dumps({"probes": len(inventory), "failures": sum(r["status"] != "ok" for r in inventory),
                      "observations": len(rows)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["check", "run", "summary"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.stage == "check":
        records = synthetic_checks()
        write_json(OUT / "synthetic_checks.json", {"created_utc": stamp(), "records": records})
        print(json.dumps(records))
    elif args.stage == "summary":
        summarize()
    else:
        checks = json.loads((OUT / "synthetic_checks.json").read_text())
        if not all(r["passed"] for r in checks["records"]):
            raise RuntimeError("synthetic checks not passed")
        records = [r for r in json.loads((E1 / "final_records.json").read_text())
                   if r["unit"] in UNITS and r["level"] in LEVELS and r["seed"] in SEEDS]
        if len(records) != 30:
            raise RuntimeError("expected exactly thirty E1 states")
        if (OUT / "freeze.json").exists():
            raise FileExistsError("E2 run already started; inspect inventory before any resume")
        sources = [Path(__file__), PROTOCOL, ROOT / "src/effective_noise.py", ROOT / "src/lowrank.py",
                   ROOT / "src/federated.py", ROOT / "src/privacy.py", ROOT / "src/threads.py"]
        write_json(OUT / "freeze.json", {"created_utc": stamp(), "probes": 120,
            "sources": {str(x.relative_to(ROOT)): sha(x) for x in sources},
            "input_manifest_sha256": sha(E1 / "MANIFEST.sha256"),
            "train_sha256": sha(ROOT / "data/processed/ml-100k/train.csv"),
            "privacy_claim": "diagnostics only; no DP release claim", "held_out_scoring": False})
        jobs = [(r, a, rep) for r in records for a in (.25, 1.) for rep in range(2)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, j) for j in jobs]
            for n, f in enumerate(as_completed(futures), 1):
                print(json.dumps({"completed": n, **f.result()}), flush=True)
        summarize()


if __name__ == "__main__":
    main()
