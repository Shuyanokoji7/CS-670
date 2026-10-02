"""E1 staged extension runner. No frozen baseline outputs are written.

Stages: --accounting, --profile, --search, --final-train, --freeze, --score-test.
All training stages evaluate validation only. Test requires a verified immutable
freeze snapshot and complete finite five-seed inventory. Existing jobs are reused
only after settings, code, split and artifact hashes match; partial outputs fail.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.threads import force_env
force_env()

import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from src.data import load_processed, split_fingerprint
from src.effective_noise import (EffectiveNoiseBPR, analytic_gaussian_sigma,
                                 bounded_popularity, gaussian_delta)
from src.evaluate import evaluate
from src.manifest import verify_manifest, write_manifest
from src.metrics import ndcg_from_rank, hit_rate_from_rank, mrr_from_rank
from src.privacy import compute_epsilon
from src.threads import enforce_and_report
from src.utils import load_config

CONFIG = ROOT / "configs/extensions/effective_noise_v1.yaml"
OUT = ROOT / "results/extensions/effective_noise_v1"
CKPT = ROOT / "checkpoints/extensions/effective_noise_v1"
LEVELS = ["nodp", "clip", "eps8", "eps4", "eps2", "eps1"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    tmp.replace(path)


def code_sha():
    paths = [ROOT/"src/effective_noise.py", Path(__file__)]
    return hashlib.sha256("".join(sha(p) for p in paths).encode()).hexdigest()


def load_all():
    cfg = load_config(CONFIG)
    split = load_processed(load_config(ROOT/cfg["dataset_config"]))
    return cfg, split, split_fingerprint(split)


def verify_protected():
    record = json.loads((OUT/"audit/protected_initial.json").read_text())
    bad = [p for p, h in record["files"].items() if not (ROOT/p).is_file() or sha(ROOT/p) != h]
    if bad:
        raise RuntimeError(f"protected baseline artifacts changed: {bad[:10]}")
    return len(record["files"])


def accounting(cfg):
    pc, fl = cfg["privacy"], cfg["federated"]
    key = {"privacy": pc, "q": fl["client_sampling_q"], "popularity": cfg["popularity"]}
    b2 = load_config(ROOT/"configs/b2.yaml")
    for name in ("T", "delta", "accountant", "cross_check_accountant", "denominator", "target_epsilons"):
        if pc[name] != b2["privacy"][name]:
            raise RuntimeError(f"E1/B2 accounting mismatch: {name}")
    if fl["client_sampling_q"] != b2["federated"]["client_sampling_q"]:
        raise RuntimeError("E1/B2 sampling mismatch")
    table = pd.read_csv(ROOT/"results/b2_accounting.csv")
    records = []
    for eps in pc["target_epsilons"]:
        match = table[(table.target_epsilon == eps) & (table["T"] == pc["T"])
                      & (table.q == fl["client_sampling_q"]) & (table.delta == pc["delta"])
                      & (table.accountant == pc["accountant"])
                      & (table.cross_check_accountant == pc["cross_check_accountant"])]
        if len(match) != 1 or float(match.iloc[0].noise_multiplier) != pc["sigmas"][eps]:
            raise RuntimeError("exact frozen B2 noise multiplier unavailable")
        sigma = pc["sigmas"][eps]
        prv = compute_epsilon(sigma, fl["client_sampling_q"], pc["T"], pc["delta"], pc["accountant"])
        rdp = compute_epsilon(sigma, fl["client_sampling_q"], pc["T"], pc["delta"], pc["cross_check_accountant"])
        if not .99*eps <= prv <= eps:
            raise RuntimeError("training epsilon outside declared tolerance")
        records.append({"mechanism": "training", "target_epsilon": eps, "sigma": sigma,
                        "epsilon": prv, "epsilon_cross_check": rdp, "q": fl["client_sampling_q"],
                        "T": pc["T"], "delta": pc["delta"], "accountant": pc["accountant"]})
        ps = analytic_gaussian_sigma(eps, pc["delta"])
        records.append({"mechanism": "popularity", "target_epsilon": eps, "sigma": ps,
                        "epsilon": eps, "epsilon_cross_check": compute_epsilon(ps, 1., 1, pc["delta"], "rdp"),
                        "q": 1., "T": 1, "delta": gaussian_delta(eps, ps), "accountant": "analytic_gaussian"})
    value = {"key": key, "records": records}
    path = OUT/"accounting.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["key"] != json.loads(json.dumps(key)):
            raise RuntimeError("existing E1 accounting key changed")
    else:
        write_json(path, value)
        pd.DataFrame(records).to_csv(OUT/"accounting.csv", index=False, float_format="%.12g")
    return value


def settings(cfg, method, rank, level, seed, candidate):
    pc, fl, m, diag = cfg["privacy"], cfg["federated"], cfg["model"], cfg["diagnostics"]
    eps = int(level[3:]) if level.startswith("eps") else None
    return {"method": method, "rank": rank, "level": level, "seed": int(seed),
            "dim": m["dim"], "init_std": m["init_std"], "public_basis_seed": m["public_basis_seed"],
            "q": fl["client_sampling_q"], "local_epochs": fl["local_epochs"], "reg": fl["l2_reg"],
            "local_lr": float(candidate["local_lr"]), "server_lr": float(candidate["server_lr"]),
            "T": pc["T"], "clip_norm": None if level == "nodp" else pc["clip_norm"],
            "sigma": pc["sigmas"][eps] if eps else 0., "delta": pc["delta"], "k": cfg["evaluation"]["k"],
            "score_rounds": diag["score_rounds"], "pair_count": diag["pair_count"]}


def unit(method, rank):
    return method if method == "full" else f"{method}_r{rank}"


def units(cfg):
    return [("full", cfg["model"]["dim"])] + [(m, r) for m in ("two", "fixed") for r in cfg["model"]["ranks"]]


def tag(c):
    suffix = hashlib.sha256(canonical(c).encode()).hexdigest()[:12]
    return (f"{unit(c['method'], c['rank'])}_{c['level']}_lr{c['local_lr']:g}"
            f"_slr{c['server_lr']:g}_seed{c['seed']}_{suffix}")


def check_record(row, c, fp):
    if row["settings"] != c or row["split_fingerprint"] != fp or row["code_sha256"] != code_sha():
        raise RuntimeError(f"settings/split/code changed: {row['tag']}")
    for rel, h in row["artifact_hashes"].items():
        if not (ROOT/rel).is_file() or sha(ROOT/rel) != h:
            raise RuntimeError(f"artifact hash mismatch: {rel}")
    if row["status"] == "ok" and row["rounds_run"] != c["T"]:
        raise RuntimeError("wrong training horizon")


def train_one(c):
    threads = enforce_and_report()
    cfg, split, fp = load_all()
    name = tag(c)
    path, checkpoint, rounds_path = OUT/"runs"/f"{name}.json", CKPT/f"{name}.npz", OUT/"runs"/f"{name}_rounds.csv"
    if path.exists():
        row = json.loads(path.read_text())
        check_record(row, c, fp)
        return row
    if checkpoint.exists() or rounds_path.exists():
        raise RuntimeError(f"partial outputs exist: {name}; refusing overwrite")
    started = time.perf_counter()
    sim = EffectiveNoiseBPR(split["train"], split["n_users"], split["n_items"], c)
    rounds, status, failure = [], "ok", None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(c["T"]):
                rounds.append(sim.run_round())
        summary, _ = evaluate(sim.full_scores, split, "validation", k=c["k"])
    except FloatingPointError as error:
        status, failure, summary = "diverged", str(error), {}
    except ValueError as error:
        if str(error) != "scores must be finite":
            raise
        status, failure, summary = "diverged", str(error), {}
    OUT.joinpath("runs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rounds).to_csv(rounds_path, index=False, float_format="%.12g")
    artifacts = {str(rounds_path.relative_to(ROOT)): sha(rounds_path)}
    if status == "ok":
        CKPT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint, **sim.state())
        artifacts[str(checkpoint.relative_to(ROOT))] = sha(checkpoint)
    row = {"tag": name, "settings": c, "unit": unit(c["method"], c["rank"]),
           "method": c["method"], "rank": c["rank"], "level": c["level"], "seed": c["seed"],
           "local_lr": c["local_lr"], "server_lr": c["server_lr"], "status": status,
           "failure": failure, "rounds_run": sim.round, "D": sim.D,
           "bytes_per_client_round": 8*sim.D, "total_bytes": sim.total_bytes,
           "wall_time_s": time.perf_counter()-started, "threads": threads,
           "split_fingerprint": fp, "code_sha256": code_sha(), "artifact_hashes": artifacts,
           "checkpoint": str(checkpoint.relative_to(ROOT)) if status == "ok" else None,
           "validation": summary}
    write_json(path, row)
    return row


def run_jobs(jobs, workers):
    if not 1 <= workers <= 12:
        raise ValueError("workers must be 1..12")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(train_one, jobs), 1):
            rows.append(row)
            if i % 36 == 0 or i == len(jobs):
                print(f"completed {i}/{len(jobs)}; failures {sum(r['status'] != 'ok' for r in rows)}", flush=True)
    return rows


def flat_table(rows):
    return pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                         | {"val_ndcg": r["validation"].get("ndcg@10", np.nan)} for r in rows])


def search(cfg, workers):
    if (OUT/"freeze.json").exists():
        raise RuntimeError("study already frozen; further search refused")
    jobs = [settings(cfg, method, rank, level, seed, candidate) for method, rank in units(cfg)
            for candidate in cfg["search"]["candidates"] for seed in cfg["search"]["seeds"] for level in LEVELS]
    rows = run_jobs(jobs, workers)
    flat_table(rows).to_csv(OUT/"search.csv", index=False, float_format="%.12g")
    chosen, screens = {}, []
    for method, rank in units(cfg):
        name = unit(method, rank)
        candidates = []
        for c in cfg["search"]["candidates"]:
            cells = [r for r in rows if r["unit"] == name and r["local_lr"] == c["local_lr"]
                     and r["server_lr"] == c["server_lr"]]
            eligible = len(cells) == 18 and all(r["status"] == "ok" for r in cells)
            mean = float(np.mean([r["validation"]["ndcg@10"] for r in cells if r["level"] == "eps4"])) if eligible else None
            screens.append({"unit": name, **c, "eligible": eligible, "eps4_validation_mean": mean,
                            "failures": sum(r["status"] != "ok" for r in cells)})
            if eligible:
                candidates.append((mean, c))
        if not candidates:
            write_json(OUT/"selection_bottleneck.json", {"unit": name, "reason": "no eligible finite candidate"})
            raise RuntimeError(f"{name}: no eligible candidate; bounded protocol bottleneck")
        candidates.sort(key=lambda x: (-x[0], abs(np.log(x[1]["server_lr"])),
                                        abs(np.log(x[1]["local_lr"]/5)), x[1]["local_lr"]))
        chosen[name] = {"method": method, "rank": rank, **candidates[0][1], "eps4_val_mean": candidates[0][0]}
    pd.DataFrame(screens).to_csv(OUT/"candidate_screen.csv", index=False, float_format="%.12g")
    write_json(OUT/"selected.json", chosen)
    print(json.dumps(chosen, indent=2), flush=True)


def popularity(cfg, split, fp):
    path = OUT/"popularity_validation.json"
    c = {"bound": cfg["popularity"]["l2_bound"], "delta": cfg["privacy"]["delta"],
         "seeds": cfg["seeds"], "epsilons": cfg["privacy"]["target_epsilons"]}
    if path.exists():
        record = json.loads(path.read_text())
        if record["settings"] != c or record["split_fingerprint"] != fp or record["code_sha256"] != code_sha():
            raise RuntimeError("popularity settings/split/code changed")
        for r in record["rows"]:
            if sha(ROOT/r["scores_path"]) != r["scores_sha256"]:
                raise RuntimeError("popularity score hash mismatch")
        return record
    base = bounded_popularity(split["train"], split["n_users"], split["n_items"], c["bound"])
    rows = []
    for eps in [None]+c["epsilons"]:
        sigma = analytic_gaussian_sigma(eps, c["delta"]) if eps else 0.
        for seed in c["seeds"] if eps else [c["seeds"][0]]:
            score = base + np.random.default_rng([seed, 9]).standard_normal(len(base))*sigma*c["bound"]
            name = f"pop_{'eps'+str(eps) if eps else 'bounded_nodp'}_seed{seed}"
            score_path = CKPT/f"{name}.npz"
            if score_path.exists():
                raise RuntimeError("partial popularity artifacts; refusing overwrite")
            CKPT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(score_path, scores=score)
            summary, _ = evaluate(lambda users: np.broadcast_to(score, (len(users), len(score))), split, "validation", k=10)
            rows.append({"tag": name, "level": f"eps{eps}" if eps else "bounded_nodp", "seed": seed,
                         "sigma": sigma, "validation": summary, "scores_path": str(score_path.relative_to(ROOT)),
                         "scores_sha256": sha(score_path)})
    record = {"settings": c, "rows": rows, "split_fingerprint": fp, "code_sha256": code_sha()}
    write_json(path, record)
    return record


def final_train(cfg, workers):
    selected = json.loads((OUT/"selected.json").read_text())
    jobs = [settings(cfg, c["method"], c["rank"], level, seed, c) for c in selected.values()
            for seed in cfg["seeds"] for level in LEVELS]
    rows = run_jobs(jobs, workers)
    flat_table(rows).to_csv(OUT/"final_validation.csv", index=False, float_format="%.12g")
    write_json(OUT/"final_records.json", rows)
    check_inventory(cfg, rows)
    ranks = []
    for method in ("two", "fixed"):
        for level in LEVELS:
            means = [(r, np.mean([x["validation"]["ndcg@10"] for x in rows if x["method"] == method
                                 and x["rank"] == r and x["level"] == level])) for r in cfg["model"]["ranks"]]
            r, mean = sorted(means, key=lambda x: (-x[1], x[0]))[0]
            ranks.append({"method": method, "level": level, "rank": r, "val_ndcg_mean": float(mean)})
    pd.DataFrame(ranks).to_csv(OUT/"rank_selection.csv", index=False, float_format="%.12g")
    _, split, fp = load_all()
    popularity(cfg, split, fp)


def check_inventory(cfg, rows):
    expected = {(unit(m, r), level, seed) for m, r in units(cfg) for level in LEVELS for seed in cfg["seeds"]}
    actual = [(r["unit"], r["level"], r["seed"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected) or any(r["status"] != "ok" for r in rows):
        raise RuntimeError("finite five-seed inventory gate failed; test refused")
    _, split, fp = load_all()
    selected = json.loads((OUT/"selected.json").read_text())
    for row in rows:
        c = selected[row["unit"]]
        expected_settings = settings(cfg, c["method"], c["rank"], row["level"], row["seed"], c)
        check_record(row, expected_settings, fp)
        state = dict(np.load(ROOT/row["checkpoint"], allow_pickle=False))
        expected_shapes = {"P": (split["n_users"], cfg["model"]["dim"])}
        expected_shapes.update({"Q": (split["n_items"], cfg["model"]["dim"])} if c["method"] == "full" else
                               {"A": (split["n_items"], c["rank"]), "B": (c["rank"], cfg["model"]["dim"])})
        if set(state) != set(expected_shapes) or any(state[k].shape != v or state[k].dtype != np.float32
                                                   or not np.isfinite(state[k]).all() for k, v in expected_shapes.items()):
            raise RuntimeError("invalid checkpoint state")
        sim = EffectiveNoiseBPR(split["train"], split["n_users"], split["n_items"], expected_settings)
        sim.load_state(state)


def freeze(cfg):
    if (OUT/"freeze.json").exists():
        require_frozen(cfg)
        print("existing freeze verified")
        return
    rows = json.loads((OUT/"final_records.json").read_text())
    check_inventory(cfg, rows)
    _, split, fp = load_all()
    popularity(cfg, split, fp)
    verify_protected()
    if not (OUT/"rank_selection.csv").is_file():
        raise RuntimeError("rank selection missing")
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot = OUT/"snapshots"/f"freeze_{now}"
    cfg["protocol_frozen"] = True
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    files = [CONFIG, ROOT/"src/effective_noise.py", Path(__file__), ROOT/"docs/extensions/EFFECTIVE_NOISE_PROTOCOL.md",
             OUT/"selected.json", OUT/"final_records.json", OUT/"final_validation.csv", OUT/"rank_selection.csv",
             OUT/"popularity_validation.json", OUT/"accounting.json", OUT/"candidate_screen.csv"]
    for file in files:
        dest = snapshot/file.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest)
    write_manifest(snapshot)
    if verify_manifest(snapshot):
        raise RuntimeError("freeze snapshot manifest failed")
    record = {"created_utc": now, "snapshot": str(snapshot.relative_to(ROOT)), "split_fingerprint": fp,
              "code_sha256": code_sha(), "files": {str(p.relative_to(ROOT)): sha(p) for p in files},
              "checkpoint_count": len(rows), "test_status": "no extension test scoring before this freeze"}
    write_json(OUT/"freeze.json", record)
    with (ROOT/"RESEARCH_LOG.md").open("a") as f:
        f.write(f"\n## {now} — E1 FINAL FREEZE before extension test scoring\n\n"
                f"- Complete finite five-seed inventory: {len(rows)} training checkpoints and 21 popularity score arrays.\n"
                "- Equal-budget selections and validation rank selection retained in the E1 output directory.\n"
                "- No extension test scoring has occurred; known historical ML-100K test exposure remains.\n"
                f"- Immutable snapshot: `{snapshot.relative_to(ROOT)}`; manifest verified.\n"
                "- Existing 3932 protected artifacts verified unchanged. Test scoring now authorized by the declared protocol.\n")
    print(f"freeze created: {snapshot}", flush=True)


def require_frozen(cfg):
    if not cfg.get("protocol_frozen") or not (OUT/"freeze.json").exists():
        raise RuntimeError("E1 is not frozen; test scoring refused")
    freeze_record = json.loads((OUT/"freeze.json").read_text())
    if verify_manifest(ROOT/freeze_record["snapshot"]):
        raise RuntimeError("freeze manifest changed")
    for p, h in freeze_record["files"].items():
        if sha(ROOT/p) != h:
            raise RuntimeError(f"frozen study input changed: {p}")
    if code_sha() != freeze_record["code_sha256"]:
        raise RuntimeError("frozen training code changed")


def save_test(name, scorer, split, checkpoint_hash):
    path, user_path = OUT/"test"/f"{name}.json", OUT/"test"/f"{name}_users.csv"
    if path.exists():
        r = json.loads(path.read_text())
        if r["checkpoint_sha256"] != checkpoint_hash or sha(user_path) != r["users_sha256"]:
            raise RuntimeError("existing test result mismatches checkpoint/output")
        return r
    if user_path.exists():
        raise RuntimeError("partial test output; refusing rescore")
    summary, ranks = evaluate(scorer, split, "test", k=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"user": np.arange(len(ranks)), "rank": ranks, "ndcg@10": ndcg_from_rank(ranks, 10),
                  "hr@10": hit_rate_from_rank(ranks, 10), "mrr@10": mrr_from_rank(ranks, 10)}).to_csv(
                      user_path, index=False, float_format="%.12g")
    result = {"tag": name, "summary": summary, "checkpoint_sha256": checkpoint_hash,
              "users_path": str(user_path.relative_to(ROOT)), "users_sha256": sha(user_path),
              "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(path, result)
    return result


def score_test(cfg):
    require_frozen(cfg)
    rows = json.loads((OUT/"final_records.json").read_text())
    check_inventory(cfg, rows)
    _, split, fp = load_all()
    results = []
    for row in rows:
        sim = EffectiveNoiseBPR(split["train"], split["n_users"], split["n_items"], row["settings"])
        sim.load_state(dict(np.load(ROOT/row["checkpoint"], allow_pickle=False)))
        result = save_test(row["tag"], sim.full_scores, split, sha(ROOT/row["checkpoint"]))
        results.append({"unit": row["unit"], "method": row["method"], "rank": row["rank"],
                        "level": row["level"], "seed": row["seed"], **result})
    for row in popularity(cfg, split, fp)["rows"]:
        score = np.load(ROOT/row["scores_path"], allow_pickle=False)["scores"]
        result = save_test(row["tag"], lambda users: np.broadcast_to(score, (len(users), len(score))),
                           split, row["scores_sha256"])
        results.append({"unit": "pop", "method": "pop", "rank": 0, "level": row["level"], "seed": row["seed"], **result})
    write_json(OUT/"test_records.json", results)
    print(f"{len(results)} retained models scored/cached; protected files: {verify_protected()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["accounting", "profile", "search", "final-train", "freeze", "score-test", "audit"])
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    cfg, split, _ = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "audit":
        print(f"{verify_protected()} protected files unchanged")
    elif args.stage == "accounting":
        print(pd.DataFrame(accounting(cfg)["records"]).to_string(index=False))
    elif args.stage == "profile":
        rows = [train_one(settings(cfg, m, r, "eps4", 42, cfg["search"]["candidates"][2])) for m, r in
                [("full", cfg["model"]["dim"]), ("two", 16), ("fixed", 16)]]
        print(flat_table(rows)[["unit", "wall_time_s", "status", "val_ndcg"]].to_string(index=False))
    elif args.stage == "search":
        if not (OUT/"accounting.json").exists():
            raise RuntimeError("run accounting before search")
        search(cfg, args.workers)
    elif args.stage == "final-train":
        final_train(cfg, args.workers)
    elif args.stage == "freeze":
        freeze(cfg)
    else:
        score_test(cfg)


if __name__ == "__main__":
    main()
