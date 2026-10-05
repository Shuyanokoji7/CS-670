"""E2b: repeat Two-r8 probes with orthogonally aligned common noise coupling."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_noise_memory as e2
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

OUT = ROOT / "results/extensions/noise_memory_aligned_v1"
PROTOCOL = ROOT / "docs/extensions/NOISE_MEMORY_ALIGNMENT_PROTOCOL.md"


def align(branch, reference):
    if np.array_equal(branch.A, reference.A) and np.array_equal(branch.B, reference.B):
        return 0.0, 0.0
    a, b = branch.A.astype(np.float64), branch.B.astype(np.float64)
    old = a @ b
    u, _, vt = np.linalg.svd(a.T @ reference.A.astype(np.float64) + b @ reference.B.astype(np.float64).T)
    r = u @ vt
    if not np.allclose(r.T @ r, np.eye(len(r)), atol=1e-12, rtol=0):
        raise AssertionError("nonorthogonal alignment")
    branch.A, branch.B = (a @ r).astype(np.float32), (r.T @ b).astype(np.float32)
    error = float(np.linalg.norm(branch.A.astype(np.float64) @ branch.B.astype(np.float64) - old)
                  / max(np.linalg.norm(old), 1e-30))
    if error > 1e-6:
        raise AssertionError("alignment changed effective matrix")
    return error, float(np.linalg.norm(r - np.eye(len(r))))


def probe(model, alpha, replicate, checks=False):
    reference = copy.deepcopy(model)
    for name, code in (("sampling_rng", 6703), ("client_rng", 6704), ("noise_rng", 6705)):
        setattr(reference, name, np.random.default_rng([model.settings["seed"], code, replicate]))
    reference.settings["score_rounds"] = []
    reference.round = 0
    retained = copy.deepcopy(reference)
    tau = e2.add_pulse(retained, alpha, replicate)
    selected = e2.sample_clients(copy.deepcopy(reference.sampling_rng), reference.n_users, reference.settings["q"])
    exposed = np.zeros(reference.n_users, dtype=bool)
    exposed[selected] = True
    row = e2.observe(retained, reference, exposed, "pulse", -1)
    pulse_rms = row["score_difference_rms"]
    row.update(branch="retained", tau=tau, score_ratio_to_pulse=1 if pulse_rms else 0)
    rows, alignment = [row], [align(retained, reference)]
    reference.run_round()
    retained.run_round()
    e2.assert_rng_equal(reference, retained)
    row = e2.observe(retained, reference, exposed, "before_reset", 0, pulse_rms)
    row.update(branch="retained", tau=tau)
    rows.append(row)
    sr, lr, both = (copy.deepcopy(retained) for _ in range(3))
    e2.shared_copy(sr, reference)
    lr.P = reference.P.copy()
    e2.shared_copy(both, reference)
    both.P = reference.P.copy()
    e2.assert_equal(both, reference)
    branches = {"retained": retained, "shared_reset": sr, "local_reset": lr}
    for lag in range(11):
        if lag:
            # All transformations use the same PRE-ROUND reference.
            for branch in branches.values():
                alignment.append(align(branch, reference))
            reference.run_round()
            for branch in branches.values():
                branch.run_round()
                e2.assert_rng_equal(branch, reference)
            if checks:
                both.run_round()
                e2.assert_equal(both, reference)
        if alpha == 0:
            for branch in branches.values():
                e2.assert_equal(branch, reference)
        if lag in (0, 1, 4, 10):
            for name, branch in branches.items():
                row = e2.observe(branch, reference, exposed, "after_reset", lag, pulse_rms)
                row.update(branch=name, tau=tau)
                if name == "shared_reset" and lag == 0 and row["q_difference_fro"] != 0:
                    raise AssertionError("shared reset failed")
                rows.append(row)
    for row in rows:
        row["alignment_product_relative_error_max"] = max(x[0] for x in alignment)
        row["alignment_rotation_distance_max"] = max(x[1] for x in alignment)
    return rows


def run_job(job):
    # Set these in each worker too, including multiprocessing spawn/forkserver.
    e2.OUT = OUT
    e2.probe = probe
    return e2.run_job(job)


def main():
    e2.enforce_and_report()
    # This independent module reuses the frozen E2 driver, changing only probe and output.
    e2.OUT = OUT
    e2.probe = probe
    e2.PROTOCOL = PROTOCOL
    e2.UNITS = ["two_r8"]
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        train = pd.DataFrame({"user": np.repeat(np.arange(8), 3),
                              "item": [(u * 2 + j) % 20 for u in range(8) for j in range(3)]})
        c = dict(method="two", rank=2, dim=4, init_std=.01, seed=42, public_basis_seed=314159,
                 q=.5, local_lr=.1, local_epochs=2, reg=1e-5, server_lr=.1,
                 clip_norm=1., sigma=.2, score_rounds=[], pair_count=32)
        model = e2.EffectiveNoiseBPR(train, 8, 20, c)
        records = []
        for alpha in (0., 1.):
            rows = probe(model, alpha, 0, checks=True)
            records.append({"alpha": alpha, "passed": True,
                            "max_preservation_error": max(r["alignment_product_relative_error_max"] for r in rows)})
        e2.write_json(OUT / "synthetic_checks.json", {"created_utc": e2.stamp(), "records": records})
        print(records)
        return
    if (OUT / "freeze.json").exists():
        raise FileExistsError("E2b already run")
    check = json.loads((OUT / "synthetic_checks.json").read_text())
    assert all(r["passed"] for r in check["records"])
    e2.write_json(OUT / "freeze.json", {"created_utc": e2.stamp(), "probes": 40,
        "protocol_sha256": e2.sha(PROTOCOL), "code_sha256": e2.sha(Path(__file__)),
        "e2_code_sha256": e2.sha(Path(e2.__file__)), "held_out_scoring": False,
        "privacy_claim": "diagnostic coupling only; not a new DP mechanism"})
    records = [r for r in json.loads((e2.E1 / "final_records.json").read_text())
               if r["unit"] == "two_r8" and r["level"] in e2.LEVELS and r["seed"] in e2.SEEDS]
    assert len(records) == 10
    jobs = [(r, a, rep) for r in records for a in (.25, 1.) for rep in range(2)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run_job, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            print(json.dumps({"completed": n, **f.result()}), flush=True)
    e2.summarize()


if __name__ == "__main__":
    main()
