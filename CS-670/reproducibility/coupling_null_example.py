"""A synthetic null example for paired noise attribution; no user data or training.

Equivalent factors (A,B) and (-A,-B) have the same effective matrix. Reusing
the same raw Gaussian arrays creates a nonzero paired effective-matrix distance.
Rotating those arrays with the factors makes that distance exactly zero.
This verifies elementary coupling algebra, not a novel theorem or a DP method.
"""
from pathlib import Path
import hashlib
import json
import numpy as np


def main():
    rng = np.random.default_rng(6707)
    m, d, r, n_users, draws, tau = 40, 12, 3, 16, 10000, .05
    a = rng.normal(size=(m, r)) / np.sqrt(r)
    b = rng.normal(size=(r, d)) / np.sqrt(d)
    p = rng.normal(size=(n_users, d)) / np.sqrt(d)
    initial_error = float(np.max(np.abs(a @ b - (-a) @ (-b))))
    energies, score_energies = [], []
    algebra_error, aligned_error = 0., 0.
    for _ in range(draws):
        za = rng.normal(scale=tau, size=a.shape)
        zb = rng.normal(scale=tau, size=b.shape)
        q1 = (a + za) @ (b + zb)
        q2_raw = (-a + za) @ (-b + zb)
        q2_aligned = (-a - za) @ (-b - zb)
        delta = q1 - q2_raw
        algebra_error = max(algebra_error, float(np.max(np.abs(delta - 2 * (za @ b + a @ zb)))))
        aligned_error = max(aligned_error, float(np.max(np.abs(q1 - q2_aligned))))
        energies.append(float(np.sum(delta ** 2)))
        score_energies.append(float(np.sum((p @ delta.T) ** 2)))
    expected_q = 4 * tau ** 2 * (m * np.sum(b ** 2) + d * np.sum(a ** 2))
    expected_s = 4 * tau ** 2 * (m * np.sum((p @ b.T) ** 2) + np.sum(a ** 2) * np.sum(p ** 2))
    assert initial_error == 0 and aligned_error == 0
    assert algebra_error < 1e-12
    result = {
        "purpose": "Synthetic null: gauge-dependent common-noise coupling; known algebra, no novelty claim",
        "design_status": "Authored after E2/E2b to verify their interpretation; no held-out data",
        "seed": 6707, "m": m, "d": d, "rank": r, "users": n_users,
        "draws": draws, "tau": tau, "dtype": "float64",
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_q_difference_max_abs": initial_error,
        "aligned_q_difference_max_abs": aligned_error,
        "linear_difference_identity_max_abs_error": algebra_error,
        "raw_q_energy_expected": float(expected_q),
        "raw_q_energy_empirical_mean": float(np.mean(energies)),
        "raw_q_energy_mc_standard_error": float(np.std(energies, ddof=1) / np.sqrt(draws)),
        "raw_score_energy_expected": float(expected_s),
        "raw_score_energy_empirical_mean": float(np.mean(score_energies)),
        "raw_score_energy_mc_standard_error": float(np.std(score_energies, ddof=1) / np.sqrt(draws)),
        "checks_passed": True,
    }
    out = Path(__file__).resolve().parents[1] / "results/noise_memory/coupling_null_example.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
