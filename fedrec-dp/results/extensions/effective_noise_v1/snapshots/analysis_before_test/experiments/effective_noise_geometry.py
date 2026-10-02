"""Public synthetic gauge sweep and Monte Carlo verification; no MovieLens labels."""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.threads import force_env, enforce_and_report
force_env()

import numpy as np
import pandas as pd
from src.effective_noise import expected_energy, perturbation_terms, public_basis


def main():
    enforce_and_report()
    out = ROOT/"results/extensions/effective_noise_v1"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    A, B = rng.normal(0, .1, (64, 8)), public_basis(16, 8).astype(np.float64)
    P = rng.normal(0, 1., (12, 16))
    gauges, mc = [], []
    for tau in (.005, .015, .03, .1):
        for scale in (.25, .5, 1., 2., 4.):
            energy = expected_energy(A*scale, B/scale, P, tau, "two")
            gauges.append({"tau": tau, "gauge": scale, "Q_norm": float(np.linalg.norm(A @ B)), **energy})
        for method in ("two", "fixed"):
            expected = expected_energy(A, B, P, tau, method)
            samples = []
            for _ in range(2000):
                ZA = rng.normal(0, tau, A.shape)
                ZB = rng.normal(0, tau, B.shape) if method == "two" else None
                dq = sum(perturbation_terms(A, B, ZA, ZB))
                samples.append([np.sum(dq*dq), np.sum((P @ dq.T)**2)])
            means = np.mean(samples, axis=0)
            mc.append({"method": method, "tau": tau, "draws": 2000,
                       "q_measured_energy": means[0], "q_expected_energy": expected["q_total"],
                       "q_ratio": means[0]/expected["q_total"],
                       "score_measured_energy": means[1], "score_expected_energy": expected["score_total"],
                       "score_ratio": means[1]/expected["score_total"]})
    pd.DataFrame(gauges).to_csv(out/"synthetic_gauge.csv", index=False, float_format="%.12g")
    pd.DataFrame(mc).to_csv(out/"synthetic_monte_carlo.csv", index=False, float_format="%.12g")
    print(pd.DataFrame(mc).to_string(index=False))


if __name__ == "__main__":
    main()
