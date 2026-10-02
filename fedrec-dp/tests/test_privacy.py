"""B2 user-level DP tests: clipping, Gaussian aggregate noise, fixed denominator, accounting, locality."""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.privacy as priv
from src.data import load_processed, split_fingerprint
from src.federated import FederatedBPR
from src.privacy import (DPFederatedBPR, clip_update, compute_epsilon, gaussian_noise, private_update,
                         privacy_record, solve_sigma, update_norm)
from tests.test_data_split import FROZEN_SPLIT_FINGERPRINT

ROOT = Path(__file__).resolve().parent.parent
TOY = pd.DataFrame({"user": [0, 0, 1, 1, 2], "item": [0, 1, 2, 3, 4]})
N_U, N_I = 3, 8


def _cfg(q=1.0, C=None, sigma=0.0, denom="expected", lr=0.5, noise_seed=None, rounds=5):
    c = {
        "model": {"dim": 4, "init_std": 0.1},
        "federated": {"client_sampling_q": q, "local_lr": lr, "local_epochs": 1, "l2_reg": 0.0, "server_lr": 1.0,
                      "aggregation": "fedavg", "max_rounds": rounds, "eval_every": 1, "patience_evals": 10**6},
        "privacy": {"clip_norm": C, "noise_multiplier": sigma, "denominator": denom, "delta": 1e-5},
        "evaluation": {"k": 10},
    }
    if noise_seed is not None:
        c["privacy"]["noise_seed"] = noise_seed
    return c


def _dp(seed=0, **kw):
    return DPFederatedBPR(TOY, N_U, N_I, _cfg(**kw), seed)


# ---------------------------------------------------------------- clipping (1-3)

def test_update_norm_is_frobenius_of_whole_update():
    d = np.zeros((5, 3), np.float32); d[0] = [3, 0, 0]; d[4] = [0, 4, 0]
    assert update_norm(d) == pytest.approx(5.0)


def test_clipping_leaves_small_updates_unchanged():
    d = np.full((4, 2), 0.1, np.float32)
    out, norm, f = clip_update(d, C=10.0)
    assert out is d and f == 1.0 and norm == pytest.approx(np.sqrt(8 * 0.01))


def test_clipping_bounds_norm_and_keeps_direction():
    d = np.zeros((6, 2), np.float32); d[1] = [30, 40]; d[5] = [0, 120]           # norm 130, two rows
    out, norm, f = clip_update(d, C=1.3)
    assert norm == pytest.approx(130.0) and f == pytest.approx(0.01, rel=1e-5)
    assert update_norm(out) <= 1.3                                              # strictly within C
    np.testing.assert_allclose(out, d * np.float32(f), rtol=1e-6)               # one global scale, not per row


# ---------------------------------------------------------------- toy DP example (section 30) and aggregation (5)

def test_toy_dp_aggregation_end_to_end():
    shape = (3, 2)
    u1 = np.array([[3, 4], [0, 0], [0, 0]], np.float32)          # norm 5   -> scaled to 1
    u2 = np.array([[0, 0], [0.3, 0.4], [0, 0]], np.float32)      # norm 0.5 -> unchanged
    u3 = np.array([[0, 0], [0, 0], [600, 800]], np.float32)      # giant: norm 1000 -> scaled to 1
    C, sigma, denom = 1.0, 0.5, 2.5
    clipped = [clip_update(u, C)[0] for u in (u1, u2, u3)]
    np.testing.assert_allclose([update_norm(c) for c in clipped], [1.0, 0.5, 1.0], rtol=1e-5)
    assert all(update_norm(c) <= C for c in clipped)
    assert update_norm(clipped[2]) <= C                          # a giant update contributes at most C
    upd, total, noise = private_update(clipped, shape, sigma, C, np.random.default_rng(7), denom)
    expected_total = np.array([[0.6, 0.8], [0.3, 0.4], [0.6, 0.8]], np.float32)
    np.testing.assert_allclose(total, expected_total, rtol=1e-5)
    z = (np.random.default_rng(7).standard_normal(shape, dtype=np.float32) * np.float32(sigma * C))
    np.testing.assert_allclose(noise, z, rtol=0, atol=0)          # noise added to the SUM, N(0, (sigma*C)^2)
    np.testing.assert_allclose(upd, (total + z) / np.float32(denom), rtol=1e-6)
    upd2, _, _ = private_update(clipped, shape, sigma, C, np.random.default_rng(7), denom)
    np.testing.assert_array_equal(upd, upd2)                      # deterministic under a fixed noise seed


# ---------------------------------------------------------------- noise (6-7)

def test_gaussian_noise_shape_and_variance():
    rng = np.random.default_rng(0)
    z = gaussian_noise(rng, (1682, 64), sigma=1.7, C=0.3)
    assert z.shape == (1682, 64) and z.dtype == np.float32
    assert abs(z.mean()) < 0.01
    assert z.std() == pytest.approx(1.7 * 0.3, rel=0.01)                       # sigma*C, not divided by clients
    assert np.all(gaussian_noise(rng, (3, 3), 0.0, 1.0) == 0)


# ---------------------------------------------------------------- denominator and empty rounds (8-9)

def test_fixed_denominator_is_q_times_n(monkeypatch):
    sim = _dp(q=0.5, C=None, sigma=0.0, denom="expected")
    assert sim.denominator(0) == sim.denominator(3) == pytest.approx(0.5 * N_U)
    seen = {}
    orig = priv.private_update

    def spy(clipped, shape, sigma, C, rng, denominator):
        seen["denom"], seen["n"] = denominator, len(clipped)
        return orig(clipped, shape, sigma, C, rng, denominator)

    monkeypatch.setattr(priv, "private_update", spy)
    Q0 = sim.Q.copy()
    sim.run_round()
    assert seen["denom"] == pytest.approx(1.5)
    assert seen["n"] >= 0 and np.isfinite(sim.Q).all() and (seen["n"] == 0 or not np.array_equal(Q0, sim.Q))


def test_empty_round_is_defined_noised_and_counted(monkeypatch):
    sim = _dp(q=0.0, C=1.0, sigma=2.0, denom="expected")
    sim.q, sim.expected_clients = 0.0, 0.3 * N_U       # sampling never selects; denominator stays q*N > 0
    Q0, P0 = sim.Q.copy(), sim.P.copy()
    info = sim.run_round()
    assert info["clients"] == 0 and not info["skipped_empty"] and sim.round == 1
    noise = gaussian_noise(np.random.default_rng([0, 2]), Q0.shape, 2.0, 1.0)
    np.testing.assert_allclose(sim.Q, Q0 + noise / np.float32(0.3 * N_U), rtol=1e-6)
    np.testing.assert_array_equal(sim.P, P0)
    assert compute_epsilon(2.0, 0.3, sim.round, 1e-5) == compute_epsilon(2.0, 0.3, 1, 1e-5)   # it is an accounted step


# ---------------------------------------------------------------- noise seeds (10-11)

def test_noise_seed_reproduces_and_changes_output():
    a, b = _dp(C=1.0, sigma=1.0, noise_seed=5), _dp(C=1.0, sigma=1.0, noise_seed=5)
    c = _dp(C=1.0, sigma=1.0, noise_seed=6)
    for s in (a, b, c):
        s.run_round()
    np.testing.assert_array_equal(a.Q, b.Q)
    assert not np.array_equal(a.Q, c.Q)
    np.testing.assert_array_equal(a.P, c.P)        # only the noise stream differs; local training identical


# ---------------------------------------------------------------- accounting (12-17)

def test_epsilon_decreases_with_sigma_increases_with_T_and_q():
    e = lambda s, q, T: compute_epsilon(s, q, T, 1e-5, "rdp")
    assert e(1.0, 0.1, 100) > e(2.0, 0.1, 100) > e(4.0, 0.1, 100)
    assert e(2.0, 0.1, 100) < e(2.0, 0.1, 500) < e(2.0, 0.1, 1000)
    assert e(2.0, 0.05, 100) < e(2.0, 0.1, 100) < e(2.0, 0.2, 100)
    assert compute_epsilon(2.0, 0.1, 300, 1e-5, "prv") == compute_epsilon(2.0, 0.1, 300, 1e-5, "prv")
    assert compute_epsilon(2.0, 0.1, 300, 1e-5, "prv") < compute_epsilon(1.5, 0.1, 300, 1e-5, "prv")


def test_epsilon_independent_of_clip_norm():
    base = {"noise_multiplier": 2.0, "delta": 1e-5, "denominator": "expected"}
    r1 = privacy_record({**base, "clip_norm": 0.01}, 0.1, 200)
    r2 = privacy_record({**base, "clip_norm": 50.0}, 0.1, 200)
    assert r1["epsilon"] == r2["epsilon"] and r1["epsilon_cross_check"] == r2["epsilon_cross_check"]
    # the noise scale moves with C, so sensitivity / noise std is the same for every C
    assert _dp(C=0.01, sigma=2.0).C * 2.0 / 0.01 == pytest.approx(_dp(C=50.0, sigma=2.0).C * 2.0 / 50.0)


def test_solver_hits_target_epsilon():
    sigma, eps = solve_sigma(4.0, 0.1, 200, 1e-5, accountant="rdp", tol=0.01)
    assert 4.0 * 0.99 <= eps <= 4.0
    assert compute_epsilon(sigma, 0.1, 200, 1e-5, "rdp") == pytest.approx(eps)
    assert compute_epsilon(sigma * 0.97, 0.1, 200, 1e-5, "rdp") > 4.0           # near-minimal sigma


def test_no_noise_has_no_finite_guarantee():
    rec = privacy_record({"noise_multiplier": 0.0, "delta": 1e-5, "clip_norm": None, "denominator": "expected"},
                         0.1, 1000)
    assert rec["dp"] is False and rec["epsilon"] == float("inf") and rec["epsilon_cross_check"] == float("inf")
    with pytest.raises(ValueError):
        _dp(C=None, sigma=1.0)                                  # noise without clipping is meaningless


# ---------------------------------------------------------------- locality of p_u (4, 18)

def test_p_u_is_never_clipped_or_uploaded(monkeypatch):
    calls = []
    orig = priv.clip_update

    def spy(delta, C):
        calls.append(delta.shape)
        return orig(delta, C)

    monkeypatch.setattr(priv, "clip_update", spy)
    b1 = FederatedBPR(TOY, N_U, N_I, _cfg(), 0)
    dp = _dp(C=1e-4, sigma=0.5)                                  # extreme clipping and noise
    b1.run_round()
    dp.run_round()
    assert calls and all(s == (N_I, 4) for s in calls)           # only Q-shaped updates are clipped
    np.testing.assert_array_equal(b1.P, dp.P)                    # p_u identical to non-private B1 after the round


def test_unselected_p_u_unchanged_under_dp(monkeypatch):
    monkeypatch.setattr(priv, "sample_clients", lambda rng, n, q: np.array([1]))
    sim = _dp(C=0.5, sigma=1.0)
    P0 = sim.P.copy()
    sim.run_round()
    np.testing.assert_array_equal(sim.P[[0, 2]], P0[[0, 2]])
    assert not np.array_equal(sim.P[1], P0[1])


# ---------------------------------------------------------------- equivalence to B1

def test_no_clip_no_noise_realised_denominator_reproduces_b1_bitwise():
    b1 = FederatedBPR(TOY, N_U, N_I, _cfg(q=0.6), 3)
    dp = DPFederatedBPR(TOY, N_U, N_I, _cfg(q=0.6, C=None, sigma=0.0, denom="realised"), 3)
    for _ in range(8):
        b1.run_round(); dp.run_round()
    np.testing.assert_array_equal(b1.Q, dp.Q)
    np.testing.assert_array_equal(b1.P, dp.P)
    assert b1.comm_totals == dp.comm_totals


@pytest.fixture(scope="module")
def split():
    return load_processed()


def test_equivalence_to_b1_on_real_split(split):
    cfg = _cfg(q=0.1, C=None, sigma=0.0, denom="realised", lr=5.0)
    cfg["model"] = {"dim": 64, "init_std": 0.01}
    cfg["federated"].update(local_epochs=2, l2_reg=1e-5)
    b1 = FederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, 42)
    dp = DPFederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, 42)
    for _ in range(3):
        b1.run_round(); dp.run_round()
    np.testing.assert_array_equal(b1.Q, dp.Q)
    np.testing.assert_array_equal(b1.P, dp.P)


# ---------------------------------------------------------------- frozen artefacts (19-20)

def test_split_fingerprint_unchanged(split):
    assert split_fingerprint(split) == FROZEN_SPLIT_FINGERPRINT


B1_FROZEN_MD5 = {
    "results/b1_test_results.csv": "8d3cfcee6f29c7c9dba6338497988cbd",
    "results/b1_validation_results.csv": "f4206c263cda33e128465044da86b288",
    "results/b1_summary.csv": "4bdbdfd9539a185398d246bbb0249803",
    "results/b1_per_user_test.csv": "441f6bc92e3eddea176398232f943091",
    "results/raw/b1_seed42.csv": "9d6a47252262d0f2dc8b3d8443c87962",
    "results/raw/b1_seed123.csv": "d3c81b1f35f6a467db6ddf93a3229526",
    "results/raw/b1_seed2026.csv": "545eea1a0e7db8c1c258a00ce23c9453",
    "configs/b1.yaml": "fbea1b6e3e33813ea42af48be04698d2",
}


def test_b1_results_untouched():
    for rel, md5 in B1_FROZEN_MD5.items():
        assert hashlib.md5((ROOT / rel).read_bytes()).hexdigest() == md5, rel


# ---------------------------------------------------------------- checkpoints

def test_b2_checkpoint_metadata_and_rejection(split, tmp_path):
    from experiments.run_b2 import load_checkpoint, save_checkpoint

    cfg = _cfg(q=0.1, C=0.5, sigma=1.0)
    cfg["model"] = {"dim": 4, "init_std": 0.01}
    sim = DPFederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, 42)
    sim.run_round()
    rec = privacy_record(cfg["privacy"], 0.1, 1)
    path = tmp_path / "b2.pt"
    save_checkpoint(path, sim, cfg, {"k": 1}, 42, split_fingerprint(split), rec, {"ndcg@10": 0.0}, split)
    loaded, ckpt = load_checkpoint(path, split, expected_privacy=rec)
    np.testing.assert_array_equal(loaded.Q, sim.Q)
    np.testing.assert_array_equal(loaded.P, sim.P)
    for key in ("seed", "noise_seed", "q", "T", "clip_norm", "noise_multiplier", "delta", "epsilon", "accountant",
                "denominator", "round", "split_fingerprint", "config"):
        assert key in ckpt, key
    with pytest.raises(ValueError, match="privacy configuration"):
        load_checkpoint(path, split, expected_privacy={**rec, "noise_multiplier": 2.0})
    tampered = dict(split, test=split["test"].assign(item=split["validation"]["item"].to_numpy()))
    with pytest.raises(ValueError, match="different data split"):
        load_checkpoint(path, tampered)
