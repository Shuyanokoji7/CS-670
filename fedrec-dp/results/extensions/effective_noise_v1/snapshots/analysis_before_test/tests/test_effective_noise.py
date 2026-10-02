"""Meaningful synthetic mechanism/gradient and freeze-gate checks for E1."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from src.effective_noise import (EffectiveNoiseBPR, analytic_gaussian_sigma, bounded_popularity,
                                 expected_energy, fixed_local_update, gaussian_delta,
                                 perturbation_terms, public_basis)
from src.federated import local_update, sample_negatives


def settings(method="fixed", rank=3):
    return {"method": method, "rank": rank, "level": "eps4", "seed": 42,
            "dim": 6, "init_std": .01, "public_basis_seed": 314159,
            "q": .5, "local_epochs": 2, "reg": 1e-5, "local_lr": 1.25,
            "server_lr": 1., "T": 3, "clip_norm": 1., "sigma": 1.2,
            "delta": 1e-5, "k": 3, "score_rounds": [1, 3], "pair_count": 16}


def toy_train():
    return pd.DataFrame({"user": [0, 0, 1, 1, 2, 2, 3, 3], "item": [0, 1, 2, 3, 4, 5, 6, 7]})


def test_public_basis_is_orthonormal_nested_independent_and_reproducible():
    small, large = public_basis(8, 3), public_basis(8, 6)
    np.testing.assert_array_equal(small, large[:3])
    np.testing.assert_allclose(small @ small.T, np.eye(3), atol=1e-7)
    np.testing.assert_array_equal(small, public_basis(8, 3))
    assert not np.array_equal(small, public_basis(8, 3, 9))


def test_fixed_gradient_matches_autograd_and_keeps_public_factor():
    rng = np.random.default_rng(4)
    A = rng.normal(0, .1, (8, 3)).astype(np.float32)
    B = public_basis(6, 3)
    p = rng.normal(0, .1, 6).astype(np.float32)
    pos = np.array([1, 2, 2])
    mask = np.zeros(8, dtype=bool)
    mask[pos] = True
    neg = sample_negatives(np.random.default_rng(9), mask, len(pos))
    pt = torch.tensor(p, requires_grad=True)
    at = torch.tensor(A, requires_grad=True)
    bt = torch.tensor(B, requires_grad=True)
    qi, qj = at[pos] @ bt, at[neg] @ bt
    loss = torch.nn.functional.softplus(-((qi-qj)*pt).sum(1)).mean()
    loss += .01*((pt*pt).sum()+(qi*qi).sum(1).mean()+(qj*qj).sum(1).mean())
    loss.backward()
    p_new, da, _ = fixed_local_update(p, A, B, pos, mask, .3, 1, .01, np.random.default_rng(9))
    np.testing.assert_allclose(p_new, p-.3*pt.grad.numpy(), atol=1e-7)
    np.testing.assert_allclose(da, -.3*at.grad.numpy(), atol=1e-7)
    np.testing.assert_array_equal(B, public_basis(6, 3))


def test_fixed_scores_and_updates_equivalent_to_rank_r_bpr():
    rng = np.random.default_rng(7)
    B = public_basis(6, 3)
    A, p = rng.normal(0, .1, (8, 3)).astype(np.float32), rng.normal(0, .1, 6).astype(np.float32)
    pos, mask = np.array([0, 1]), np.array([True, True, False, False, False, False, False, False])
    pf, da, _ = fixed_local_update(p, A, B, pos, mask, .5, 3, .01, np.random.default_rng(4))
    vr, dr, _, _ = local_update(B @ p, A, pos, mask, .5, 3, .01, np.random.default_rng(4))
    np.testing.assert_allclose(da, dr, atol=1e-7)
    np.testing.assert_allclose(B @ pf, vr, atol=1e-7)
    np.testing.assert_allclose((A+da) @ B @ pf, (A+dr) @ vr, atol=1e-7)


def test_two_and_fixed_initial_effective_matrix_and_local_users_match():
    fixed = EffectiveNoiseBPR(toy_train(), 4, 8, settings())
    two = EffectiveNoiseBPR(toy_train(), 4, 8, settings("two"))
    np.testing.assert_array_equal(fixed.P, two.P)
    np.testing.assert_allclose(fixed.Q, two.Q, atol=1e-8)


@pytest.mark.parametrize("method", ["fixed", "two"])
def test_exact_noise_decomposition_and_conditional_expectations(method):
    rng = np.random.default_rng(123)
    A, B, P = rng.normal(size=(5, 2)), rng.normal(size=(2, 3)), rng.normal(size=(4, 3))
    tau = .4
    theory = expected_energy(A, B, P, tau, method)
    energies, scores = [], []
    for _ in range(6000):
        ZA = rng.normal(0, tau, A.shape)
        ZB = rng.normal(0, tau, B.shape) if method == "two" else None
        terms = perturbation_terms(A, B, ZA, ZB)
        delta = sum(terms)
        exact = (A+ZA) @ (B+ZB if ZB is not None else B)-A @ B
        np.testing.assert_allclose(delta, exact, atol=1e-13)
        energies.append(np.sum(delta**2))
        scores.append(np.sum((P @ delta.T)**2))
    assert np.mean(energies) == pytest.approx(theory["q_total"], rel=.04)
    assert np.mean(scores) == pytest.approx(theory["score_total"], rel=.04)
    if method == "fixed":
        assert theory["q_cross"] == 0


@pytest.mark.parametrize("method", ["full", "two", "fixed"])
def test_empty_round_is_noised_accounted_and_local_users_are_unchanged(method):
    c = settings(method)
    c["q"] = 1e-9
    sim = EffectiveNoiseBPR(toy_train(), 4, 8, c)
    p0, q0 = sim.P.copy(), sim.Q.copy()
    info = sim.run_round()
    assert sim.round == 1 and info["clients"] == 0 and info["total_bytes"] == 0
    np.testing.assert_array_equal(sim.P, p0)
    assert not np.array_equal(sim.Q, q0)
    assert info["q_noise_energy"] > 0


def test_fixed_public_factor_is_bit_identical_through_private_rounds_and_reload():
    sim = EffectiveNoiseBPR(toy_train(), 4, 8, settings())
    basis = sim.B.copy()
    for _ in range(3):
        sim.run_round()
    np.testing.assert_array_equal(sim.B, basis)
    state = sim.state()
    state["B"][0, 0] += .1
    with pytest.raises(ValueError, match="public B changed"):
        sim.load_state(state)


@pytest.mark.parametrize("eps", [1, 2, 4, 8])
def test_analytic_gaussian_calibration(eps):
    sigma = analytic_gaussian_sigma(eps, 1e-5)
    assert gaussian_delta(eps, sigma) == pytest.approx(1e-5, rel=1e-8)
    assert gaussian_delta(eps, .99*sigma) > 1e-5
    assert gaussian_delta(eps, 1.01*sigma) < 1e-5


def test_popularity_has_whole_user_sensitivity_bound_and_deduplicates():
    train = toy_train()
    extra = pd.DataFrame({"user": [4]*8, "item": range(8)})
    base = bounded_popularity(train, 5, 8, 1.)
    enlarged = bounded_popularity(pd.concat([train, extra, extra]), 5, 8, 1.)
    assert np.linalg.norm(enlarged-base) <= 1.+1e-12
    np.testing.assert_allclose(base[:2], [1/np.sqrt(2)]*2)


def test_test_gate_requires_real_freeze_snapshot(tmp_path, monkeypatch):
    import experiments.run_effective_noise as runner
    monkeypatch.setattr(runner, "OUT", tmp_path)
    for frozen in (False, True):
        with pytest.raises(RuntimeError, match="not frozen"):
            runner.require_frozen({"protocol_frozen": frozen})


def test_fixed_isometry_and_exact_score_shock():
    rng = np.random.default_rng(12)
    B = public_basis(6, 3).astype(np.float64)
    A, ZA, P = rng.normal(size=(8, 3)), rng.normal(size=(8, 3)), rng.normal(size=(5, 6))
    dq = sum(perturbation_terms(A, B, ZA))
    np.testing.assert_allclose(np.linalg.norm(dq), np.linalg.norm(ZA), rtol=1e-7)
    np.testing.assert_allclose(P @ dq.T, P @ ((A+ZA) @ B).T-P @ (A @ B).T, atol=1e-12)


@pytest.mark.parametrize("method", ["full", "two", "fixed"])
def test_only_selected_local_users_change_and_diagnostics_do_not_consume_noise_rng(method):
    c = settings(method)
    sim = EffectiveNoiseBPR(toy_train(), 4, 8, c)
    other = EffectiveNoiseBPR(toy_train(), 4, 8, {**c, "score_rounds": []})
    selected = np.random.default_rng([c["seed"], 0]).random(4) < c["q"]
    initial = sim.P.copy()
    sim.run_round()
    other.run_round()
    np.testing.assert_array_equal(sim.P[~selected], initial[~selected])
    assert np.any(sim.P[selected] != initial[selected])
    np.testing.assert_array_equal(sim.Q, other.Q)


def test_full_extension_reproduces_existing_dp_mechanism():
    from src.privacy import DPFederatedBPR
    c = settings("full")
    cfg = {"model": {"dim": c["dim"], "init_std": c["init_std"]},
           "federated": {"client_sampling_q": c["q"], "local_lr": c["local_lr"],
                         "local_epochs": c["local_epochs"], "l2_reg": c["reg"],
                         "server_lr": c["server_lr"], "aggregation": "fedavg"},
           "privacy": {"clip_norm": c["clip_norm"], "noise_multiplier": c["sigma"], "denominator": "expected"}}
    baseline = DPFederatedBPR(toy_train(), 4, 8, cfg, c["seed"])
    extension = EffectiveNoiseBPR(toy_train(), 4, 8, c)
    for _ in range(3):
        baseline.run_round()
        extension.run_round()
    np.testing.assert_array_equal(baseline.P, extension.P)
    np.testing.assert_array_equal(baseline.Q, extension.Q)


def test_training_runner_cannot_touch_real_test_or_baseline_outputs(tmp_path, monkeypatch):
    import experiments.run_effective_noise as runner
    from tests.test_b2_protocol import SYNTH
    # Existing real-format synthetic fixture: no real held-out labels are loaded.
    split = copy.deepcopy(SYNTH)
    monkeypatch.setattr(runner, "OUT", tmp_path/"results")
    monkeypatch.setattr(runner, "CKPT", tmp_path/"checkpoints")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "code_sha", lambda: "synthetic-code")
    monkeypatch.setattr(runner, "load_all", lambda: ({}, split, "synthetic-split"))
    original = runner.evaluate
    def spy(scorer, value, target, **kwargs):
        assert value is split and target == "validation"
        return original(scorer, value, target, **kwargs)
    monkeypatch.setattr(runner, "evaluate", spy)
    c = settings()
    c["T"] = 2
    row = runner.train_one(c)
    assert row["status"] == "ok" and row["rounds_run"] == 2
    assert runner.train_one(c) == row  # hash-checked resume, no reevaluation
    path = tmp_path/row["checkpoint"]
    path.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner.train_one(c)
