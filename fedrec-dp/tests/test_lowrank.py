"""B3/B4 low-rank tests — SYNTHETIC data only (no MovieLens validation/test labels are ever scored here)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.lowrank as lr_mod
import src.train_federated
from src.bpr import bpr_loss
from src.federated import init_state, sample_negatives
from src.lowrank import (LowRankFederatedBPR, balance, init_lowrank_state, init_scale, lowrank_communication,
                         lowrank_local_update, shared_coordinates)
from src.train_lowrank import train_lowrank_converge, train_lowrank_fixed
from tests.test_b2_protocol import SYNTH, research_outputs_untouched  # noqa: F401  (module guard re-used)

ROOT = Path(__file__).resolve().parent.parent
TOY = pd.DataFrame({"user": [0, 0, 1, 1, 2], "item": [0, 1, 2, 3, 4]})
N_U, N_I = 3, 8


def _cfg(rank=2, q=1.0, lr=0.5, epochs=1, reg=0.0, dim=4, privacy=None, rounds=5):
    c = {"model": {"dim": dim, "init_std": 0.1, "rank": rank},
         "federated": {"client_sampling_q": q, "local_lr": lr, "local_epochs": epochs, "l2_reg": reg, "server_lr": 1.0,
                       "aggregation": "fedavg", "max_rounds": rounds, "eval_every": 1, "patience_evals": 10**6},
         "evaluation": {"k": 10}}
    if privacy is not None:
        c["privacy"] = {"delta": 1e-5, **privacy}
    return c


# ---------------------------------------------------------------- factors

def test_init_scale_matches_b1_variance_and_p_matches_b1():
    for r in (4, 8, 16, 32):
        assert r * init_scale(r) ** 4 == pytest.approx(1e-4)
    P_b1, _ = init_state(30, 40, 8, 0.01, 7)
    P, A, B = init_lowrank_state(30, 40, 8, 4, 7, 0.01)
    np.testing.assert_array_equal(P, P_b1)                          # same local user init as B1/B2
    assert A.shape == (40, 4) and B.shape == (4, 8)


def test_balance_preserves_product_and_balances_gauge():
    rng = np.random.default_rng(0)
    A = (rng.normal(size=(60, 5)) * 4).astype(np.float32)
    B = (rng.normal(size=(5, 12)) * 0.05).astype(np.float32)
    A2, B2 = balance(A, B)
    np.testing.assert_allclose(A2 @ B2, A @ B, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(A2.T @ A2, B2 @ B2.T, atol=1e-5)
    assert np.linalg.norm(A2) == pytest.approx(np.linalg.norm(B2), rel=1e-5)


# ---------------------------------------------------------------- local training maths

def test_chained_gradients_match_autograd_of_b1_objective():
    rng = np.random.default_rng(1)
    M, r, d, lr, reg = 9, 3, 5, 0.3, 0.02
    p = rng.normal(size=d).astype(np.float32)
    A = rng.normal(size=(M, r)).astype(np.float32)
    B = rng.normal(size=(r, d)).astype(np.float32)
    pos = np.array([1, 4, 4, 7])
    mask = np.zeros(M, bool); mask[pos] = True
    p_new, dA, dB, _, _ = lowrank_local_update(p, A, B, pos, mask, lr, 1, reg, np.random.default_rng(11))
    neg = sample_negatives(np.random.default_rng(11), mask, len(pos))            # the same draw
    pt, At, Bt = (torch.tensor(x, requires_grad=True) for x in (p, A, B))
    Q = At @ Bt
    u = pt.expand(len(pos), -1)
    qi, qj = Q[torch.tensor(pos)], Q[torch.tensor(neg)]
    sq = u.pow(2).sum(-1) + qi.pow(2).sum(-1) + qj.pow(2).sum(-1)              # B1's L2 on p and effective q rows
    total, _ = bpr_loss((u * qi).sum(-1), (u * qj).sum(-1), sq, reg)
    total.backward()
    np.testing.assert_allclose((p - p_new) / lr, pt.grad.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(-dA / lr, At.grad.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(-dB / lr, Bt.grad.numpy(), rtol=1e-4, atol=1e-6)


def test_dA_row_sparse_dB_dense_and_untouched_rows_unchanged():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(20, 3)).astype(np.float32)
    B = rng.normal(size=(3, 4)).astype(np.float32)
    p = rng.normal(size=4).astype(np.float32)
    mask = np.zeros(20, bool); mask[[0, 1]] = True
    _, dA, dB, _, touched = lowrank_local_update(p, A, B, np.array([0, 1]), mask, 0.5, 2, 0.1, rng)
    assert np.all(dA[~touched] == 0)       # no L2 applied directly to unseen rows' A; their q can still move via B
    assert np.count_nonzero(dB) == dB.size                           # B is shared by every item (by design)


def test_communication_counts_rank_coordinates():
    assert shared_coordinates(1682, 64, 8) == 8 * 1746
    c = lowrank_communication(3, n_items=10, dim=4, rank=2)
    assert c["download_bytes"] == c["upload_bytes"] == 3 * 2 * (10 + 4) * 4
    sim = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(q=0.6), 0)
    infos = [sim.run_round() for _ in range(6)]
    assert sim.comm_totals["upload_bytes"] == sum(i["clients"] for i in infos) * 2 * (N_I + 4) * 4


# ---------------------------------------------------------------- B4 mechanism

def test_joint_clip_is_one_norm_over_dA_and_dB(monkeypatch):
    giant_A = np.zeros((N_I, 2), np.float32); giant_A[0] = [300.0, 400.0]       # ||dA|| = 500
    small_B = np.full((2, 4), 0.5, np.float32)                                  # ||dB|| = 1.414 (< C alone)
    monkeypatch.setattr(lr_mod, "lowrank_local_update",
                        lambda p, A, B, *a, **k: (p, giant_A.copy(), small_B.copy(), 0.0, np.zeros(N_I, bool)))
    monkeypatch.setattr(lr_mod, "sample_clients", lambda rng, n, q: np.array([0]))
    monkeypatch.setattr(lr_mod, "balance", lambda A, B: (A, B))                 # inspect the raw update
    sim = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(privacy={"clip_norm": 2.0, "noise_multiplier": 0.0,
                                                            "denominator": "realised"}), 0)
    A0, B0 = sim.A.copy(), sim.B.copy()
    sim.run_round()
    uA, uB = sim.A - A0, sim.B - B0
    assert np.sqrt((uA.astype(np.float64) ** 2).sum() + (uB.astype(np.float64) ** 2).sum()) <= 2.0 + 1e-5
    f = 2.0 / np.sqrt(500.0 ** 2 + 8 * 0.25)
    np.testing.assert_allclose(uA, giant_A * f, rtol=1e-4, atol=1e-7)          # the SAME factor on both parts
    np.testing.assert_allclose(uB, small_B * f, rtol=1e-4, atol=1e-7)


def test_noise_covers_all_shared_coordinates_with_sigma_C(monkeypatch):
    seen = {}
    orig = lr_mod.private_update

    def spy(clipped, shape, sigma, C, rng, denominator):
        out = orig(clipped, shape, sigma, C, rng, denominator)
        seen.update(shape=shape, noise=out[2], denom=denominator)
        return out

    monkeypatch.setattr(lr_mod, "private_update", spy)
    sim = LowRankFederatedBPR(TOY, 3, 400, _cfg(rank=8, dim=16, q=0.5,
                                                privacy={"clip_norm": 0.5, "noise_multiplier": 2.0,
                                                         "denominator": "expected"}), 0)
    sim.run_round()
    assert seen["shape"] == (8 * (400 + 16),) and seen["noise"].shape == (8 * 416,)
    assert seen["noise"].std() == pytest.approx(1.0, rel=0.05)                  # sigma * C = 1.0
    assert seen["denom"] == pytest.approx(0.5 * 3)                              # fixed q N


def test_empty_round_is_noised_and_counted(monkeypatch):
    monkeypatch.setattr(lr_mod, "sample_clients", lambda rng, n, q: np.array([], dtype=int))
    sim = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(q=0.4, privacy={"clip_norm": 1.0, "noise_multiplier": 1.5,
                                                                  "denominator": "expected"}), 0)
    Q0, P0 = sim.Q.copy(), sim.P.copy()
    info = sim.run_round()
    assert info["clients"] == 0 and not info["skipped_empty"] and sim.round == 1 and info["noise_norm"] > 0
    assert not np.allclose(sim.Q, Q0)
    np.testing.assert_array_equal(sim.P, P0)


def test_b4_without_clip_noise_realised_equals_b3_bitwise():
    b3 = LowRankFederatedBPR(SYNTH["train"], SYNTH["n_users"], SYNTH["n_items"], _cfg(rank=3, dim=8, q=0.4, lr=2.0), 5)
    b4 = LowRankFederatedBPR(SYNTH["train"], SYNTH["n_users"], SYNTH["n_items"],
                             _cfg(rank=3, dim=8, q=0.4, lr=2.0, privacy={"clip_norm": None, "noise_multiplier": 0.0,
                                                                          "denominator": "realised"}), 5)
    for _ in range(6):
        b3.run_round(); b4.run_round()
    for x, y in ((b3.A, b4.A), (b3.B, b4.B), (b3.P, b4.P)):
        np.testing.assert_array_equal(x, y)
    assert b3.comm_totals == b4.comm_totals


def test_noise_seed_reproduces_and_changes_output():
    pc = {"clip_norm": 1.0, "noise_multiplier": 1.0, "denominator": "expected"}
    a = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(privacy={**pc, "noise_seed": 3}), 0)
    b = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(privacy={**pc, "noise_seed": 3}), 0)
    c = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(privacy={**pc, "noise_seed": 4}), 0)
    for s in (a, b, c):
        s.run_round()
    np.testing.assert_array_equal(a.Q, b.Q)
    assert not np.array_equal(a.Q, c.Q)


def test_p_u_local_only_and_unselected_unchanged(monkeypatch):
    monkeypatch.setattr(lr_mod, "sample_clients", lambda rng, n, q: np.array([1]))
    sim = LowRankFederatedBPR(TOY, N_U, N_I, _cfg(privacy={"clip_norm": 0.5, "noise_multiplier": 1.0,
                                                            "denominator": "expected"}), 0)
    P0 = sim.P.copy()
    sim.run_round()
    np.testing.assert_array_equal(sim.P[[0, 2]], P0[[0, 2]])
    assert not np.array_equal(sim.P[1], P0[1])


# ---------------------------------------------------------------- loops on the SYNTHETIC split

def _spy_validation_only(monkeypatch):
    seen = []
    real = src.train_federated.evaluate

    def wrapped(score_fn, split, target, *a, **k):
        assert split is SYNTH and split.get("synthetic"), "low-rank loop evaluated a non-synthetic split"
        assert target == "validation", "a training loop evaluated the test split"
        seen.append(target)
        return real(score_fn, split, target, *a, **k)

    monkeypatch.setattr(src.train_federated, "evaluate", wrapped)
    return seen


def test_converge_loop_validation_only_and_restores_best(monkeypatch):
    seen = _spy_validation_only(monkeypatch)
    cfg = _cfg(rank=3, dim=8, q=0.5, lr=2.0, rounds=30)
    cfg["federated"].update(eval_every=5, patience_evals=3)
    sim, hist, rounds, best = train_lowrank_converge(SYNTH, cfg, 0, log=lambda *_: None)
    assert seen and best["stopped_by"] in ("early_stopping", "max_rounds")
    assert len(rounds) == best["rounds_run"] and best["round"] % 5 == 0
    assert np.isfinite(sim.full_scores(np.arange(SYNTH["n_users"]))).all()


def test_converge_loop_records_divergence(monkeypatch):
    _spy_validation_only(monkeypatch)
    cfg = _cfg(rank=3, dim=8, q=0.5, lr=1e6, rounds=30)
    cfg["federated"].update(eval_every=5, patience_evals=100)
    cfg["model"]["init_std"] = 1.0
    with np.errstate(all="ignore"):
        _, _, _, best = train_lowrank_converge(SYNTH, cfg, 0, log=lambda *_: None)
    assert best["stopped_by"] == "diverged"


def test_fixed_loop_exactly_T_rounds(monkeypatch):
    _spy_validation_only(monkeypatch)
    cfg = _cfg(rank=3, dim=8, q=0.3, lr=2.0, privacy={"clip_norm": 1.0, "noise_multiplier": 1.0,
                                                       "denominator": "expected"})
    sim, hist, rounds, norms = train_lowrank_fixed(SYNTH, cfg, 0, T=7, eval_every=3, keep_client_norms=True)
    assert sim.round == 7 and len(rounds) == 7 and hist[-1]["round"] == 7
    assert all(r["noise_norm"] > 0 for r in rounds)


def test_toy_lowrank_federation_learns_personalised_rankings():
    # lr 1.0 (not the full-rank toy's 2.0): with Q = A B the effective step on q grows with the factor norms, and a
    # synthetic probe showed lr 2.0 diverging by round ~20 while lr 0.5-1.0 learns the rankings (recorded in the log).
    cfg = _cfg(rank=3, dim=4, q=1.0, lr=1.0, epochs=3, rounds=150)
    sim = LowRankFederatedBPR(TOY, N_U, N_I, cfg, 0)
    for _ in range(150):
        sim.run_round()
    scores = sim.full_scores([0, 1, 2])
    for u, pos in {0: [0, 1], 1: [2, 3], 2: [4]}.items():
        others = [i for i in range(N_I) if i not in pos]
        assert scores[u, pos].min() > scores[u, others].max(), (u, scores[u])


def test_confirmed_nonfinite_scores_are_recorded_as_divergence(monkeypatch):
    calls = {"n": 0}
    real = src.train_federated.validation_metrics

    def flaky(sim, split, k):
        calls["n"] += 1
        if calls["n"] == 3:                                  # make the scores genuinely overflow, then evaluate
            sim.P[:] = np.float32(3e38); sim.A[:] = 10.0; sim.B[:] = 10.0   # finite factors, overflowing scores
        return real(sim, split, k)

    import src.train_lowrank as tl
    monkeypatch.setattr(tl, "validation_metrics", flaky)
    cfg = _cfg(rank=3, dim=8, q=0.5, lr=1.0, rounds=40)
    cfg["federated"].update(eval_every=5, patience_evals=100)
    _, hist, _, best = train_lowrank_converge(SYNTH, cfg, 0, log=lambda *_: None)
    assert best["stopped_by"] == "diverged" and best["rounds_run"] == 10 and len(hist) == 2


def test_unrelated_value_errors_are_not_relabelled_as_divergence(monkeypatch):
    import src.train_lowrank as tl

    def broken(sim, split, k):
        raise ValueError("score_fn returned shape (3, 2), expected (3, 4)")     # a programming error

    monkeypatch.setattr(tl, "validation_metrics", broken)
    with pytest.raises(ValueError, match="score_fn returned shape"):
        train_lowrank_converge(SYNTH, _cfg(rank=3, dim=8, q=0.5, lr=1.0, rounds=5), 0, log=lambda *_: None)


def test_finite_scores_with_finite_message_is_not_relabelled(monkeypatch):
    import src.train_lowrank as tl

    def lying(sim, split, k):
        raise ValueError("scores must be finite")                               # message, but scores ARE finite

    monkeypatch.setattr(tl, "validation_metrics", lying)
    with pytest.raises(ValueError, match="scores must be finite"):
        train_lowrank_converge(SYNTH, _cfg(rank=3, dim=8, q=0.5, lr=1.0, rounds=5), 0, log=lambda *_: None)
