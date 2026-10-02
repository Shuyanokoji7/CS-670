"""B1 federated BPR tests: locality of p_u, aggregation maths, sampling, leakage, checkpoints."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch

import src.federated as fed
from src.bpr import bpr_loss
from src.data import load_processed, split_fingerprint
from src.federated import (FederatedBPR, aggregate, bpr_grads, local_update, round_communication,
                           sample_clients)
from src.train_federated import train_federated
from tests.test_data_split import FROZEN_SPLIT_FINGERPRINT


def _cfg(q=1.0, lr=1.0, epochs=1, reg=0.0, dim=8, rounds=10, agg="fedavg"):
    return {
        "model": {"dim": dim, "init_std": 0.1},
        "federated": {"client_sampling_q": q, "local_lr": lr, "local_epochs": epochs, "l2_reg": reg,
                      "server_lr": 1.0, "aggregation": agg, "max_rounds": rounds, "eval_every": 1,
                      "patience_evals": 10**6},
        "evaluation": {"k": 10},
    }


# Client 0 likes items 0, 1; client 1 likes items 2, 3; client 2 likes item 4; items 5-7 liked by nobody.
TOY = pd.DataFrame({"user": [0, 0, 1, 1, 2], "item": [0, 1, 2, 3, 4]})
N_U, N_I = 3, 8


def _sim(**kw):
    return FederatedBPR(TOY, N_U, N_I, _cfg(**kw), seed=0)


# ---------------------------------------------------------------- local training maths

def test_manual_gradients_match_autograd_of_b0_loss():
    rng = np.random.default_rng(0)
    p = rng.normal(size=6).astype(np.float32)
    Q = rng.normal(size=(9, 6)).astype(np.float32)
    pos, neg = np.array([1, 2, 2, 5]), np.array([0, 7, 3, 3])     # repeated rows on purpose
    reg = 0.01
    loss, _, gp, gqi, gqj = bpr_grads(p, Q, pos, neg, reg)
    grad_Q = np.zeros_like(Q)
    np.add.at(grad_Q, pos, gqi)
    np.add.at(grad_Q, neg, gqj)

    pt = torch.tensor(p, requires_grad=True)
    Qt = torch.tensor(Q, requires_grad=True)
    u = pt.expand(len(pos), -1)
    qi, qj = Qt[torch.tensor(pos)], Qt[torch.tensor(neg)]
    sq = u.pow(2).sum(-1) + qi.pow(2).sum(-1) + qj.pow(2).sum(-1)
    total, _ = bpr_loss((u * qi).sum(-1), (u * qj).sum(-1), sq, reg)
    total.backward()
    assert loss == pytest.approx(total.item(), rel=1e-5)
    np.testing.assert_allclose(gp, pt.grad.numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(grad_Q, Qt.grad.numpy(), rtol=1e-4, atol=1e-6)


def test_client_update_equals_local_minus_global_and_is_sparse():
    rng = np.random.default_rng(1)
    Q = rng.normal(size=(8, 4)).astype(np.float32)
    Q_before = Q.copy()
    p = rng.normal(size=4).astype(np.float32)
    mask = np.zeros(8, dtype=bool)
    mask[[0, 1]] = True
    p_new, dQ, _, touched = local_update(p, Q, np.array([0, 1]), mask, lr=0.5, local_epochs=1,
                                         reg=0.0, rng=np.random.default_rng(5))
    assert np.array_equal(Q, Q_before)                        # global Q untouched by the client
    neg = fed.sample_negatives(np.random.default_rng(5), mask, 2)   # same RNG stream -> same negatives
    _, _, gp, gqi, gqj = bpr_grads(p, Q, np.array([0, 1]), neg, 0.0)
    Q_local = Q.copy()
    np.add.at(Q_local, np.array([0, 1]), -0.5 * gqi)
    np.add.at(Q_local, neg, -0.5 * gqj)
    np.testing.assert_array_equal(dQ, Q_local - Q)
    np.testing.assert_array_equal(p_new, p - 0.5 * gp)
    assert np.all(dQ[~touched] == 0)                          # rows the client never used are exactly 0
    assert set(np.flatnonzero(touched)) == {0, 1} | set(neg.tolist())


# ---------------------------------------------------------------- aggregation

def test_fedavg_aggregation_toy_example():
    a = np.zeros((4, 2), np.float32); a[0] = [2, 2]; a[1] = [4, 0]
    b = np.zeros((4, 2), np.float32); b[1] = [0, 4]; b[2] = [6, 6]
    out = aggregate([a, b])
    np.testing.assert_array_equal(out, [[1, 1], [2, 2], [3, 3], [0, 0]])   # equal-user mean, row 3 untouched
    touched = [np.array([1, 1, 0, 0], bool), np.array([0, 1, 1, 0], bool)]
    diag = aggregate([a, b], touched, "item_mean")
    np.testing.assert_array_equal(diag, [[2, 2], [2, 2], [6, 6], [0, 0]])


def test_untouched_item_rows_are_not_shrunk():
    sim = _sim(q=1.0, lr=0.5, reg=0.1)
    before = sim.Q.copy()
    fed_touched = {}
    orig = fed.local_update

    def spy(*a, **k):
        out = orig(*a, **k)
        fed_touched.setdefault("t", []).append(out[3])
        return out

    fed.local_update = spy
    try:
        sim.run_round()
    finally:
        fed.local_update = orig
    untouched = ~np.any(fed_touched["t"], axis=0)
    assert untouched.any()
    np.testing.assert_array_equal(sim.Q[untouched], before[untouched])


def test_round_update_is_mean_of_client_deltas(monkeypatch):
    sim = _sim(q=1.0, lr=0.5, reg=0.01)
    Q_t = sim.Q.copy()
    seen = []
    orig = fed.aggregate

    def spy(deltas, touched=None, aggregation="fedavg"):
        seen.append([d.copy() for d in deltas])
        return orig(deltas, touched, aggregation)

    monkeypatch.setattr(fed, "aggregate", spy)
    sim.run_round()
    deltas = seen[0]
    assert len(deltas) == N_U and all(d.shape == (N_I, sim.dim) for d in deltas)   # only Q-shaped updates
    np.testing.assert_allclose(sim.Q, Q_t + np.mean(deltas, axis=0), rtol=0, atol=1e-7)


# ---------------------------------------------------------------- locality of p_u

def test_selected_client_receives_current_global_q(monkeypatch):
    sim = _sim(q=1.0)
    for _ in range(3):
        sim.run_round()
    Q_now = sim.Q.copy()
    received = []
    orig = fed.local_update

    def spy(p_u, Q_global, *a, **k):
        received.append(Q_global.copy())
        return orig(p_u, Q_global, *a, **k)

    monkeypatch.setattr(fed, "local_update", spy)
    sim.run_round()
    assert len(received) == N_U
    for r in received:
        np.testing.assert_array_equal(r, Q_now)


def test_selected_p_changes_unselected_p_does_not():
    sim = _sim(q=1.0, lr=0.5)
    P0 = sim.P.copy()
    selected = np.array([0, 2])
    fed_sample = fed.sample_clients
    fed.sample_clients = lambda rng, n, q: selected
    try:
        sim.run_round()
    finally:
        fed.sample_clients = fed_sample
    assert not np.array_equal(sim.P[0], P0[0]) and not np.array_equal(sim.P[2], P0[2])
    np.testing.assert_array_equal(sim.P[1], P0[1])


def test_p_u_never_influences_server_update():
    a, b = _sim(q=1.0, lr=0.5), _sim(q=1.0, lr=0.5)
    fed_sample = fed.sample_clients
    fed.sample_clients = lambda rng, n, q: np.array([0])      # only client 0 participates
    try:
        b.P[1] += 5.0                                          # change a NON-participating client's vector
        b.P[2] -= 3.0
        a.run_round()
        b.run_round()
    finally:
        fed.sample_clients = fed_sample
    np.testing.assert_array_equal(a.Q, b.Q)


def test_p_u_persists_across_rounds_and_is_never_reset():
    sim = _sim(q=1.0, lr=0.5)
    sim.run_round()
    after_one = sim.P.copy()
    sim.q = 0.0                      # empty rounds: nothing may change
    for _ in range(5):
        info = sim.run_round()
        assert info["skipped_empty"]
    np.testing.assert_array_equal(sim.P, after_one)


def test_empty_rounds_are_skipped_not_resampled():
    sim = _sim(q=0.0)
    Q0, P0 = sim.Q.copy(), sim.P.copy()
    info = sim.run_round()
    assert info["clients"] == 0 and info["skipped_empty"] and info["upload_bytes"] == 0
    np.testing.assert_array_equal(sim.Q, Q0)
    np.testing.assert_array_equal(sim.P, P0)


# ---------------------------------------------------------------- sampling / reproducibility

def test_fixed_seed_reproduces_client_selection():
    a, b = np.random.default_rng([7, 0]), np.random.default_rng([7, 0])
    for _ in range(20):
        np.testing.assert_array_equal(sample_clients(a, 942, 0.1), sample_clients(b, 942, 0.1))
    sizes = [len(sample_clients(np.random.default_rng([s, 0]), 942, 0.1)) for s in range(200)]
    assert 90 < np.mean(sizes) < 98                                 # Binomial(942, 0.1): mean 94.2, sd 9.2


def test_fixed_seed_reproduces_training():
    a, b = _sim(q=0.7, lr=0.5), _sim(q=0.7, lr=0.5)
    c = FederatedBPR(TOY, N_U, N_I, _cfg(q=0.7, lr=0.5), seed=1)
    for _ in range(5):
        a.run_round(); b.run_round(); c.run_round()
    np.testing.assert_array_equal(a.Q, b.Q)
    np.testing.assert_array_equal(a.P, b.P)
    assert not np.array_equal(a.Q, c.Q)


# ---------------------------------------------------------------- communication

def test_communication_counting():
    c = round_communication(3, n_items=5, dim=4, touched_rows_per_client=[2, 3, 1])
    assert c["download_bytes"] == 3 * 5 * 4 * 4 == c["upload_bytes"]
    assert c["bytes_per_client"] == 2 * 5 * 4 * 4
    assert c["sparse_upload_bytes_diag"] == 6 * (4 * 4 + 4)
    sim = _sim(q=0.6)
    infos = [sim.run_round() for _ in range(6)]
    per_client = N_I * sim.dim * 4
    assert sim.comm_totals["download_bytes"] == sum(i["clients"] for i in infos) * per_client
    assert sim.comm_totals["upload_bytes"] == sim.comm_totals["download_bytes"]


# ---------------------------------------------------------------- toy federated sanity

def test_toy_federation_learns_personalised_rankings():
    sim, _, _, _ = train_federated(TOY, N_U, N_I, _cfg(q=1.0, lr=2.0, epochs=3, rounds=150), seed=0,
                                   validate=None, log=lambda *_: None)
    init_P, init_Q = fed.init_state(N_U, N_I, 8, 0.1, 0)
    assert not np.allclose(sim.Q, init_Q)                           # the global item model changed
    assert not np.allclose(sim.P[0], sim.P[1])                      # user vectors stayed different
    scores = sim.full_scores([0, 1, 2])
    likes = {0: [0, 1], 1: [2, 3], 2: [4]}
    for u, pos in likes.items():
        others = [i for i in range(N_I) if i not in pos]
        assert scores[u, pos].min() > scores[u, others].max(), (u, scores[u])


# ---------------------------------------------------------------- real split: leakage, checkpoints

@pytest.fixture(scope="module")
def split():
    return load_processed()


def test_split_fingerprint_unchanged(split):
    assert split_fingerprint(split) == FROZEN_SPLIT_FINGERPRINT


def test_no_heldout_positive_used_for_training(split, monkeypatch):
    calls = []
    orig = fed.local_update

    def spy(p_u, Q_global, pos, pos_mask_row, *a, **k):
        calls.append((pos.copy(), pos_mask_row.copy()))
        return orig(p_u, Q_global, pos, pos_mask_row, *a, **k)

    monkeypatch.setattr(fed, "local_update", spy)
    cfg = _cfg(q=0.3, lr=1.0, dim=8, rounds=1)
    sim = FederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, seed=0)
    sim.run_round()
    train_by_user = split["train"].groupby("user")["item"].apply(set)
    heldout = set(zip(split["validation"].user, split["validation"].item)) | \
        set(zip(split["test"].user, split["test"].item))
    selected = sample_clients(np.random.default_rng([0, 0]), split["n_users"], 0.3)
    assert len(calls) == len(selected) > 0
    for u, (pos, mask) in zip(selected, calls):
        assert set(pos.tolist()) == train_by_user[u]
        assert not any((u, i) in heldout for i in pos)
        assert set(np.flatnonzero(mask)) == train_by_user[u]      # negatives exclude exactly train positives


def test_checkpoint_contains_all_clients_and_rejects_wrong_split(split, tmp_path):
    from experiments.run_b1 import load_checkpoint, save_checkpoint

    cfg = _cfg(q=0.1, dim=8)
    sim = FederatedBPR(split["train"], split["n_users"], split["n_items"], cfg, seed=3)
    sim.run_round()
    path = tmp_path / "b1.pt"
    save_checkpoint(path, sim, cfg, {"k": 1}, 3, split_fingerprint(split), {"round": 1, "rounds_run": 1, "stopped_by": "early_stopping"},
                    {"ndcg@10": 0.0}, split)
    loaded, ckpt = load_checkpoint(path, split)
    assert ckpt["P"].shape == (split["n_users"], 8) and ckpt["Q"].shape == (split["n_items"], 8)
    np.testing.assert_array_equal(loaded.P, sim.P)
    np.testing.assert_array_equal(loaded.Q, sim.Q)
    assert {"config", "seed", "round", "split_fingerprint", "validation", "client_sampling"} <= set(ckpt)
    tampered = dict(split, test=split["test"].assign(item=split["validation"]["item"].to_numpy()))
    with pytest.raises(ValueError, match="different data split"):
        load_checkpoint(path, tampered)
