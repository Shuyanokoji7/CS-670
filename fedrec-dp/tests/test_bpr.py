"""BPR-MF (B0) unit tests: model maths, gradients, toy sanity, leakage, reproducibility."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.bpr import BPRMF, NegativeSampler, bpr_loss
from src.data import load_processed, split_fingerprint
from src.train_bpr import train_bpr
from tests.test_data_split import FROZEN_SPLIT_FINGERPRINT


def _cfg(lr=0.05, reg=0.0, dim=8, epochs=100, bs=4, patience=1000):
    return {
        "model": {"dim": dim, "init_std": 0.1},
        "training": {"optimizer": "adam", "learning_rate": lr, "l2_reg": reg, "batch_size": bs,
                     "max_epochs": epochs, "patience": patience, "num_threads": 1},
        "evaluation": {"k": 10},
    }


def _quiet(*_):
    pass


# Toy data: user 0 likes items 0, 1; user 1 likes items 2, 3; items 4, 5 liked by nobody.
TOY = pd.DataFrame({"user": [0, 0, 1, 1], "item": [0, 1, 2, 3]})


def test_output_dimensions():
    m = BPRMF(5, 7, 16)
    assert m.user_emb.weight.shape == (5, 16) and m.item_emb.weight.shape == (7, 16)
    assert sum(p.numel() for p in m.parameters()) == (5 + 7) * 16
    u, i, j = torch.tensor([0, 4]), torch.tensor([1, 6]), torch.tensor([2, 3])
    pos, neg, sq = m(u, i, j)
    assert pos.shape == neg.shape == sq.shape == (2,)
    assert m.full_scores([0, 1, 2]).shape == (3, 7)


def test_scores_are_dot_products():
    m = BPRMF(2, 3, 2)
    with torch.no_grad():
        m.user_emb.weight[:] = torch.tensor([[1.0, 2.0], [0.5, -1.0]])
        m.item_emb.weight[:] = torch.tensor([[3.0, 1.0], [-1.0, 4.0], [0.0, 2.0]])
    pos, neg, sq = m(torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
    assert pos.tolist() == [5.0, -4.5]            # [1,2].[3,1], [.5,-1].[-1,4]
    assert neg.tolist() == [4.0, -2.0]            # [1,2].[0,2], [.5,-1].[0,2]
    assert sq.tolist() == pytest.approx([5 + 10 + 4, 1.25 + 17 + 4])
    np.testing.assert_allclose(m.full_scores([0]), [[5.0, 7.0, 4.0]])


def test_bpr_loss_value_and_finiteness():
    pos, neg, sq = torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 3.0])
    total, pure = bpr_loss(pos, neg, sq, reg=0.1)
    expected = (-math.log(1 / (1 + math.exp(-1.0))) + math.log(2)) / 2
    assert pure.item() == pytest.approx(expected)
    assert total.item() == pytest.approx(expected + 0.1 * 2.0)
    # stable for extreme score gaps
    big, _ = bpr_loss(torch.tensor([-1e4]), torch.tensor([1e4]), torch.tensor([0.0]), 0.0)
    small, _ = bpr_loss(torch.tensor([1e4]), torch.tensor([-1e4]), torch.tensor([0.0]), 0.0)
    assert torch.isfinite(big) and torch.isfinite(small) and small.item() == 0.0


def test_embeddings_receive_gradients():
    m = BPRMF(3, 4, 4)
    total, _ = bpr_loss(*m(torch.tensor([0]), torch.tensor([1]), torch.tensor([2])), reg=0.0)
    total.backward()
    assert m.user_emb.weight.grad[0].abs().sum() > 0
    assert m.item_emb.weight.grad[1].abs().sum() > 0 and m.item_emb.weight.grad[2].abs().sum() > 0
    # rows not in the batch get zero gradient
    assert m.user_emb.weight.grad[1:].abs().sum() == 0
    assert m.item_emb.weight.grad[[0, 3]].abs().sum() == 0


def test_negative_sampler_excludes_train_positives_only():
    rng = np.random.default_rng(0)
    s = NegativeSampler(np.array([0, 0, 1]), np.array([0, 1, 2]), 2, 4, rng)
    negs0 = s.sample(np.zeros(2000, dtype=int))
    negs1 = s.sample(np.ones(2000, dtype=int))
    assert set(negs0) == {2, 3} and set(negs1) == {0, 1, 3}


def test_loss_decreases_on_toy_data():
    _, history, _ = train_bpr(TOY, 2, 6, _cfg(epochs=50), seed=0, log=_quiet)
    losses = [h["bpr_loss"] for h in history]
    assert losses[-1] < 0.5 * losses[0]
    assert all(np.isfinite(losses))


def test_toy_sanity_ranks_positives_first():
    model, _, _ = train_bpr(TOY, 2, 6, _cfg(epochs=150), seed=0, log=_quiet)
    scores = model.full_scores([0, 1])
    likes = {0: [0, 1], 1: [2, 3]}
    for u, pos in likes.items():
        others = [i for i in range(6) if i not in pos]
        assert scores[u, pos].min() > scores[u, others].max(), (u, scores[u])


def test_seed_reproducibility():
    a, ha, _ = train_bpr(TOY, 2, 6, _cfg(epochs=5), seed=7, log=_quiet)
    b, hb, _ = train_bpr(TOY, 2, 6, _cfg(epochs=5), seed=7, log=_quiet)
    c, _, _ = train_bpr(TOY, 2, 6, _cfg(epochs=5), seed=8, log=_quiet)
    assert torch.equal(a.user_emb.weight, b.user_emb.weight)
    assert torch.equal(a.item_emb.weight, b.item_emb.weight)
    assert [h["loss"] for h in ha] == [h["loss"] for h in hb]
    assert not torch.equal(a.item_emb.weight, c.item_emb.weight)


# ---------------------------------------------------------------- real frozen split

@pytest.fixture(scope="module")
def split():
    return load_processed()


def test_split_fingerprint_unchanged(split):
    assert split_fingerprint(split) == FROZEN_SPLIT_FINGERPRINT


def test_training_uses_only_train_positives(split, monkeypatch):
    seen = []
    original = BPRMF.forward

    def spy(self, users, pos_items, neg_items):
        seen.append((users.numpy().copy(), pos_items.numpy().copy(), neg_items.numpy().copy()))
        return original(self, users, pos_items, neg_items)

    monkeypatch.setattr(BPRMF, "forward", spy)
    cfg = _cfg(lr=0.005, reg=1e-5, dim=8, epochs=1, bs=1024)
    train_bpr(split["train"], split["n_users"], split["n_items"], cfg, seed=0, log=_quiet)

    u = np.concatenate([s[0] for s in seen])
    i = np.concatenate([s[1] for s in seen])
    j = np.concatenate([s[2] for s in seen])
    train_pairs = set(zip(split["train"].user, split["train"].item))
    heldout = set(zip(split["validation"].user, split["validation"].item)) | \
        set(zip(split["test"].user, split["test"].item))
    positives = set(zip(u, i))
    assert positives == train_pairs                        # every train positive, once per epoch
    assert len(u) == len(split["train"])
    assert not positives & heldout                         # no held-out positive used as positive
    assert not any(p in train_pairs for p in zip(u, j))    # negatives are never train positives


def test_checkpoint_metadata_matches_frozen_split(split):
    from pathlib import Path
    from experiments.run_b0 import load_checkpoint

    path = Path(__file__).resolve().parent.parent / "checkpoints" / "b0_best.pt"
    if not path.exists():
        pytest.skip("run `python experiments/run_b0.py --config configs/b0.yaml --seed 42` first")
    model, ckpt = load_checkpoint(path, split)               # safe (weights_only) load + fingerprint check
    assert ckpt["split_fingerprint"] == FROZEN_SPLIT_FINGERPRINT
    assert {"config", "seed", "best_epoch", "validation", "dataset_config"} <= set(ckpt)
    assert ckpt["config"]["model"]["dim"] == 64
    assert model.user_emb.weight.shape == (split["n_users"], 64)
    assert model.item_emb.weight.shape == (split["n_items"], 64)
    tampered = dict(split, test=split["test"].assign(item=split["validation"]["item"].to_numpy()))
    with pytest.raises(ValueError, match="different data split"):
        load_checkpoint(path, tampered)


def test_user_uniform_sampler_is_user_then_item_uniform():
    from src.train_bpr import user_uniform_indices
    users = np.array([0, 1, 1, 1, 1, 2, 2])           # n_u = 1, 4, 2 (sorted, as in the frozen split)
    idx = user_uniform_indices(users, 3, 300_000, np.random.default_rng(0))
    sampled_users = users[idx]
    np.testing.assert_allclose(np.bincount(sampled_users) / len(idx), [1 / 3] * 3, atol=0.005)
    within = np.bincount(idx[sampled_users == 1] - 1, minlength=4) / (sampled_users == 1).sum()
    np.testing.assert_allclose(within, [0.25] * 4, atol=0.01)
    with pytest.raises(ValueError):
        user_uniform_indices(np.array([0, 2]), 3, 10, np.random.default_rng(0))   # user 1 has no positives


def test_default_example_sampling_is_b0_interaction_path():
    a, ha, _ = train_bpr(TOY, 2, 6, _cfg(epochs=3), seed=3, log=_quiet)
    cfg = _cfg(epochs=3)
    cfg["training"]["example_sampling"] = "interaction"
    b, hb, _ = train_bpr(TOY, 2, 6, cfg, seed=3, log=_quiet)
    assert torch.equal(a.item_emb.weight, b.item_emb.weight) and [h["loss"] for h in ha] == [h["loss"] for h in hb]
    cfg["training"]["example_sampling"] = "user_uniform"
    c, hc, _ = train_bpr(TOY, 2, 6, cfg, seed=3, log=_quiet)
    assert all(np.isfinite([h["loss"] for h in hc]))
