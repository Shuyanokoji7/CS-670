"""Centralised BPR-MF training (B0) with validation-based early stopping.

Training only ever receives the TRAIN split. Validation goes through the frozen
evaluator in src/evaluate.py via `validation_metrics`; the test split is not
touched here at all.
"""

import copy
import time

import numpy as np
import torch

from src.bpr import BPRMF, NegativeSampler, bpr_loss
from src.evaluate import evaluate


def set_determinism(seed, num_threads):
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True)


def build_model(n_users, n_items, cfg):
    return BPRMF(n_users, n_items, cfg["model"]["dim"], cfg["model"]["init_std"])


def validation_metrics(model, split, k):
    """Full-ranking validation metrics using the frozen evaluator."""
    model.eval()
    summary, ranks = evaluate(model.full_scores, split, "validation", k=k)
    return summary, ranks


def user_uniform_indices(users, n_users, n, rng):
    """Indices into the training pairs drawn as: user ~ Uniform(users), then positive ~ Uniform(that user's).

    In expectation this makes the batch-mean BPR loss equal to the user-weighted objective
    (1/N) sum_u (1/n_u) sum_k L_uk  (B0-UW), instead of B0's interaction-weighted mean.
    `users` must be sorted by user (true for the frozen train split).
    """
    starts = np.searchsorted(users, np.arange(n_users))
    counts = np.diff(np.append(starts, len(users)))
    if (counts == 0).any():
        raise ValueError("every user needs at least one training positive")
    u = rng.integers(0, n_users, size=n)
    return starts[u] + np.floor(rng.random(n) * counts[u]).astype(np.int64)


def train_bpr(train, n_users, n_items, cfg, seed, validate=None, log=print):
    """Train BPR-MF on `train` (DataFrame with user/item columns of TRAIN positives).

    validate: optional callable(model) -> dict with 'ndcg@K' (+ other metrics); called
    after every epoch and drives early stopping.
    Returns (model loaded with the best-validation weights, history list, best dict).
    """
    tc = cfg["training"]
    k = cfg["evaluation"]["k"]
    key = f"ndcg@{k}"
    set_determinism(seed, tc["num_threads"])
    rng = np.random.default_rng(seed)

    model = build_model(n_users, n_items, cfg)
    if tc["optimizer"] != "adam":
        raise ValueError("only adam is supported")
    opt = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])

    users = train["user"].to_numpy(dtype=np.int64)
    items = train["item"].to_numpy(dtype=np.int64)
    sampler = NegativeSampler(users, items, n_users, n_items, rng)
    n, bs = len(users), tc["batch_size"]
    example_sampling = tc.get("example_sampling", "interaction")
    if example_sampling == "user_uniform" and np.any(np.diff(users) < 0):
        raise ValueError("user_uniform sampling expects train rows sorted by user")

    history, best = [], {"epoch": 0, key: -np.inf, "state": copy.deepcopy(model.state_dict())}
    bad_epochs = 0
    for epoch in range(1, tc["max_epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        if example_sampling == "interaction":       # B0: every training interaction once per epoch
            order = rng.permutation(n)
        elif example_sampling == "user_uniform":    # B0-UW: |D| draws, user-uniform then positive-uniform
            order = user_uniform_indices(users, n_users, n, rng)
        else:
            raise ValueError(f"unknown example_sampling {example_sampling!r}")
        neg = sampler.sample(users[order])
        tot, tot_bpr = 0.0, 0.0
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            u = torch.from_numpy(users[idx])
            i = torch.from_numpy(items[idx])
            j = torch.from_numpy(neg[s:s + bs])
            pos, negs, sq = model(u, i, j)
            loss, pure = bpr_loss(pos, negs, sq, tc["l2_reg"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            tot_bpr += pure.item() * len(idx)
        train_time = time.perf_counter() - t0

        row = {"epoch": epoch, "loss": tot / n, "bpr_loss": tot_bpr / n,
               "learning_rate": opt.param_groups[0]["lr"], "train_time_s": train_time}
        if validate is not None:
            metrics = validate(model)
            row.update({f"val_{m}": metrics[m] for m in (f"ndcg@{k}", f"hr@{k}", f"recall@{k}", f"mrr@{k}")})
            if metrics[key] > best[key]:
                best = {"epoch": epoch, key: metrics[key], "state": copy.deepcopy(model.state_dict())}
                bad_epochs = 0
            else:
                bad_epochs += 1
        history.append(row)
        log(f"epoch {epoch:3d}  loss {row['loss']:.4f}  bpr {row['bpr_loss']:.4f}"
            + (f"  val ndcg@{k} {row[f'val_ndcg@{k}']:.4f}" if validate else "")
            + f"  ({train_time:.2f}s)")
        if validate is not None and bad_epochs >= tc["patience"]:
            log(f"early stop: no validation improvement for {tc['patience']} epochs "
                f"(best epoch {best['epoch']})")
            break

    if validate is not None:
        model.load_state_dict(best["state"])
    else:
        best = {"epoch": len(history), key: None, "state": copy.deepcopy(model.state_dict())}
    return model, history, best
