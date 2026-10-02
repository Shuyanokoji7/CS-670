"""B2 robustness-protocol regression tests (privacy-aware horizon study).

Every test that runs the B2 runner redirects RESULTS / RAW / CHECKPOINTS / CONFIG into a temporary directory and uses
an RDP-accountant copy of configs/b2.yaml (fast). Real experiments use PRV. A module-level guard asserts that no
research output under results/ or checkpoints/ is created or modified by this file.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import experiments.run_b2 as rb
import src.evaluate
import src.privacy as priv
import src.train_federated
from src.data import chronological_leave_two_out, split_fingerprint, validate_split
from src.privacy import compute_epsilon, privacy_record, solve_sigma
from src.train_dp_federated import train_dp_federated

ROOT = Path(__file__).resolve().parent.parent


def _snapshot():
    snap = {}
    for d in ("results", "checkpoints"):
        for dirpath, _, files in os.walk(ROOT / d):
            for f in files:
                p = Path(dirpath) / f
                st = p.stat()
                snap[str(p)] = (st.st_size, st.st_mtime_ns)
    return snap


@pytest.fixture(scope="module", autouse=True)
def research_outputs_untouched():
    before = _snapshot()
    yield
    after = _snapshot()
    assert after == before, "a B2 protocol test created or modified files under results/ or checkpoints/"


def synthetic_split():
    """A small valid split built by the real split code from synthetic ratings (no MovieLens data)."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(30):
        items = rng.choice(40, size=12, replace=False)
        for t, i in enumerate(items):
            rows.append((u + 1, int(i) + 1, int(rng.integers(1, 6)) if t % 3 else 5, 1000 + 10 * t))
    ratings = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    split, _ = chronological_leave_two_out(ratings, threshold=4, min_positives=3, split_seed=2026)
    validate_split(split)
    split["synthetic"] = True
    return split


SYNTH = synthetic_split()
SYNTH_FP = split_fingerprint(SYNTH)

# Explicit, test-only protocol values: independent of the mutable production configs/b2.yaml.
TEST_PRIVACY = {"unit": "user", "T": 3, "delta": 1e-5, "denominator": "expected", "accountant": "rdp",
                "cross_check_accountant": "rdp", "solver_tolerance": 0.01, "target_epsilons": [8, 4, 2, 1],
                "reference_epsilon": 4, "clip_grid": [1.0, 1.5, 2.4], "clip_norm": 1.5,
                "server_lr_grid": [0.5, 1.0, 2.0], "noise_multiplier": 0.0, "target_epsilon": None,
                "protocol_frozen": False, "t_grid": [100, 250, 500, 1000], "t_boundary": {"low": 50, "high": 1500},
                "t_seed_rule_margin": 0.003, "t_seed_rule_seeds": [123, 2026]}
TEST_FEDERATED = {"client_sampling_q": 0.1, "local_optimizer": "sgd", "local_lr": 5.0, "local_epochs": 2,
                  "l2_reg": 1e-5, "server_lr": 2.0, "aggregation": "fedavg", "max_rounds": 3, "eval_every": 10,
                  "patience_evals": 100}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolated runner: temp output dirs, a temp RDP config with explicit values, and the SYNTHETIC split.

    load_all is replaced, so no runner call made through this fixture can read MovieLens validation/test labels.
    """
    res, raw, ck = tmp_path / "results", tmp_path / "results" / "raw", tmp_path / "checkpoints"
    for d in (res, raw, ck):
        d.mkdir(parents=True, exist_ok=True)
    base = {"name": "b2_protocol_test", "dataset_config": "synthetic", "model": {"dim": 8, "init_std": 0.01},
            "federated": dict(TEST_FEDERATED), "privacy": dict(TEST_PRIVACY), "evaluation": {"k": 10}, "seed": 42}
    path = tmp_path / "b2_test.yaml"

    def write(**privacy_overrides):
        cfg = yaml.safe_load(yaml.safe_dump(base))
        cfg["privacy"].update(privacy_overrides)
        path.write_text(yaml.safe_dump(cfg))
        return cfg

    def fake_load_all():
        return yaml.safe_load(path.read_text()), {"synthetic": True}, SYNTH, SYNTH_FP

    monkeypatch.setattr(rb, "RESULTS", res)
    monkeypatch.setattr(rb, "RAW", raw)
    monkeypatch.setattr(rb, "CHECKPOINTS", ck)
    monkeypatch.setattr(rb, "CONFIG", str(path))
    monkeypatch.setattr(rb, "load_all", fake_load_all)
    write(protocol_frozen=False)
    return write, tmp_path


# ---------------------------------------------------------------- fresh sigma per T, monotone, cached

def test_sigma_recomputed_per_T_and_cached(iso, monkeypatch):
    write, tmp = iso
    cfg = write()
    calls = []
    orig = rb.solve_sigma

    def spy(*a, **k):
        calls.append(a[2])                                    # T
        return orig(*a, **k)

    monkeypatch.setattr(rb, "solve_sigma", spy)
    s100 = rb.sigma_for(cfg, 4, 100)
    s250 = rb.sigma_for(cfg, 4, 250)
    assert calls == [100, 250]                                # a fresh solve for every new T
    assert s250 > s100
    assert rb.sigma_for(cfg, 4, 100) == s100 and calls == [100, 250]   # cached: no re-solve
    tab = pd.read_csv(rb.RESULTS / "b2_accounting.csv")
    assert sorted(tab["T"].tolist()) == [100, 250]
    for _, r in tab.iterrows():                                # stored eps is the accountant's value for that T
        assert r["epsilon"] == pytest.approx(compute_epsilon(r["noise_multiplier"], 0.1, int(r["T"]), 1e-5, "rdp"))
        assert r["epsilon"] <= 4.0


def test_required_sigma_increases_with_T_at_fixed_epsilon():
    sigmas = [solve_sigma(4.0, 0.1, T, 1e-5, accountant="rdp")[0] for T in (50, 100, 250, 500)]
    assert all(a < b for a, b in zip(sigmas, sigmas[1:]))


# ---------------------------------------------------------------- exactly T steps, including empty rounds

def test_exactly_T_steps_executed_and_accounted_including_empty_rounds(monkeypatch):
    toy = pd.DataFrame({"user": [0, 0, 1], "item": [0, 1, 2]})
    split = {"train": toy, "n_users": 2, "n_items": 5}
    cfg = {"model": {"dim": 3, "init_std": 0.1},
           "federated": {"client_sampling_q": 0.1, "local_lr": 0.5, "local_epochs": 1, "l2_reg": 0.0,
                         "server_lr": 1.0, "aggregation": "fedavg", "eval_every": 1},
           "privacy": {"clip_norm": 1.0, "noise_multiplier": 1.3, "denominator": "expected", "delta": 1e-5},
           "evaluation": {"k": 10}}
    monkeypatch.setattr(priv, "sample_clients", lambda rng, n, q: np.array([], dtype=int))   # every round empty
    sim, _, rounds, _ = train_dp_federated(split, cfg, seed=0, T=7, eval_every=7, validate=False)
    assert sim.round == 7 and len(rounds) == 7
    assert all(r["clients"] == 0 for r in rounds)
    assert all(r["noise_norm"] > 0 for r in rounds)           # empty rounds are still noised (a mechanism step)
    rec = privacy_record(cfg["privacy"], 0.1, sim.round, "rdp", "rdp")
    assert rec["T"] == 7 and rec["epsilon"] == compute_epsilon(1.3, 0.1, 7, 1e-5, "rdp")


# ---------------------------------------------------------------- validation-only never touches test

def _spy_evaluator(monkeypatch, allow_test):
    """Wrap every evaluator reference. Test-split calls are forbidden, or (allow_test) must receive the SYNTHETIC split."""
    seen = []
    real = src.evaluate.evaluate

    def wrapped(*a, **k):
        target = a[2] if len(a) > 2 else k.get("target")
        split = a[1] if len(a) > 1 else k.get("split")
        seen.append(target)
        assert split is SYNTH and split.get("synthetic"), "a regression job evaluated a non-synthetic split"
        if target == "test" and not allow_test:
            raise AssertionError("test split evaluated in a validation-only path")
        return real(*a, **k)

    for mod in (rb, src.train_federated, src.evaluate):
        monkeypatch.setattr(mod, "evaluate", wrapped)
    return seen


def test_validation_only_run_one_never_calls_test(iso, monkeypatch):
    seen = _spy_evaluator(monkeypatch, allow_test=False)
    df = rb.run_one(rb.job(4, 42, T=2))
    assert "validation" in seen and "test" not in seen
    assert list(df["split"]) == ["validation"]
    assert int(df["T"].iloc[0]) == 2 and int(df["rounds"].iloc[0]) == 2
    assert not list(rb.CHECKPOINTS.iterdir())                 # no checkpoint for validation-only runs
    assert not list(rb.RAW.glob("**/per_user_test_*"))


# ---------------------------------------------------------------- frozen gate

def test_unfrozen_protocol_rejects_test_jobs_and_sweep(iso, monkeypatch):
    trained = []
    monkeypatch.setattr(rb, "train_one", lambda *a, **k: trained.append(1))
    with pytest.raises(RuntimeError, match="not frozen"):
        rb.run_one(rb.job(4, 42, evaluate_test=True, T=2))
    with pytest.raises(RuntimeError, match="not frozen"):
        rb.sweep([42], workers=1)
    assert not trained                                        # refused before any training


# ---------------------------------------------------------------- output / checkpoint identity

def test_run_tag_distinguishes_T_C_and_server_lr():
    def c(T, C, slr):
        return {"privacy": {"T": T, "clip_norm": C}, "federated": {"server_lr": slr}}
    tags = {rb.run_tag(c(T, C, s), e, 42) for T in (100, 1000) for C in (None, 1.5) for s in (1.0, 2.0) for e in (None, 4)}
    assert len(tags) == 16
    assert rb.run_tag(c(250, None, 1.0), None, 42) != rb.run_tag(c(250, None, 2.0), None, 42)   # the old collision


def test_frozen_test_jobs_write_distinct_T_C_slr_outputs(iso, monkeypatch):
    write, _ = iso
    write(protocol_frozen=True)
    seen = _spy_evaluator(monkeypatch, allow_test=True)      # every evaluation must receive the SYNTHETIC split
    rb.run_one(rb.job(4, 42, evaluate_test=True, T=2))
    rb.run_one(rb.job(None, 42, evaluate_test=True, T=2))
    rb.run_one(rb.job(None, 42, server_lr=1.0, evaluate_test=True, T=2))
    assert seen.count("test") == 3                            # the test path ran, on synthetic data only
    C, slr = TEST_PRIVACY["clip_norm"], TEST_FEDERATED["server_lr"]
    names = sorted(p.name for p in rb.CHECKPOINTS.iterdir())
    assert names == sorted([f"b2_eps4_T2_C{C}_slr{slr}_seed42.pt", "b2_nodp_T2_CNone_slr1.0_seed42.pt",
                            f"b2_nodp_T2_CNone_slr{slr}_seed42.pt"])
    import torch
    ck = torch.load(rb.CHECKPOINTS / f"b2_eps4_T2_C{C}_slr{slr}_seed42.pt", weights_only=True)
    assert ck["T"] == 2 and ck["round"] == 2 and ck["clip_norm"] == C
    assert ck["config"]["federated"]["server_lr"] == slr
    assert ck["split_fingerprint"] == SYNTH_FP and ck["n_users"] == SYNTH["n_users"]
    assert len(list(rb.RAW.glob("b2_runs/per_user_test_*_T2_*"))) == 3


def test_synthetic_split_is_valid_and_not_movielens():
    assert SYNTH["synthetic"] and SYNTH["n_items"] <= 40 and SYNTH["n_users"] <= 30
    from tests.test_data_split import FROZEN_SPLIT_FINGERPRINT
    assert SYNTH_FP != FROZEN_SPLIT_FINGERPRINT


# ---------------------------------------------------------------- sigma cache key (q, delta, accountants)

def _legacy_row(**kw):
    row = {"target_epsilon": 4.0, "noise_multiplier": 9.99, "epsilon": 3.99, "accountant": "rdp",
           "epsilon_cross_check": 3.99, "cross_check_accountant": "rdp", "delta": 1e-5, "q": 0.1, "T": 100}
    row.update(kw)
    return row


def _count_solves(monkeypatch):
    calls = []
    orig = rb.solve_sigma

    def spy(*a, **k):
        calls.append(a)
        return orig(*a, **k)

    monkeypatch.setattr(rb, "solve_sigma", spy)
    return calls


@pytest.mark.parametrize("change", ["primary", "cross_check", "q", "delta"])
def test_cache_key_mismatch_forces_fresh_solve_and_keeps_old_row(iso, monkeypatch, change):
    write, _ = iso
    cfg = write()
    legacy = {"primary": _legacy_row(accountant="prv"), "cross_check": _legacy_row(cross_check_accountant="prv"),
              "q": _legacy_row(q=0.2), "delta": _legacy_row(delta=1e-6)}[change]
    pd.DataFrame([legacy]).to_csv(rb.RESULTS / "b2_accounting.csv", index=False, float_format="%.12g")
    calls = _count_solves(monkeypatch)
    sigma = rb.sigma_for(cfg, 4, 100)
    assert len(calls) == 1 and sigma != 9.99                  # fresh solve under the configured key
    tab = pd.read_csv(rb.RESULTS / "b2_accounting.csv")
    assert len(tab) == 2 and 9.99 in tab["noise_multiplier"].tolist()   # old row retained
    assert rb.cached_accounting_row(cfg, 4, 100)["noise_multiplier"] == sigma
    assert rb.sigma_for(cfg, 4, 100) == sigma and len(calls) == 1      # now cached under the full key


def test_matching_full_key_is_reused_without_solving(iso, monkeypatch):
    write, _ = iso
    cfg = write()
    pd.DataFrame([_legacy_row()]).to_csv(rb.RESULTS / "b2_accounting.csv", index=False, float_format="%.12g")
    calls = _count_solves(monkeypatch)
    assert rb.sigma_for(cfg, 4, 100) == 9.99 and not calls


def test_cached_row_outside_solver_tolerance_is_not_reused(iso, monkeypatch):
    write, _ = iso
    cfg = write()
    pd.DataFrame([_legacy_row(epsilon=3.0)]).to_csv(rb.RESULTS / "b2_accounting.csv", index=False)   # far below 4
    calls = _count_solves(monkeypatch)
    assert rb.sigma_for(cfg, 4, 100) != 9.99 and len(calls) == 1


def test_conflicting_cached_rows_raise(iso):
    write, _ = iso
    cfg = write()
    pd.DataFrame([_legacy_row(), _legacy_row(noise_multiplier=8.88)]).to_csv(rb.RESULTS / "b2_accounting.csv",
                                                                             index=False)
    with pytest.raises(ValueError, match="conflicting"):
        rb.sigma_for(cfg, 4, 100)


def test_accounting_uses_full_key_rows(iso, capsys):
    write, _ = iso
    cfg = write(T=50, target_epsilons=[4])
    pd.DataFrame([_legacy_row(T=50, accountant="prv", noise_multiplier=9.99)]).to_csv(
        rb.RESULTS / "b2_accounting.csv", index=False)
    rb.accounting()
    out = capsys.readouterr().out
    assert "sigma 9.99" not in out and "-> rdp eps" in out    # reported row is the fresh rdp one, not the prv row
    tab = pd.read_csv(rb.RESULTS / "b2_accounting.csv")
    assert set(tab["accountant"]) == {"prv", "rdp"} and len(tab) == 2


# ---------------------------------------------------------------- T-selection rules (fake run results)

def _fake_runner(scores, log):
    """scores: {(T, seed): ndcg}. Records every job it is asked to run."""
    def run(jobs):
        rows = []
        for j in jobs:
            log.append((j["T"], j["seed"]))
            assert j["evaluate_test"] is False and j["epsilon"] == 4
            row = {c: 0.0 for c in rb.STUDY_COLS}
            row.update(T=j["T"], seed=j["seed"], **{"ndcg@10": scores[(j["T"], j["seed"])]})
            rows.append(row)
        return pd.DataFrame(rows)
    return run


@pytest.fixture
def fake_sigma(monkeypatch):
    monkeypatch.setattr(rb, "sigma_for", lambda cfg, eps, T: 1.0)


def _scores(grid, extra=None, seeds=(123, 2026)):
    s = {(T, 42): v for T, v in grid.items()}
    for (T, seed), v in (extra or {}).items():
        s[(T, seed)] = v
    return s


def test_best_at_lower_edge_expands_once_to_50(iso, fake_sigma):
    log = []
    sc = _scores({100: 0.030, 250: 0.020, 500: 0.015, 1000: 0.010, 50: 0.040})   # 50 even better: still no 25
    _, sel, dec = rb.t_sweep(1, run_fn=_fake_runner(sc, log))
    assert [t for t, s in log if s == 42] == [100, 250, 500, 1000, 50]
    assert sel == 50 and "boundary expansion to T=50" in dec
    assert len(log) == 5                                       # one expansion only, no seed rule (gap 0.010)


def test_best_at_upper_edge_expands_once_to_1500(iso, fake_sigma):
    log = []
    sc = _scores({100: 0.010, 250: 0.012, 500: 0.015, 1000: 0.030, 1500: 0.020})
    _, sel, _ = rb.t_sweep(1, run_fn=_fake_runner(sc, log))
    assert (1500, 42) in log and sel == 1000 and len(log) == 5


def test_interior_best_has_no_boundary_expansion(iso, fake_sigma):
    log = []
    sc = _scores({100: 0.010, 250: 0.030, 500: 0.015, 1000: 0.012})
    _, sel, dec = rb.t_sweep(1, run_fn=_fake_runner(sc, log))
    assert sel == 250 and len(log) == 4 and "boundary" not in dec
    sel_csv = pd.read_csv(rb.RESULTS / "b2_t_selection.csv")
    assert int(sel_csv["selected_T"].iloc[0]) == 250


def test_top_two_within_margin_triggers_seed_rule_and_mean_selection(iso, fake_sigma):
    log = []
    sc = _scores({100: 0.010, 250: 0.0300, 500: 0.0285, 1000: 0.012},          # gap 0.0015 < 0.003
                 {(250, 123): 0.020, (250, 2026): 0.021, (500, 123): 0.030, (500, 2026): 0.031})
    df, sel, dec = rb.t_sweep(1, run_fn=_fake_runner(sc, log))
    assert sorted(x for x in log if x[1] != 42) == [(250, 123), (250, 2026), (500, 123), (500, 2026)]
    assert sel == 500 and "mean over seeds" in dec                # 500 wins on the 3-seed mean
    assert set(df[df.stage == "seed_rule"]["T"]) == {250, 500}


def test_top_two_gap_at_or_above_margin_skips_seed_rule(iso, fake_sigma):
    log = []
    sc = _scores({100: 0.010, 250: 0.0330, 500: 0.0300, 1000: 0.012})          # gap 0.003 (not < margin)
    _, sel, dec = rb.t_sweep(1, run_fn=_fake_runner(sc, log))
    assert sel == 250 and len(log) == 4 and "provisional single-seed" in dec
