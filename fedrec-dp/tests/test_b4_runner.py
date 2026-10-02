"""B4 runner gates and invariants — SYNTHETIC data only, all outputs in temporary directories."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

import experiments.run_b4 as rb4
import src.evaluate
import src.train_federated
from tests.test_b2_protocol import SYNTH, SYNTH_FP, research_outputs_untouched  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def iso4(tmp_path, monkeypatch):
    res, raw, ck = tmp_path / "results", tmp_path / "results" / "raw", tmp_path / "checkpoints"
    for d in (res, raw, ck):
        d.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "configs" / "b4.yaml"))
    cfg["model"]["dim"] = 8
    cfg["privacy"]["T"] = 3
    cfg["seeds"] = [42, 123]
    cfg["model"]["ranks"] = [2, 4]
    cfg["per_rank"] = {2: {"local_lr": 1.0, "clip_norm": 0.5, "server_lr": 1.0},
                       4: {"local_lr": 5.0, "clip_norm": 1.0, "server_lr": 1.0}}
    cfg["lr_c_search"].update(local_lr_grid=[0.5, 1.0], edge_lr_if_b3_selected=0.25, clip_grid=[0.5, 1.0],
                              b3_reference_config=str(tmp_path / "b3_test.yaml"))
    path = tmp_path / "b4_test.yaml"

    def write(**top):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        c.update(top)
        path.write_text(yaml.safe_dump(c))
        return c

    # B2-style accounting rows for T = 3 under B2's exact key (prv / rdp, q 0.1, delta 1e-5)
    pd.DataFrame([{"target_epsilon": e, "noise_multiplier": s, "epsilon": e * 0.995, "accountant": "prv",
                   "epsilon_cross_check": e * 1.1, "cross_check_accountant": "rdp", "delta": 1e-5, "q": 0.1, "T": 3}
                  for e, s in ((8, 0.4), (4, 0.6), (2, 0.9), (1, 1.5))]).to_csv(res / "b2_accounting.csv", index=False)
    b2 = yaml.safe_load(open(ROOT / "configs" / "b2.yaml"))
    b2["privacy"]["T"] = 3                                     # temp frozen-B2 copy matching the synthetic horizon
    b2path = tmp_path / "b2_test.yaml"
    b2path.write_text(yaml.safe_dump(b2))
    monkeypatch.setattr(rb4, "B2_CONFIG", str(b2path))
    b3 = yaml.safe_load(open(ROOT / "configs" / "b3.yaml"))
    b3.update(protocol_frozen=True, selected_lr={2: 0.25, 4: 1.0})            # rank 2 "selected the edge"
    b3path = tmp_path / "b3_test.yaml"
    b3path.write_text(yaml.safe_dump(b3))
    monkeypatch.setattr(rb4, "B3_CONFIG", str(b3path))
    monkeypatch.setattr(rb4, "RESULTS", res)
    monkeypatch.setattr(rb4, "RAW", raw)
    monkeypatch.setattr(rb4, "CHECKPOINTS", ck)
    monkeypatch.setattr(rb4, "CONFIG", str(path))
    monkeypatch.setattr(rb4, "load_all", lambda: (yaml.safe_load(path.read_text()), {"synthetic": True}, SYNTH, SYNTH_FP))
    write(protocol_frozen=False)
    return write


def _guard_evaluator(monkeypatch, allow_test):
    seen = []
    real = src.evaluate.evaluate

    def wrapped(score_fn, split, target, *a, **k):
        assert split is SYNTH, "B4 runner evaluated a non-synthetic split"
        if target == "test" and not allow_test:
            raise AssertionError("test evaluated in a validation-only phase")
        seen.append(target)
        return real(score_fn, split, target, *a, **k)

    for mod in (rb4, src.train_federated):
        monkeypatch.setattr(mod, "evaluate", wrapped)
    return seen


def test_sigma_is_reused_from_b2_rows_and_never_solved(iso4):
    cfg = rb4.load_all()[0]
    assert rb4.b2_sigma(cfg, 4) == 0.6 and rb4.b2_sigma(cfg, 1) == 1.5
    with pytest.raises(RuntimeError, match="reuse B2"):
        rb4.b2_sigma(cfg, 3)                                  # no B2 row for target eps 3 -> refuse, never solve


@pytest.mark.parametrize("key,val", [("T", 50), ("delta", 1e-6), ("accountant", "rdp"),
                                     ("cross_check_accountant", "prv"), ("solver_tolerance", 0.05),
                                     ("denominator", "realised"), ("target_epsilons", [8, 4])])
def test_b4_must_match_frozen_b2_accounting_before_sigma_reuse(iso4, key, val):
    cfg = rb4.load_all()[0]
    cfg["privacy"][key] = val
    with pytest.raises(RuntimeError, match="not compatible with frozen B2"):
        rb4.b2_sigma(cfg, 4)


def test_b4_q_must_match_b2(iso4):
    cfg = rb4.load_all()[0]
    cfg["federated"]["client_sampling_q"] = 0.2
    with pytest.raises(RuntimeError, match="client_sampling_q"):
        rb4.b2_sigma(cfg, 4)


def test_run_tag_identity_includes_rank_T_C_lrs_seed(iso4):
    cfg = rb4.load_all()[0]
    tags = {rb4.run_tag(rb4.run_cfg(cfg, j), j) for j in
            [rb4.job(2, 4, 42), rb4.job(4, 4, 42), rb4.job(2, None, 42), rb4.job(2, 4, 123),
             rb4.job(2, 4, 42, clip=1.0), rb4.job(2, 4, 42, server_lr=2.0), rb4.job(2, 4, 42, local_lr=5.0, kind="fixeddiag")]}
    assert len(tags) == 7
    t = rb4.run_tag(rb4.run_cfg(cfg, rb4.job(2, 4, 42)), rb4.job(2, 4, 42))
    assert t == "main_r2_eps4_T3_C0.5_lr1.0_slr1.0_seed42"


def test_validation_phase_never_tests_and_keeps_exact_T_checkpoint(iso4, monkeypatch):
    seen = _guard_evaluator(monkeypatch, allow_test=False)
    row = rb4.train_validate(rb4.job(2, 4, 42), keep_checkpoint=True)
    assert row["status"] == "ok" and "validation" in seen and "test" not in seen
    import torch
    ck = torch.load(rb4._ckpt_path(row["tag"]), weights_only=True)
    assert ck["round"] == 3 and ck["T"] == 3 and ck["rank"] == 2 and ck["noise_multiplier"] == 0.6
    assert ck["split_fingerprint"] == SYNTH_FP
    assert not list(rb4.RAW.glob("**/per_user_test_*"))


def test_test_scoring_refused_unless_frozen_and_rank_selected(iso4, monkeypatch):
    write = iso4
    with pytest.raises(RuntimeError, match="not frozen"):
        rb4.score_test(1)
    with pytest.raises(RuntimeError, match="not frozen"):
        rb4.final_train(1)
    write(protocol_frozen=True)
    with pytest.raises(RuntimeError, match="validation-selected ranks"):
        rb4.score_test(1)


def test_full_frozen_pipeline_on_synthetic_data(iso4, monkeypatch):
    write = iso4
    write(protocol_frozen=True)
    seen = _guard_evaluator(monkeypatch, allow_test=True)
    rb4.final_train(1)                                        # in-process so the evaluator spy observes every call
    val = pd.read_csv(rb4.RESULTS / "b4_final_validation.csv")
    # main: 2 seeds x 2 ranks x 5 levels; fixed diagnostic only for rank 4? rank 2 differs (lr 1.0, C 0.5) -> runs;
    # rank 4 has lr 5, C 1, eta_s 1 == fixed settings -> reused, not retrained
    assert (val.kind == "main").sum() == 20 and (val.kind == "fixeddiag").sum() == 2
    assert set(val[val.kind == "fixeddiag"]["rank"]) == {2}
    assert "test" not in seen
    rb4.select_rank()
    sel = pd.read_csv(rb4.RESULTS / "b4_rank_selection.csv")
    assert set(sel["privacy_level"]) == {"nodp", "eps8", "eps4", "eps2", "eps1"}
    rb4.score_test(1)
    res = pd.read_csv(rb4.RESULTS / "b4_final_results.csv")
    assert res["test_ndcg@10"].notna().sum() == (val.status == "ok").sum()
    assert seen.count("test") == (val.status == "ok").sum()   # each retained checkpoint scored on test exactly once


@pytest.mark.parametrize("corrupt", ["A_shape", "rank", "fingerprint", "q", "T", "clip", "local_lr", "server_lr",
                                     "sigma", "delta", "accountant", "cross_check", "epsilon", "epsilon_cross_check",
                                     "local_epochs", "l2_reg", "init_std", "dim_cfg", "denominator", "target_epsilon",
                                     "noise_seed", "nonfinite_P", "float64_B"])
def test_malformed_checkpoint_is_refused(iso4, corrupt):
    import torch
    write = iso4
    write(protocol_frozen=True)
    cfg = rb4.load_all()[0]
    j = rb4.job(2, 4, 42)
    row = rb4.train_validate(j, keep_checkpoint=True)
    ck = torch.load(rb4._ckpt_path(row["tag"]), weights_only=True)
    rb4.validate_checkpoint(ck, cfg, j, SYNTH, SYNTH_FP)               # the untouched checkpoint is accepted
    bad = dict(ck)
    bad["config"] = yaml.safe_load(yaml.safe_dump(ck["config"]))
    if corrupt == "A_shape":
        bad["A"] = torch.zeros(SYNTH["n_items"], 3)
    elif corrupt == "rank":
        bad["rank"] = 4
    elif corrupt == "fingerprint":
        bad["split_fingerprint"] = "0" * 64
    elif corrupt == "q":
        bad["q"] = 0.2
    elif corrupt == "T":
        bad["T"] = 50
    elif corrupt == "clip":
        bad["clip_norm"] = 9.0
    elif corrupt == "local_lr":
        bad["config"]["federated"]["local_lr"] = 5.0
    elif corrupt == "server_lr":
        bad["config"]["federated"]["server_lr"] = 2.0
    elif corrupt == "sigma":
        bad["noise_multiplier"] = 0.123
    elif corrupt == "delta":
        bad["delta"] = 1e-6
    elif corrupt == "accountant":
        bad["accountant"] = "rdp"
    elif corrupt == "cross_check":
        bad["cross_check_accountant"] = "prv"
    elif corrupt == "epsilon":
        bad["epsilon"] = ck["epsilon"] * 0.5
    elif corrupt == "epsilon_cross_check":
        bad["epsilon_cross_check"] = ck["epsilon_cross_check"] + 1.0
    elif corrupt == "local_epochs":
        bad["config"]["federated"]["local_epochs"] = 5
    elif corrupt == "l2_reg":
        bad["config"]["federated"]["l2_reg"] = 0.0
    elif corrupt == "init_std":
        bad["config"]["model"]["init_std"] = 0.1
    elif corrupt == "dim_cfg":
        bad["config"]["model"]["dim"] = 16
    elif corrupt == "denominator":
        bad["denominator"] = "realised"
    elif corrupt == "target_epsilon":
        bad["config"]["privacy"]["target_epsilon"] = 8.0
    elif corrupt == "noise_seed":
        bad["noise_seed"] = 7
    elif corrupt == "nonfinite_P":
        bad["P"] = ck["P"].clone(); bad["P"][0, 0] = float("inf")
    elif corrupt == "float64_B":
        bad["B"] = ck["B"].double()
    with pytest.raises(ValueError, match="does not match the frozen job"):
        rb4.validate_checkpoint(bad, cfg, j, SYNTH, SYNTH_FP)


def test_inventory_must_be_complete_unique_and_valid(iso4):
    write = iso4
    cfg = write(protocol_frozen=True)
    rb4.final_train(1)
    df = pd.read_csv(rb4.RESULTS / "b4_final_validation.csv")
    rb4.check_main_inventory(cfg, df)                                    # complete: accepted
    with pytest.raises(RuntimeError, match="missing"):
        rb4.check_main_inventory(cfg, df.drop(df[df.kind == "main"].index[0]))
    with pytest.raises(RuntimeError, match="duplicates"):
        rb4.check_main_inventory(cfg, pd.concat([df, df[df.kind == "main"].iloc[[0]]]))
    bad = df.copy(); bad.loc[bad[bad.kind == "main"].index[0], "status"] = "training diverged at round 2"
    with pytest.raises(RuntimeError, match="failed main jobs"):
        rb4.check_main_inventory(cfg, bad)
    bad.to_csv(rb4.RESULTS / "b4_final_validation.csv", index=False)
    with pytest.raises(RuntimeError, match="failed main jobs"):
        rb4.select_rank()                                                # never selects from partial cells


def test_second_scoring_invocation_makes_zero_test_calls_and_never_overwrites(iso4, monkeypatch):
    write = iso4
    write(protocol_frozen=True)
    rb4.final_train(1)
    rb4.select_rank()
    seen = _guard_evaluator(monkeypatch, allow_test=True)
    rb4.score_test(1)
    first = seen.count("test")
    assert first == (pd.read_csv(rb4.RESULTS / "b4_final_validation.csv").status == "ok").sum()
    rb4.score_test(1)                                                    # resumed: everything verified & reused
    assert seen.count("test") == first
    tag = pd.read_csv(rb4.RESULTS / "b4_final_validation.csv")["tag"].iloc[0]
    (rb4.RAW / "b4_runs" / f"test_result_{tag}.csv").unlink()            # result lost, per-user file still present
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        rb4.score_test(1)


def test_cached_result_cannot_bypass_checkpoint_validation(iso4, monkeypatch):
    """A consistent-looking cache (hashes updated) over a checkpoint with a wrong config must still be refused."""
    import torch
    write = iso4
    write(protocol_frozen=True)
    rb4.final_train(1)
    rb4.select_rank()
    rb4.score_test(1)
    row = pd.read_csv(rb4.RESULTS / "b4_final_validation.csv").iloc[0]
    path = rb4._ckpt_path(row["tag"])
    ck = torch.load(path, weights_only=True)
    ck["config"]["federated"]["local_lr"] = 99.0                    # scientific mismatch
    torch.save(ck, path)
    new_sha = rb4.file_sha256(path)
    for f in (rb4.RAW / "b4_runs" / f"test_result_{row['tag']}.csv", rb4.RAW / "b4_runs" / f"result_{row['tag']}.csv"):
        d = pd.read_csv(f); d["checkpoint_sha256"] = new_sha; d.to_csv(f, index=False)
    v = pd.read_csv(rb4.RESULTS / "b4_final_validation.csv"); v.loc[0, "checkpoint_sha256"] = new_sha
    v.to_csv(rb4.RESULTS / "b4_final_validation.csv", index=False)
    seen = _guard_evaluator(monkeypatch, allow_test=True)
    with pytest.raises(ValueError, match="does not match the frozen job"):
        rb4.score_test(1)
    assert "test" not in seen


def test_amended_grid_is_exactly_existing_values_with_edge_only_where_b3_selected_it():
    cfg = yaml.safe_load(open(ROOT / "configs" / "b4.yaml"))
    b3 = {"selected_lr": {4: 1.25, 8: 1.25, 16: 1.25, 32: 0.625}}
    for r, n in ((4, 15), (8, 15), (16, 15), (32, 18)):
        g = rb4.lr_c_grid(cfg, b3, r)
        assert len(g) == n == len(set(g))
        assert {lr for lr, _ in g} <= {0.625, 1.25, 2.5, 5.0, 10.0, 20.0} and {C for _, C in g} == {1.0, 1.5, 2.4}
        assert (0.625 in {lr for lr, _ in g}) == (r == 32)


def test_lr_c_search_refused_unless_b3_frozen(iso4):
    b3 = yaml.safe_load(open(rb4.B3_CONFIG))
    b3["protocol_frozen"] = False
    Path(rb4.B3_CONFIG).write_text(yaml.safe_dump(b3))
    with pytest.raises(RuntimeError, match="B3 is not frozen"):
        rb4.lr_c_search(1)


def test_lr_c_tie_rules_and_failures_never_selected():
    g = pd.DataFrame({"status": ["ok", "ok", "ok", "ok", "diverged"],
                      "val_ndcg@10": [0.05, 0.05, 0.05, 0.04, float("nan")],
                      "privacy_clip_norm": [1.5, 1.0, 1.0, 1.0, 1.0], "local_lr": [5.0, 10.0, 2.5, 5.0, 20.0]})
    b = rb4.select_lr_c(g)                                      # smaller C, then |log(lr/5)| tie 2.5 vs 10 -> smaller lr
    assert (b["privacy_clip_norm"], b["local_lr"]) == (1.0, 2.5)
    g.loc[1, "val_ndcg@10"] = 0.06
    assert rb4.select_lr_c(g)["local_lr"] == 10.0
    assert rb4.select_lr_c(g.assign(status="diverged")) is None
    s = pd.DataFrame({"status": ["ok"] * 3, "val_ndcg@10": [0.05, 0.05, 0.05], "server_lr": [0.5, 1.0, 2.0]})
    assert rb4.select_slr(s)["server_lr"] == 1.0
    assert rb4.select_slr(s.iloc[[0, 2]])["server_lr"] == 0.5


def test_lr_c_then_slr_search_validation_only_and_eta1_cell_reused(iso4, monkeypatch):
    seen = _guard_evaluator(monkeypatch, allow_test=False)
    calls = []
    real = rb4.train_validate
    monkeypatch.setattr(rb4, "train_validate", lambda j, keep_checkpoint=False: (calls.append(j), real(j, keep_checkpoint))[1])
    rb4.lr_c_search(1)
    lrc = pd.read_csv(rb4.RESULTS / "b4_lr_c_search.csv")
    assert len(lrc) == 6 + 4 and len(calls) == 10                # rank 2: 3 lr (edge) x 2 C; rank 4: 2 x 2
    assert set(lrc[lrc["rank"] == 2]["local_lr"]) == {0.25, 0.5, 1.0}
    sel = pd.read_csv(rb4.RESULTS / "b4_lr_c_selection.csv")
    with pytest.raises(RuntimeError, match="per_rank lr/C"):
        rb4.slr_search(1)                                        # config not yet updated from the selection
    cfg = yaml.safe_load(Path(rb4.CONFIG).read_text())
    cfg["per_rank"] = {int(r["rank"]): {"local_lr": float(r["local_lr"]), "clip_norm": float(r["clip_norm"])}
                       for _, r in sel.iterrows()}
    Path(rb4.CONFIG).write_text(yaml.safe_dump(cfg))
    calls.clear()
    rb4.slr_search(1)
    slr = pd.read_csv(rb4.RESULTS / "b4_slr_search.csv")
    assert len(slr) == 6 and len(calls) == 4                     # eta 0.5 / 2 trained; eta 1 reused per rank
    assert (slr["reused_from"] == "lrcsearch").sum() == 2
    assert "test" not in seen
