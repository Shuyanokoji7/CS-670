"""B3 runner gates — SYNTHETIC data only, outputs in temporary directories."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

import experiments.run_b3 as rb3
import src.evaluate
import src.train_federated
from tests.test_b2_protocol import SYNTH, SYNTH_FP, research_outputs_untouched  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def iso3(tmp_path, monkeypatch):
    res, raw, ck = tmp_path / "results", tmp_path / "results" / "raw", tmp_path / "checkpoints"
    for d in (res, raw, ck):
        d.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "configs" / "b3.yaml"))
    cfg["model"].update(dim=8, ranks=[3])
    cfg["federated"].update(max_rounds=60, eval_every=5, patience_evals=2)
    cfg["selected_lr"] = {3: 1.0}
    cfg["seeds"] = [42, 7]
    path = tmp_path / "b3_test.yaml"

    def write(**top):
        c = yaml.safe_load(yaml.safe_dump(cfg))
        for key, val in top.items():
            if isinstance(val, dict):
                c[key].update(val)
            else:
                c[key] = val
        path.write_text(yaml.safe_dump(c))

    monkeypatch.setattr(rb3, "RESULTS", res)
    monkeypatch.setattr(rb3, "RAW", raw)
    monkeypatch.setattr(rb3, "CHECKPOINTS", ck)
    monkeypatch.setattr(rb3, "CONFIG", str(path))
    monkeypatch.setattr(rb3, "load_all", lambda: (yaml.safe_load(path.read_text()), {"synthetic": True}, SYNTH, SYNTH_FP))
    write(protocol_frozen=False)
    return write


def _spy(monkeypatch):
    seen = []
    real = src.evaluate.evaluate

    def spy(score_fn, split, target, *a, **k):
        assert split is SYNTH, "B3 runner evaluated a non-synthetic split"
        seen.append(target)
        return real(score_fn, split, target, *a, **k)

    for mod in (rb3, src.train_federated):
        monkeypatch.setattr(mod, "evaluate", spy)
    return seen


def test_final_phases_and_b1_supplemental_refused_unless_frozen(iso3):
    for fn in (lambda: rb3._final_train_one((3, 42)), lambda: rb3.final_score(1), rb3.b1_supplemental):
        with pytest.raises(RuntimeError, match="not frozen"):
            fn()


def test_train_is_validation_only_score_once_resumable_and_no_overwrite(iso3, monkeypatch):
    write = iso3
    write(protocol_frozen=True)
    seen = _spy(monkeypatch)
    rb3.final_train(1)
    assert "test" not in seen and seen
    val = pd.read_csv(rb3.RESULTS / "b3_final_validation.csv")
    assert {"best_round", "rounds_run", "comm_bytes_to_best_round", "comm_bytes_total_run", "checkpoint_sha256"} <= set(val.columns)
    assert (val["comm_bytes_to_best_round"] <= val["comm_bytes_total_run"]).all()
    rb3.final_score(1)
    assert seen.count("test") == 2                                            # 2 seeds, scored once each
    rb3.final_score(1)
    assert seen.count("test") == 2                                            # resumed: zero new test calls
    rb3.final_train(1)
    assert seen.count("test") == 2                                            # verified checkpoints not retrained
    t = val["tag"].iloc[0]
    (rb3.RAW / "b3_runs" / f"test_result_{t}.csv").unlink()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        rb3.final_score(1)


def test_cap_hit_or_incomplete_inventory_blocks_scoring(iso3, monkeypatch):
    write = iso3
    write(protocol_frozen=True)
    _spy(monkeypatch)
    rb3.final_train(1)
    df = pd.read_csv(rb3.RESULTS / "b3_final_validation.csv")
    cfg = rb3.load_all()[0]
    capped = df.copy(); capped.loc[0, "stopped_by"] = "max_rounds"
    with pytest.raises(RuntimeError, match="not converged"):
        rb3.check_b3_inventory(cfg, capped)
    with pytest.raises(RuntimeError, match="missing"):
        rb3.check_b3_inventory(cfg, df.iloc[[0]])


def test_validated_b3_checkpoint_load(iso3, monkeypatch):
    write = iso3
    write(protocol_frozen=True)
    _spy(monkeypatch)
    rb3._final_train_one((3, 42))
    path = rb3.CHECKPOINTS / "b3" / "b3_r3_lr1.0_seed42.pt"
    sim, ck = rb3.load_b3_checkpoint(path, SYNTH, 3, 1.0, 42)
    assert sim.A.shape == (SYNTH["n_items"], 3) and ck["rank"] == 3
    for bad in ((4, 1.0, 42), (3, 2.5, 42), (3, 1.0, 7)):
        with pytest.raises(ValueError, match="does not match"):
            rb3.load_b3_checkpoint(path, SYNTH, *bad)


def test_unverifiable_artifacts_block_resume_and_cache(iso3, monkeypatch):
    import torch
    write = iso3
    write(protocol_frozen=True)
    seen = _spy(monkeypatch)
    rb3.final_train(1)
    rb3.final_score(1)
    path = rb3.CHECKPOINTS / "b3" / "b3_r3_lr1.0_seed42.pt"
    ck = torch.load(path, weights_only=True)
    ck["config"]["federated"]["l2_reg"] = 0.5                        # stored config no longer the frozen job's
    torch.save(ck, path)
    new = rb3._sha(path)
    rec = rb3.RAW / "b3_runs" / "train_result_r3_lr1.0_seed42.csv"
    d = pd.read_csv(rec); d["checkpoint_sha256"] = new; d.to_csv(rec, index=False)
    with pytest.raises(RuntimeError, match="stored config differs"):
        rb3.final_train(1)                                          # resume refuses, no silent retraining
    n = seen.count("test")
    v = pd.read_csv(rb3.RESULTS / "b3_final_validation.csv"); v.loc[v.seed == 42, "checkpoint_sha256"] = new
    v.to_csv(rb3.RESULTS / "b3_final_validation.csv", index=False)
    tr = rb3.RAW / "b3_runs" / "test_result_r3_lr1.0_seed42.csv"
    d = pd.read_csv(tr); d["checkpoint_sha256"] = new; d.to_csv(tr, index=False)
    with pytest.raises(ValueError, match="stored config differs"):
        rb3.final_score(1)                                          # cache cannot bypass validation
    assert seen.count("test") == n


def test_resume_ignores_other_ranks_selection_but_not_own_scientific_fields(iso3, monkeypatch):
    import torch
    write = iso3
    write(protocol_frozen=True)
    seen = _spy(monkeypatch)
    rb3.final_train(1)
    n = len(seen)
    write(protocol_frozen=True, selected_lr={3: 1.0, 9: 2.5})               # another rank's correction
    rb3.final_train(1)
    assert len(seen) == n                                                    # verified reuse, no retraining
    path = rb3.CHECKPOINTS / "b3" / "b3_r3_lr1.0_seed42.pt"
    ck = torch.load(path, weights_only=True)
    ck["config"]["federated"]["patience_evals"] = 7
    torch.save(ck, path)
    rec = rb3.RAW / "b3_runs" / "train_result_r3_lr1.0_seed42.csv"
    d = pd.read_csv(rec); d["checkpoint_sha256"] = rb3._sha(path); d.to_csv(rec, index=False)
    with pytest.raises(RuntimeError, match="patience_evals"):
        rb3.final_train(1)
