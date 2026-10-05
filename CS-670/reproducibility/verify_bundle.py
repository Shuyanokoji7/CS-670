"""Verify report provenance, numerical summaries and optional research-archive integrity.

Reads saved evidence only. It never trains or scores a model. --write-audit
writes a machine-readable verification record after all requested checks pass.
Archive verification requires the sibling fedrec-dp checkout and local artifacts.
"""
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import unquote
import argparse
import ast
import csv
import hashlib
import json
import re
import numpy as np
import pandas as pd

BUNDLE = Path(__file__).resolve().parents[1]
ARCHIVE = BUNDLE.parent / "fedrec-dp"
CACHE = {}


def sha(path):
    path = Path(path).resolve()
    if path not in CACHE:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        CACHE[path] = h.hexdigest()
    return CACHE[path]


def manifest(path):
    records = Path(path).read_text().splitlines()
    for line in records:
        expected, relative = line.split("  ", 1)
        assert sha(Path(path).parent / relative) == expected, relative
    return len(records)


def pilot_check(name, expected_probes, expected_seed_rows, expected_summary_rows):
    root = BUNDLE / "results" / name
    inv = pd.read_csv(root / "inventory.csv")
    seed = pd.read_csv(root / "by_seed.csv")
    summary = pd.read_csv(root / "summary.csv")
    assert len(inv) == expected_probes and inv.tag.is_unique
    assert set(inv.status) == {"ok"}
    assert len(seed) == expected_seed_rows and len(summary) == expected_summary_rows
    assert set(seed.seed) == {7, 42, 99, 123, 2026}
    keys = ["unit", "level", "alpha", "phase", "lag", "branch"]
    assert not seed.duplicated(keys + ["seed"]).any()
    assert seed.groupby(keys).size().eq(5).all()
    numerical = [c for c in seed.select_dtypes(include="number") if c not in keys + ["seed"]]
    assert np.isfinite(seed[numerical].to_numpy()).all()
    recalculated = seed.groupby(keys)[numerical].agg(["mean", "median", "std", "min", "max"])
    recalculated.columns = ["_".join(c) for c in recalculated.columns]
    expected = summary.set_index(keys).sort_index()
    assert recalculated.index.equals(expected.index)
    assert np.allclose(recalculated.to_numpy(), expected[recalculated.columns].to_numpy(),
                       rtol=2e-8, atol=1e-9), name
    churn = seed[[c for c in seed if c.startswith("top10_churn_")]]
    assert ((churn >= 0) & (churn <= 1)).all().all()
    assert (seed.users_exposed + seed.users_unexposed == 942).all()
    assert (seed.users_low + seed.users_medium + seed.users_high == 942).all()
    reset = seed[(seed.phase == "after_reset") & (seed.lag == 0)]
    assert reset[reset.branch == "shared_reset"].q_difference_fro.eq(0).all()
    assert reset[reset.branch == "local_reset"].p_difference_fro.eq(0).all()
    terms = seed.direct_score_energy + seed.local_score_energy + seed.interaction_score_energy
    terms += 2 * (seed.direct_local_inner + seed.direct_interaction_inner + seed.local_interaction_inner)
    scale = (seed.direct_score_energy + seed.local_score_energy + seed.interaction_score_energy).clip(lower=1)
    assert ((terms - seed.score_difference_energy).abs() / scale < 2e-8).all()
    checks = json.loads((root / "synthetic_checks.json").read_text())["records"]
    assert all(x["passed"] for x in checks)
    return {"probes": len(inv), "failures": 0, "seed_rows": len(seed),
            "summary_rows": len(summary), "synthetic_cases_passed": len(checks),
            "summaries_recomputed_from_seed_tables": True, "reset_and_energy_identities": True}


def archive_check(provenance):
    report = {}
    protected = json.loads((ARCHIVE / "results/extensions/effective_noise_v1/audit/protected_initial.json").read_text())["files"]
    for path, expected in protected.items():
        assert sha(ARCHIVE / path) == expected, path
    report["historical_files_unchanged"] = len(protected)
    e1 = ARCHIVE / "results/extensions/effective_noise_v1"
    report["e1_manifests_verified"] = {
        str(p.relative_to(ARCHIVE)): manifest(p) for p in sorted(e1.rglob("MANIFEST.sha256"))}
    for row in provenance:
        assert sha(BUNDLE.parent / row["source"]) == row["sha256"], row["source"]
    freeze = json.loads((ARCHIVE / "results/extensions/noise_memory_v1/freeze.json").read_text())
    for path, expected in freeze["sources"].items():
        assert sha(ARCHIVE / path) == expected, path
    assert sha(e1 / "MANIFEST.sha256") == freeze["input_manifest_sha256"]
    assert sha(ARCHIVE / "data/processed/ml-100k/train.csv") == freeze["train_sha256"]
    aligned = json.loads((ARCHIVE / "results/extensions/noise_memory_aligned_v1/freeze.json").read_text())
    for key, path in [("code_sha256", "experiments/probe_noise_memory_aligned.py"),
                      ("protocol_sha256", "docs/extensions/NOISE_MEMORY_ALIGNMENT_PROTOCOL.md"),
                      ("e2_code_sha256", "experiments/probe_noise_memory.py")]:
        assert sha(ARCHIVE / path) == aligned[key]
    final = {x["checkpoint"]: x for x in json.loads((e1 / "final_records.json").read_text())}
    probe_counts, observations, unique_checkpoints = {}, 0, set()
    alignment_error = 0.
    for name, count in [("noise_memory_v1", 120), ("noise_memory_aligned_v1", 40)]:
        root = ARCHIVE / "results/extensions" / name
        paths = sorted((root / "runs").glob("*.json"))
        assert len(paths) == count
        raw = []
        for path in paths:
            run = json.loads(path.read_text())
            assert run["status"] == "ok" and len(run["observations"]) == 14
            checkpoint = run["checkpoint"]
            expected = final[checkpoint]["artifact_hashes"][checkpoint]
            assert sha(ARCHIVE / checkpoint) == run["checkpoint_sha256"] == expected
            unique_checkpoints.add(checkpoint)
            raw.extend({**{k: run[k] for k in ("unit", "level", "seed", "replicate", "alpha")}, **r}
                       for r in run["observations"])
            alignment_error = max(alignment_error, max(r.get("alignment_product_relative_error_max", 0)
                                                       for r in run["observations"]))
        data = pd.DataFrame(raw)
        keys = ["unit", "level", "alpha", "phase", "lag", "branch", "seed"]
        columns = [c for c in data.select_dtypes(include="number") if c not in keys + ["replicate"]]
        assert data.groupby(keys).size().eq(2).all()
        derived = data.groupby(keys)[columns].mean().sort_index()
        saved = pd.read_csv(root / "by_seed.csv").set_index(keys).sort_index()
        assert derived.index.equals(saved.index)
        assert np.allclose(derived.to_numpy(), saved[columns].to_numpy(), rtol=2e-10, atol=1e-10)
        probe_counts[name] = count
        observations += len(raw)
        if (root / "MANIFEST.sha256").exists():
            report.setdefault("new_manifests_verified", {})[name] = manifest(root / "MANIFEST.sha256")
    assert len(unique_checkpoints) == 30
    report.update(probes_verified=probe_counts, original_observations=observations,
                  checkpoint_hashes_verified=len(unique_checkpoints), frozen_sources_unchanged=True,
                  source_copies_match=True, seed_averages_recomputed_from_original_probes=True,
                  alignment_product_relative_error_max=alignment_error)
    for name in ['noise_replication_v1','ranking_tails_v1']:
        root=ARCHIVE/'results/extensions'/name
        freeze=json.loads((root/'freeze.json').read_text())
        for path,expected in freeze['sources'].items():
            assert sha(ARCHIVE/path)==expected, path
        report.setdefault('new_manifests_verified',{})[name]=manifest(root/'MANIFEST.sha256')
    root=ARCHIVE/'results/extensions/noise_replication_v1'
    dataset=json.loads((root/'dataset.json').read_text())
    data_root=ARCHIVE/'data/extensions/ml-1m_noise_v1'
    for path,expected in {**dataset['raw_sha256'],**dataset['split_sha256']}.items():
        assert sha(data_root/path)==expected
    training=json.loads((root/'training_records.json').read_text())
    assert len(training)==30 and all(r['status']=='ok' and r['round']==50 for r in training)
    for r in training:
        assert sha(ARCHIVE/r['checkpoint'])==r['checkpoint_sha256']
    endpoints=[]
    for path in (root/'probes').glob('*.json'):
        r=json.loads(path.read_text())
        assert r['status']=='ok' and len(r['observations'])==14
        assert sha(ARCHIVE/r['checkpoint'])==r['checkpoint_sha256']
        for row in r['observations']:
            if row['phase']=='after_reset' and row['branch']=='shared_reset' and row['lag']==10:
                endpoints.append({**{k:r[k] for k in ['dataset','unit','level','balancing','coupling','seed']},**row})
    keys=['dataset','unit','level','balancing','coupling','seed']
    assert len(endpoints)==200
    calculated=pd.DataFrame(endpoints).groupby(keys).mean(numeric_only=True).sort_index()
    saved=pd.read_csv(BUNDLE/'results/noise_replication/endpoint_by_seed.csv').set_index(keys).sort_index()
    assert calculated.index.equals(saved.index)
    assert np.allclose(calculated,saved[calculated.columns],rtol=2e-8,atol=1e-10)
    pairs=[]
    for path in (ARCHIVE/'results/extensions/ranking_tails_v1/states').glob('*.json'):
        r=json.loads(path.read_text())
        assert r['status']=='ok' and len(r['observations'])==64 and r['flagged_pairs']==0
        assert sha(ARCHIVE/r['checkpoint'])==r['checkpoint_sha256']
        pairs.extend({**{k:r[k] for k in ['dataset','unit','level','seed']},**row} for row in r['observations'])
    assert len(pairs)==3200 and all(r['status']=='ok' for r in pairs)
    keys=['dataset','unit','level','pair','seed']
    calculated=pd.DataFrame(pairs).groupby(keys).mean(numeric_only=True).sort_index()
    saved=pd.read_csv(BUNDLE/'results/ranking_tails/by_seed.csv').set_index(keys).sort_index()
    assert calculated.index.equals(saved.index)
    assert np.allclose(calculated[saved.columns],saved,rtol=2e-8,atol=1e-10)
    report['e3_e4_original_sources_and_checkpoints_verified']=True
    report['e3_endpoint_recomputed_from_probes']=200
    report['e4_seed_summaries_recomputed_from_pairs']=3200
    return report


def followup_check():
    root=BUNDLE/'results/noise_replication'
    seed=pd.read_csv(root/'endpoint_by_seed.csv')
    assert len(seed)==100 and set(seed.seed)=={7,42,99,123,2026}
    assert seed.groupby(['dataset','unit','level','balancing','coupling']).size().eq(5).all()
    assert seed.phase.eq('after_reset').all() and seed.branch.eq('shared_reset').all() and seed.lag.eq(10).all()
    assert (seed.users_exposed+seed.users_unexposed).eq(128).all()
    columns=['top10_churn','score_rms','score_ratio_to_pulse','validation_ndcg_difference']
    keys=['dataset','unit','level','balancing','coupling']
    calculated=seed.groupby(keys)[columns].agg(['mean','std','min','max'])
    calculated.columns=['_'.join(c) for c in calculated.columns]
    saved=pd.read_csv(root/'endpoint_summary.csv').set_index(keys).sort_index()
    assert calculated.index.equals(saved.index)
    assert np.allclose(calculated,saved[calculated.columns],rtol=2e-8,atol=1e-10)
    unbalanced=seed[(seed.unit=='two_r8')&(seed.balancing=='none')]
    wide=unbalanced.pivot(index=['dataset','level','seed'],columns='coupling',values='top10_churn')
    assert np.array_equal(wide.raw,wide.aligned)
    val=pd.read_csv(root/'validation_by_seed.csv')
    assert len(val)==40 and val.groupby(['unit','level']).size().eq(5).all()
    calculated=val.groupby(['unit','level']).validation_ndcg.agg(['mean','std','min','max'])
    saved=pd.read_csv(root/'validation_summary.csv').set_index(['unit','level']).sort_index()
    assert calculated.index.equals(saved.index) and np.allclose(calculated,saved,rtol=2e-8,atol=1e-12)
    for level in ['eps1','eps2']:
        block=saved.xs(level,level='level')
        assert (block.drop('dp_popularity')['mean']<block.loc['dp_popularity','mean']).all()
    root=BUNDLE/'results/ranking_tails'
    seed=pd.read_csv(root/'by_seed.csv')
    assert len(seed)==200 and seed.groupby(['dataset','unit','level','pair']).size().eq(5).all()
    assert seed.error_matched_over_01.eq(0).all()
    keys=['dataset','unit','level','pair']
    columns=[c for c in seed.select_dtypes(include='number') if c!='seed']
    calculated=seed.groupby(keys)[columns].agg(['mean','std','min','max'])
    calculated.columns=['_'.join(c) for c in calculated.columns]
    saved=pd.read_csv(root/'summary.csv').set_index(keys).sort_index()
    assert calculated.index.equals(saved.index) and np.allclose(calculated,saved[calculated.columns],rtol=2e-8,atol=1e-10)
    audit=json.loads((root/'audit_summary.json').read_text())
    assert audit['states']==50 and audit['pairs']==3200 and audit['flagged_pairs']==0
    assert not audit['advancement_gate_passed'] and audit['max_absolute_error_matched']<8e-6
    for name in ['noise_replication','ranking_tails']:
        report=json.loads((BUNDLE/'results'/name/'completion_audit.json').read_text())
        assert report['all_checks_passed'] and report['e3']['failures']==0 and report['e4']['flagged_pairs']==0
    return dict(e3_endpoint_rows=100,e3_validation_rows=40,e4_seed_rows=200,
                all_summaries_recomputed=True,no_balancing_churn_equal_across_couplings=True,
                e3_collaborative_means_below_dp_popularity=True,e4_advancement_gate_passed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--write-audit", action="store_true")
    args = parser.parse_args()
    provenance = list(csv.DictReader((BUNDLE / "reproducibility/provenance.csv").open()))
    for row in provenance:
        assert sha(BUNDLE / row["destination"]) == row["sha256"], row["destination"]
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "copied_files_verified": len(provenance), "held_out_scoring": False,
              "training_executed": False, "scope": "Saved-evidence integrity and summary checks"}
    report["e2"] = pilot_check("noise_memory", 120, 840, 168)
    report["e2b"] = pilot_check("noise_memory_aligned", 40, 280, 56)
    report['e3_e4']=followup_check()
    e1 = pd.read_csv(BUNDLE / "results/effective_noise/contrasts.csv")
    primary = e1[e1.family == "primary_fixed_vs_two"]
    assert len(primary) == 16
    assert (primary.sim_lo > 0).sum() == 0 and (primary.sim_hi < 0).sum() == 8
    old = pd.read_csv(BUNDLE / "results/baselines/paired_contrasts.csv")
    assert (old[old.family == "B3_vs_B1"].sim_hi < 0).sum() == 4
    assert (old[old.family == "B4_vs_B2"].sim_lo > 0).sum() == 4
    means = pd.read_csv(BUNDLE / "results/effective_noise/test_means.csv").set_index("model")
    ranks = pd.read_csv(BUNDLE / "results/effective_noise/rank_selection.csv")
    selected = pd.read_csv(BUNDLE / "results/effective_noise/selected_rank_utility.csv")
    assert len(selected) == 16
    for row in selected.itertuples():
        assert np.isclose(row.ndcg_mean, means.loc[row.model, "ndcg_mean"], rtol=0, atol=1e-12)
        assert np.isclose(row.ndcg_seed_sd, means.loc[row.model, "ndcg_seed_sd"], rtol=0, atol=1e-12)
        if row.method in ("Two", "FixedB"):
            method = {"Two": "two", "FixedB": "fixed"}[row.method]
            rank = ranks[(ranks.method == method) & (ranks.level == f"eps{row.epsilon_target}")].iloc[0]["rank"]
            assert row.model == f"{method}_r{int(rank)}_eps{row.epsilon_target}"
    for level in ("eps1", "eps2", "eps4"):
        collaborative = means[means.index.str.startswith(("full_", "fixed_", "two_")) & means.index.str.endswith(level)]
        assert len(collaborative) == 9
        assert (collaborative.ndcg_mean < means.loc[f"pop_{level}", "ndcg_mean"]).all()
    report["primary_contrast_counts_and_selected_table_verified"] = True
    report["all_collaborative_e1_means_below_dp_popularity_eps_le_4"] = True
    null = json.loads((BUNDLE / "results/noise_memory/coupling_null_example.json").read_text())
    assert null["checks_passed"] and null["aligned_q_difference_max_abs"] == 0
    assert null["code_sha256"] == sha(BUNDLE / "reproducibility/coupling_null_example.py")
    report["synthetic_null_identity_verified"] = True
    broken, checked = [], 0
    for path in BUNDLE.rglob("*.md"):
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http:", "https:", "mailto:", "#")):
                continue
            destination = unquote(target.split("#", 1)[0].strip("<>"))
            checked += 1
            if not (path.parent / destination).exists():
                broken.append((str(path.relative_to(BUNDLE)), target))
    assert not broken, broken
    report["local_document_links_verified"] = checked
    keys = re.findall(r"@\w+\{([^,]+),", (BUNDLE / "references/references.bib").read_text())
    assert len(keys) == len(set(keys)) == 28
    report["unique_bibliography_entries"] = len(keys)
    for path in BUNDLE.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
    for stem in ("01_nonprivate_controls", "02_historical_private_utility", "03_effective_noise_utility",
                 "04_noise_and_clipping", "05_local_state_memory", "06_coupling_control",
                 "07_balancing_replication", "08_ml1m_validation", "09_ranking_tail_screen"):
        for suffix in ("png", "pdf", "svg"):
            assert (BUNDLE / "figures" / f"{stem}.{suffix}").stat().st_size > 1000
    report["figure_exports_present"] = 27
    if args.archive:
        report["archive"] = archive_check(provenance)
    report["all_requested_checks_passed"] = True
    if args.write_audit:
        (BUNDLE / "reproducibility/audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
