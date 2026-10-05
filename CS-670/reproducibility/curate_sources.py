"""Refresh the explicit report evidence whitelist from the sibling research archive.

No experiment is run. No archive file is modified. Run from any directory with
the research environment's Python. Superseded runs, checkpoints and bulk logs are
deliberately outside this report bundle; see CURATION_POLICY.md.
"""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parents[1]
SOURCE = BUNDLE.parent / "fedrec-dp"

FILES = {
    "results/weighting_control_summary.csv": ("results/baselines/centralized_federated_3seeds.csv", "Matched three-seed B0/B0-UW/B1 comparison"),
    "results/weighting_control_paired.csv": ("results/baselines/weighting_paired.csv", "Intervals prevent overstating weighting and federation effects"),
    "results/b34_test_means.csv": ("results/baselines/final_5seed_means.csv", "Canonical B1-B4 five-seed results including labeled diagnostics"),
    "results/b34_contrasts.csv": ("results/baselines/paired_contrasts.csv", "Adjusted primary comparisons plus labeled secondary results"),
    "results/b34_communication.csv": ("results/baselines/communication.csv", "Payload and total-volume distinction"),
    "results/b4_rank_selection.csv": ("results/baselines/validation_rank_selection.csv", "Protect validation-selected ranks from test-best substitution"),
    "results/b2_accounting.csv": ("results/baselines/privacy_accounting.csv", "Exact final privacy calibration"),
    "results/popularity_baseline.csv": ("results/baselines/nonprivate_popularity.csv", "Historical nonprivate point reference"),
    "results/random_baseline.csv": ("results/baselines/random_reference.csv", "Phase-0 ranking floor"),
    "results/DATASET_REPORT.md": ("appendices/DATASET_REPORT.md", "Canonical split and evaluation statistics"),
    "requirements.txt": ("reproducibility/requirements.txt", "Pinned experimental environment"),
}
for name in ("dataset", "b0", "b0_uw", "b1", "b2", "b3", "b4"):
    FILES[f"configs/{name}.yaml"] = (f"reproducibility/configs/{name}.yaml", "Frozen baseline configuration; see config interpretation notes")
FILES["configs/extensions/effective_noise_v1.yaml"] = (
    "reproducibility/configs/effective_noise_v1.yaml", "Frozen E1 protocol")
for name, reason in {
    "test_means.csv": "E1 final utility and seed SD",
    "contrasts.csv": "E1 adjusted primary and exploratory comparisons",
    "rank_selection.csv": "E1 validation-selected ranks",
    "geometry_means.csv": "Corrected float64 summary of effective noise and local norms",
    "geometry_per_run.csv": "Seed-level evidence including unstable finite trajectories",
    "subgroups_test.csv": "Descriptive activity/cold-target results",
    "personalization_validation.csv": "Exploratory local-state ablation; not a deployable private model",
    "accounting.csv": "Training and separate one-release popularity accounting",
    "synthetic_monte_carlo.csv": "Numerical verification of Gaussian moments",
    "selected.json": "Frozen equal-budget hyperparameter selection",
    "audit/final_audit.json": "Completed E1 integrity and verification record",
}.items():
    FILES[f"results/extensions/effective_noise_v1/{name}"] = (
        f"results/effective_noise/{Path(name).name}", reason)
FILES["docs/extensions/EFFECTIVE_NOISE_THEORY.md"] = (
    "appendices/EFFECTIVE_NOISE_THEORY.md", "Derivations and limitations, with prior-art citations")
FILES["docs/extensions/EFFECTIVE_NOISE_PROTOCOL.md"] = (
    "appendices/E1_PROTOCOL.md", "Predeclared E1 design and privacy scope")
FILES["docs/extensions/NOISE_MEMORY_PROTOCOL.md"] = (
    "research/E2_PROTOCOL.md", "Pre-output exploratory intervention design")
FILES["docs/extensions/NOISE_MEMORY_ALIGNMENT_PROTOCOL.md"] = (
    "research/E2B_ALIGNMENT_PROTOCOL.md", "Post-E2, pre-E2b disclosed coupling control")
FILES["docs/extensions/NOISE_MEMORY_RESULTS.md"] = (
    "research/E2_RESULTS.md", "Complete exploratory outcomes, coupling confound and limitations")
for origin,destination in [
    ("NOISE_REPLICATION_PROTOCOL", "E3_PROTOCOL"),
    ("NOISE_REPLICATION_RESULTS", "E3_RESULTS"),
    ("RANKING_TAIL_PROTOCOL", "E4_PROTOCOL"),
    ("RANKING_TAIL_RESULTS", "E4_RESULTS"),
]:
    FILES[f"docs/extensions/{origin}.md"] = (f"research/{destination}.md", "Separately declared follow-up and its limitations")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    # E2 aggregate/seed-level diagnostics enter only once the full run has finished.
    e2 = SOURCE / "results/extensions/noise_memory_v1"
    if (e2 / "summary.csv").exists():
        for name in ("summary.csv", "by_seed.csv", "inventory.csv", "freeze.json", "synthetic_checks.json"):
            FILES[f"results/extensions/noise_memory_v1/{name}"] = (
                f"results/noise_memory/{name}", "Exploratory E2; prediction disturbance, no relevance/DP-release claim")
    if (SOURCE / "results/extensions/noise_memory_aligned_v1/summary.csv").exists():
        for name in ("summary.csv", "by_seed.csv", "inventory.csv", "freeze.json", "synthetic_checks.json"):
            FILES[f"results/extensions/noise_memory_aligned_v1/{name}"] = (
                f"results/noise_memory_aligned/{name}", "E2b diagnostic coupling control; no improved marginal mechanism claim")
    for name in ("dataset.json", "freeze.json", "synthetic_checks.json", "completion_audit.json",
                 "validation_by_seed.csv", "validation_summary.csv", "endpoint_by_seed.csv",
                 "endpoint_summary.csv", "coupling_contrasts_by_seed.csv"):
        FILES[f"results/extensions/noise_replication_v1/{name}"] = (
            f"results/noise_replication/{name}", "E3 independent-data and balancing control; validation only")
    for name in ("freeze.json", "synthetic_checks.json", "completion_audit.json", "audit_summary.json",
                 "by_seed.csv", "summary.csv", "screen_summary.csv"):
        FILES[f"results/extensions/ranking_tails_v1/{name}"] = (
            f"results/ranking_tails/{name}", "E4 rejected ranking-tail candidate; no held-out labels used")
    for origin, (destination, reason) in FILES.items():
        source, target = SOURCE / origin, BUNDLE / destination
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        records.append(dict(destination=destination, source=f"fedrec-dp/{origin}",
                            sha256=digest(source), bytes=source.stat().st_size, reason=reason))
    with (BUNDLE / "reproducibility/provenance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (BUNDLE / "reproducibility/curation.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "copied_files": len(records),
        "copied_bytes": sum(x["bytes"] for x in records),
        "source_repository_commit_at_start": "c7df5ac964c67971e5058a2fd34b41b52a508eb9",
        "policy": "Explicit relevance whitelist; all historical originals remain in sibling archive",
    }, indent=2) + "\n")
    print(f"Curated {len(records)} source files ({sum(x['bytes'] for x in records):,} bytes)")


if __name__ == "__main__":
    main()
