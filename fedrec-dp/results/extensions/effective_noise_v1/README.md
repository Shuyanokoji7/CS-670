# E1 effective-noise study artifacts

Completed 2026-10-02. This directory contains separately named follow-up research;
all historical B0–B4 artifacts remain unchanged.

- [Results report](../../../docs/extensions/EFFECTIVE_NOISE_RESULTS.md)
- [Predeclared protocol](../../../docs/extensions/EFFECTIVE_NOISE_PROTOCOL.md)
- [Moment derivations and recommendation geometry](../../../docs/extensions/EFFECTIVE_NOISE_THEORY.md)
- [Related work and contribution audit](../../../docs/extensions/RESEARCH_POSITIONING.md)

## Main outputs

| Artifact | Meaning |
|---|---|
| `accounting.json`, `accounting.csv` | Exact B2 sigma checks and separate one-release analytic-Gaussian popularity calibration |
| `search.csv`, `candidate_screen.csv`, `selected.json` | All 648 declared search jobs, eligibility/failures, validation-only choices |
| `final_records.json`, `final_validation.csv` | 270 retained finite T50 training states, scientific settings and artifact hashes |
| `rank_selection.csv` | Five-seed validation-selected ranks, recorded before test |
| `popularity_validation.json` | Twenty private score vectors and one deterministic bounded no-noise control |
| `freeze.json`, `snapshots/` | Freeze time, frozen inputs, checkpoint inventory and relative SHA-256 manifests |
| `runs/` | 756 unique training records including 11 failures, and instantaneous per-round diagnostics |
| `test/`, `test_records.json` | 291 test outputs/per-user ranks; each retained model scored once |
| `test_means.csv`, `contrasts.csv` | Five-seed utility and common paired user-bootstrap intervals |
| `primary_seed_differences.csv` | Each training seed's FixedB-minus-Two NDCG contrast |
| `geometry_per_run.csv`, `geometry_means.csv` | Actual Q/score shocks, conditional energy moments, clipping and corrected float64 local norms |
| `personalization_validation.csv` | Own/permuted/mean local-P validation-only ablations |
| `subgroups_validation.csv`, `subgroups_test.csv` | Descriptive activity/cold-target groups |
| `synthetic_gauge.csv`, `synthetic_monte_carlo.csv` | Public synthetic checks with no MovieLens labels |
| `plots/*.{png,svg}` | Standalone utility and effective-noise figures |
| `audit/` | Initial historical hashes, archived pre-correction diagnostic tables and final checks |

Training/local-state checkpoints are in `checkpoints/extensions/effective_noise_v1/`
and are ignored by git, as are historical checkpoints. A fresh clone needs those
checkpoints and the frozen processed/raw dataset to check them. The per-user
metric CSVs and reports can be inspected without training. Diagnostics and local
state are research artifacts outside the DP server-release claim.

## Commands executed from the project directory

Each command below was run with the project environment; logs are retained here.
These are stage records, not instructions to retrain or rescore finalized models.
The runner verifies hashes before reusing jobs and refuses partial artifacts.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py accounting
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/effective_noise_geometry.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py profile
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py search --workers 12
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py final-train --workers 12
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/analyse_effective_noise.py validation
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py freeze
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py score-test
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/analyse_effective_noise.py test
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/analyse_effective_noise.py plots
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py audit
```

The analysis was corrected after first scoring to compute final local norms in
float64 from saved P. The original frozen runner/raw diagnostics were preserved,
along with the original analysis in `snapshots/analysis_before_test/` and the
earlier geometry tables in `audit/geometry_before_float64_norm_correction/`.
There was no retraining, changed test rank or excluded finite trajectory.

`MANIFEST.sha256` is written outside its target directory and excludes MANIFEST*
files. Verify it from this directory with `sha256sum -c MANIFEST.sha256`.
