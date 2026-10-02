# E1 continuation handoff

E1 effective-noise study v1 is complete and frozen. Read the
[results report](EFFECTIVE_NOISE_RESULTS.md),
[protocol](EFFECTIVE_NOISE_PROTOCOL.md),
[theory note](EFFECTIVE_NOISE_THEORY.md) and
[contribution audit](RESEARCH_POSITIONING.md), then the latest append-only
research-log entries. The original `handoff.md` remains the baseline history.

## Current conclusion

Fixed public orthonormal B is a useful linear baseline, but did not improve
private utility over jointly trained factors at T50 under the equal four-candidate
budget. Sixteen adjusted primary contrasts: zero positive, eight negative,
eight include zero. All four ranks are reliably worse at eps2. No grid expansion.

User-level DP popularity is strong: eps1 test NDCG .048389±.001125 versus
.011958 for validation-selected FixedB r16 and .025136 for selected Two r4.
Bounded no-noise popularity is .049885; the old raw-count point reference remains
.044292. Bounding changes user weighting, so do not attribute that gain to DP.

The bilinear term contributes only about .2–.3% of expected matrix-shock energy
at eps1 in the selected Two runs. Linear factor scale and private local P norms
matter more here. Finite states can be badly conditioned: Two r32/eps1/seed123
has maximum local norm about 2.91e20. That run remains in every reported mean.

## Frozen artifacts and safeguards

- New names: E1-Full, E1-Two, E1-FixedB and E1-DPPop. No historical B0–B4
  source/config/result/checkpoint/data file was changed; 3932 initial hashes verify.
- Output root: `results/extensions/effective_noise_v1/`; ignored local checkpoints:
  `checkpoints/extensions/effective_noise_v1/`.
- Config: `configs/extensions/effective_noise_v1.yaml`, frozen true. All q/T/delta/
  accounting settings match B2; exact B2 sigma values were recomputed and checked.
- Search: 648 jobs, 11 failures retained, four candidates for each of nine model
  units, all six levels on three seeds for eligibility, eps4 validation selection.
- Final: 270 finite T50 states, five seeds, 21 popularity arrays. 756 unique
  training records including failures; selected search cells reused by hashes.
- Freeze at `20261002T153741Z`; first extension test output at
  `2026-10-02T15:38:44.686919Z`. Each of 291 retained models scored once.
- Analysis: 100000 common paired user-bootstrap draws, seed 2026; primary
  99.6875% intervals across sixteen contrasts. Conditional on these training runs.
- Test remains the historically exposed ML-100K benchmark. Nothing is tuned on
  the extension test. MovieLens-1M is still unevaluated and reserved for replication.
- Do not rerun training to regenerate analysis or rerun test for selection.
  Read saved `test/` CSVs, `test_records.json` and retained diagnostics.
- Runner verifies settings, split, source/checkpoint hashes and complete inventory;
  test requires the immutable freeze snapshot. Preserve every snapshot/manifest.

## Diagnostic correction to preserve

The frozen runner used float32 to compute mean user norms. On the large but
finite P of r32/eps1/seed123 it overflowed in two rounds. It did not alter P,
effective-noise moments (which use float64), or test scores/ranks. The live
analysis corrects final mean/median/max user norms from checkpoints in float64.
Original code, raw logs, pre-test analysis and pre-correction geometry tables
are preserved. Do not reopen or retrain E1 for this diagnostic issue.

The full suite has 203 passing tests. Avoid redirecting its live output under
`results/` or writing experimental artifacts while the suite runs: existing
fixtures intentionally guard that directory. Capture stdout under `/tmp` and
copy it after completion. The first final verification exposed this harness/log
interaction; its error log is retained separately from the corrected rerun.

## Productive next research

Do not try to rescue the fixed-factor narrative with test-informed grid expansion.
Declare a separate study that controls effective shared-step scale separately
from local user updates, with sensitivity/clipping geometry and local norm
diagnostics included. Orthogonal FixedB is score-equivalent to rank-r BPR; that
equivalence is an essential baseline interpretation, not an algorithmic novelty.

Next priorities are an independent larger-data replication, a published private
recommender comparison with compatible whole-user accounting, and total
communication-budget controls with sigma solved for every different horizon.
Any private warm start or private user-state-informed global scale needs explicit
privacy treatment. A metadata residual is a new direction 3 method, not a silent
addition to E1 or the old B2/B4 configurations.

The user authorized independent research, implementation, experiments and docs,
without waiting for more instructions or using Claude. Existing local historical
supervisor gates are superseded by that instruction. Preserve the frozen history
and report unfinished replication accurately.
