# Handoff for the next research agent

**Update 2026-10-03:** E1 is complete, and a later independent continuation
completed E2/E2b/E3/E4 and created the curated `../CS-670/` UGP report folder.
E3 adds MovieLens-1M and a balancing control; E4 rejects a ranking-tail
candidate. ML-1M validation is exposed, but its test has not been scored.
Read [the latest continuation handoff](docs/extensions/RESEARCH_CONTINUATION_HANDOFF.md)
and [the report entry point](../CS-670/README.md) first. The research direction
is flexible; paper-level novelty remains unconfirmed. The remainder of this
file preserves the earlier 2026-10-02 baseline handoff.

Prepared 2026-10-02. Repository: `/home/ag/Amazon ML/fedrec-dp/`.
Git root: `/home/ag/Amazon ML/`. This is an ongoing CS670 research project.

## Start here

**Phase 0 and B0–B4 are completed, frozen and test-scored. Do not restart B3 or B4.**
The user's next objective is to develop this work into a research paper through a
carefully controlled follow-up study.

**Latest execution instruction:** work independently, without Claude. The user
explicitly replaced the earlier arrangement in which Codex only supervised Claude.
You may now inspect files, edit code and documentation, and run commands yourself.
Do not try to resume the old Claude/tmux session or delegate work to Claude.

Read this file, then `docs/CODEX_HANDOFF.md`, the **complete** `RESEARCH_LOG.md`,
current configs, `docs/B2_RESULTS.md`, `docs/B3_RESULTS.md`, `docs/B4_RESULTS.md`,
and `results/b34_*.csv`. Resolve conflicts using:

1. The user's latest instructions.
2. Latest dated research-log entries.
3. Current configs and result files.
4. Handoffs, README, then older reports and archives.

Some older documentation is stale: README's early Status section still says low-rank
models are unimplemented; older sections of `docs/CODEX_HANDOFF.md` mention 113 tests
and waiting to start B3/B4. Those statements are historical, not outstanding tasks.
`configs/b3.yaml` has a placeholder `local_lr: null`; actual frozen rates are in
`selected_lr`. B4 settings come from `per_rank`. B2 runs `privacy.T`, not its unused
`federated.max_rounds` field.

## Publication and workspace

- GitHub: https://github.com/Shuyanokoji7/CS-670
- Branch: `main`, tracking `origin/main`.
- Research release: `f9c0a2b257f0ba1669719fce5dd969d5a50d1c2b`.
- Publication documentation: `81680fd6808425fc355f445e8004be0892bf0129`.
- At the start of this handoff task, the working tree was clean at `81680fd`.
- Code, configs, tests, reports, CSVs, plots, raw experiment outputs, snapshots and
  superseded stages are published. Checkpoints, `data/raw/`, `data/processed/` and
  `.venv` are ignored and retained on this machine. Fresh clones need those artifacts
  supplied or regenerated for checks that require them.
- This new handoff and its research-log entry were created after publication;
  creating them does not commit or push them.

## Research question and conclusion

**H1:** At the same formal user-level DP guarantee, a lower-dimensional shared
representation may outperform a full-dimensional representation in sufficiently
noisy privacy regimes.

The result is **limited, conditional support**, not a blanket confirmation or
disproof. After multiplicity correction, B4 beats B2 in four of sixteen primary
comparisons: ranks 8/16/32 at epsilon approximately 2, and rank 8 at epsilon
approximately 1. Every rank is worse at epsilon approximately 8; no reliable
difference is detected at epsilon approximately 4. Every B3 rank is worse than B1
without DP. No monotonic movement toward smaller optimal ranks was established.

All five-seed private means at epsilon <= 2 remain below the train-only popularity
point reference, NDCG@10 = 0.044292. No paired significance claim is made against
popularity. B4 rank 32 at epsilon 4 is merely numerically close to it.

Low rank received **more tuning than B2**. Factorization, optimization dynamics,
learning rates and balancing differ. The results do not isolate a causal effect
of coordinate count. The user expressed concern that the hypothesis was wrong;
explain the narrower result accurately rather than trying to manufacture a win.

## Frozen data and evaluation

- MovieLens-100K: 943 raw users, 1,682 items, 100,000 ratings.
- Positive threshold: rating >= 4; 55,375 positives; 942 retained users.
- Per-user chronological leave-two-out: train 53,491, validation 942, test 942.
- Timestamp ties use a stable SHA-256 tie-break with split seed 2026, independent
  of model-training seeds. This is not a global calendar-time split.
- Split fingerprint:
  `6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989`.
- Raw `u.data` SHA-256:
  `06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490`.
- Full ranking excludes every previously rated item, including low ratings, while
  retaining the target. Future ratings do not filter earlier candidates.
- Cold targets remain included: 6 validation, 14 test.
- NDCG@10 is primary; MRR@10 is secondary. HR@10 = Recall@10 because there is one
  held-out target. Do not treat them as independent evidence.
- Training negatives exclude only training positives. Excluding held-out labels
  would use future information.

Do not change Phase 0, the evaluator, candidate construction, thresholds, targets,
or frozen baseline configurations/results. A genuine correctness issue must be
identified and reported before reopening a frozen phase.

## Frozen models

- **B0:** centralized, interaction-weighted BPR, d=64, Adam lr .001, L2 .01,
  batch 1024, patience 100 epochs, maximum 1000 epochs.
- **B0-UW:** same settings with user-uniform sampling; a strict weighting control.
  Original three-seed NDCG means: B0 .0860, B0-UW .0917, B1 .0875. All overall
  paired weighting/federation decomposition intervals include zero.
- **B1:** one client per user; persistent private local p_u in R^64 and shared Q.
  Poisson q=.1, local full-batch SGD lr 5, E=2, L2=1e-5, server lr 1,
  equal-user averaging divided by realized participants. Empty rounds are skipped.
  Validation every 10 rounds, patience 100 evaluations, maximum 8000 rounds.
- **Final B2:** full-rank user-level DP; exactly T=50, C=1, server lr 1, q=.1,
  delta=1e-5, local lr 5/E=2. Clip each whole shared update, add Gaussian noise
  to the clipped SUM, then divide by fixed qN=94.2. Empty rounds receive noise
  and are accounted. No early stopping or checkpoint selection.
- **B3:** Q=AB, A in R^(1682 x r), B in R^(r x 64), ranks 4/8/16/32;
  p_u remains local and 64-dimensional. Frozen local rates are
  {4:.625, 8:1.25, 16:1.25, 32:.625}. B1 convergence rules otherwise apply.
  Rank 32/seed 123 alone had a declared 16000-round ceiling; best round 8350,
  stopped at 9350. All other final runs stopped below 8000. All 20 runs passed.
- **B4:** same low-rank architecture; one joint clip over concatenated [delta A,
  delta B], noise on ALL shared factor coordinates, fixed qN, T=50 and the same
  privacy accounting as B2. C=1 for every rank. Rank 4 uses local lr 2.5/server
  lr 1; ranks 8/16/32 use local lr 5/server lr .5.

B3/B4 use B1's regularization on local p and effective touched item rows, chained
through the factors; there is no additional factor-only L2 penalty. QR/core-SVD
balancing preserves AB to floating-point tolerance. It is post-processing in B4.

Opacus is used **only for accounting**, PRV primary and RDP cross-check. At T=50:

| Target epsilon | Sigma | Achieved PRV epsilon | RDP epsilon |
|---|---|---|---|
| 8 | .805072021484 | 7.9675 | 9.1198 |
| 4 | 1.15192871094 | 3.9639 | 4.4973 |
| 2 | 1.76044921875 | 1.9933 | 2.2206 |
| 1 | 2.97749023438 | .9901 | 1.0872 |

Epsilon depends on q, sigma, T and delta, not dimension or C when the noise
multiplier relative to C is fixed. Changing T requires fresh accounting.

## Current five-seed results

Seeds: 42, 123, 2026, 7, 99. B1/B2 seeds 7/99 are exact frozen replications on
separate paths. Their original three-seed core files are unchanged. Do not mix
the three-seed B2 report with the five-seed B3/B4 comparison table.

| Model | Test NDCG@10 mean | Training-seed SD |
|---|---|---|
| B1 | .089008 | .002494 |
| B3 rank 4 | .059468 | .003814 |
| B3 rank 8 | .071010 | .005964 |
| B3 rank 16 | .076333 | .007223 |
| B3 rank 32 | .068074 | .013541 |

| Privacy level | B2 | B4 r4 | B4 r8 | B4 r16 | B4 r32 |
|---|---|---|---|---|---|
| Matched no DP | .057889 | .040834 | .048083 | .049143 | .050931 |
| epsilon ~8 | .052271 | .038544 | .045385 | .043628 | .044022 |
| epsilon ~4 | .042169 | .037846 | .042653 | .038274 | .044489 |
| epsilon ~2 | .025428 | .029079 | .034485 | .033993 | .036504 |
| epsilon ~1 | .011461 | .015976 | .018162 | .016685 | .015390 |

Exact means/SDs for NDCG, HR and MRR: `results/b34_test_means.csv`.
Contrasts: `results/b34_contrasts.csv`. Diagnostics and communication:
`results/b34_perturbation.csv`, `results/b34_communication.csv`.

Statistics: average each user's metric over the five seeds first, then 100,000
common paired user-bootstrap resamples, seed 2026. Intervals are conditional on
these training runs. Bonferroni intervals: B3 four contrasts, 98.75%; B4 sixteen
primary contrasts, 99.6875%. Ordinary 95% intervals are also reported.

B4 validation-selected ranks were frozen before test: noDP 32, epsilon 8 rank 8,
epsilon 4 rank 32, epsilon 2 rank 16, epsilon 1 rank 16. Do not replace epsilon 1's
selected rank 16 with test-best rank 8. Rank 16's epsilon-1 adjusted interval
includes zero; rank 8's positive primary contrast is a separate result.

## Communication and privacy limits

- Dense float32 upload + download per selected client/round: full rank 861,184 B;
  ranks 4/8/16/32: 55,872 / 111,744 / 223,488 / 446,976 B.
- At the common T=50, B2 mean volume is 4.026 GB; B4 volumes are approximately
  .261 / .522 / 1.045 / 2.090 GB. These are simulated tensor volumes, not measured
  network traffic, and this dense format is not a universal DP requirement.
- B3 trains longer. Rank 32 uses 180.3 GB to its selected checkpoint versus
  B1's 93.9 GB: 1.92 times as much. Whole-run volumes are 222.3 versus 174.9 GB.
  Smaller payload does not guarantee smaller total communication.
- Privacy unit: an entire user, add/remove adjacency. Secure aggregation is
  assumed/simulated, not cryptographically implemented. p_u is never uploaded,
  averaged or clipped. Simulator checkpoints include private local state and
  are not themselves a DP server release.
- The DP claim covers the final training mechanism conditional on fixed
  hyperparameters and the stated threat model. Selection, diagnostics,
  evaluation and the complete research process are outside that guarantee.
  The reproducible noise RNG is not cryptographically secure.

## History and audit that must remain visible

- Initial B2 T=1000/C=1.5/server lr=2 is superseded, not the current baseline.
  The final protocol changed T, C AND server lr; do not attribute improvement
  to T alone. Preserve `results/raw/superseded_b2_T1000/` and other archives.
- Initial B3 settings diverged for some seeds; rank 32 also hit the budget.
  The correction and final convergence diagnostic preceded any B3 test scoring.
- B4's inherited long-horizon B3 rates undertrained at T=50. A larger bounded
  search followed; initial winners failed multi-seed stability. The final
  fallback required all 25 rank/configuration jobs to be finite at exactly T=50.
- Final freeze snapshots: `results/raw/b2_freeze_snapshot_20261002T172616/`,
  `results/raw/b3_rev2_freeze_snapshot_20261002T134133Z/`,
  `results/raw/b4_freeze_snapshot_20261002T134435Z/`.
- B3 freeze/first real test: 19:11:33/19:11:35; B4: 19:14:35/19:16:38,
  2026-10-02, Asia/Kolkata.
- Disclosed exception: nine accidental pre-freeze real-data evaluations of
  two-round B2 models in an earlier regression test. They were not inspected
  or used for selection. Evidence is preserved in
  `results/raw/audit_pre_freeze_test_regression/`. New runner tests use synthetic
  fixtures. Do not claim a pristine MovieLens-100K holdout.
- Final audit: all 23 raw files matched; 876/877 post-B2 non-doc files matched,
  with one append-only B2 command log changed during documentation. All 120 B4
  final checkpoints matched frozen settings. All 170 per-user files had the
  same 942 unique users and finite metrics.
- Preserve manifests and snapshot contents. Generate manifests outside their
  target directory and exclude MANIFEST files themselves. Older manifest
  correction exceptions are documented in the research log.

## Proposed continuation toward a paper

These directions were discussed with the user; **no follow-up method has been
selected or implemented, no new experiment protocol is frozen, and MovieLens-1M
has not been evaluated**. Do not describe proposals as completed work.

Recommended direction: explain when low-rank sharing helps through a controlled
study of parameterization and effective noise, then test a fixed-public-factor
alternative and independently replicate on MovieLens-1M.

For Q=AB, conditional on the signal-updated factors, a noisy step includes

    delta Q = A Z_B + Z_A B + Z_A Z_B.

The bilinear term and factor scales matter. Balancing preserves AB and does not
remove the term. Final Q norm includes signal and optimization drift; it is not
a direct measurement of effective noise. Measure perturbation to Q and scores.
Fixing one public factor removes the bilinear noise term, but may alter learning
dynamics; compare no-DP controls as well. A basis derived from private interactions
needs an appropriate privacy treatment, not an unaccounted non-private warm start.

Other useful directions: population/horizon scaling with equal tuning budgets;
public item metadata plus local personalization and a small private residual.
Adding metadata is a new method, not a silent change to frozen baselines.

First prepare a bounded protocol with matched tuning budgets, clipping-only
controls, a user-level DP popularity baseline, appropriate published DP baselines,
and uncertainty across training seeds. Select on validation, document/freeze the
protocol, and reserve MovieLens-1M test for the final evaluation. If T or the
mechanism changes, recompute privacy rather than reusing accounting blindly.
Keep every extension separately named; preserve original B0–B4 results.

Related work already identified; verify details and novelty before implementation:

- [FFA-LoRA](https://arxiv.org/abs/2403.12313): fixed-factor federated LoRA.
- [FedASK](https://arxiv.org/abs/2507.09990): factor-noise amplification and sketching.
- [From Bilinear to Linear](https://arxiv.org/abs/2609.26091): related linear parameterization work.
- [Private Matrix Factorization with Public Item Features](https://arxiv.org/abs/2309.11516).
- [Private Alternating Least Squares](https://proceedings.mlr.press/v139/chien21a.html).
- [Personalization Improves Privacy–Accuracy Tradeoffs](https://proceedings.mlr.press/v162/bietti22a.html).

These ideas have prior art. Applying a known technique alone is not an established
novel contribution, and publication is not guaranteed by an improved score.

## Environment, verification and code map

Use the project `.venv/bin/python`, Python 3.10, pinned CPU PyTorch and Opacus in
`requirements.txt`. Opacus is accounting-only; do not use PrivacyEngine to claim
whole-user DP. Preserve the pinned torch build when installing dependencies.
B0 uses four torch threads; federated workers force one BLAS/OpenMP/torch thread.
Existing orchestration used at most 20 one-thread workers; profile new workloads.

Before any code/experiment changes: inspect git state; read the full log/configs/
reports/results; run the full suite while experiments are idle; verify raw files,
split fingerprint and protected baseline hashes; report any source disagreements.

```bash
cd "/home/ag/Amazon ML/fedrec-dp"
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
sha256sum -c results/raw/audit_final_2026-10-02/raw_hashes_session_start_2026-09-28.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -c 'from src.data import load_processed, split_fingerprint; print(split_fingerprint(load_processed()))'
```

Latest full suite before publication: **181 passed**, 26 warnings. This handoff
task rechecked raw integrity (23/23), the fingerprint/counts, clean initial git
state, and collection of 181 tests. It did not rerun training or test scoring.

Core files: `src/data.py`, `src/evaluate.py`, `src/metrics.py` (frozen data/evaluator);
`src/bpr.py`, `src/train_bpr.py` (centralized); `src/federated.py`,
`src/train_federated.py` (B1); `src/privacy.py`, `src/train_dp_federated.py` (B2);
`src/lowrank.py`, `src/train_lowrank.py` (B3/B4); `src/threads.py`, `src/manifest.py`.
Runners: `experiments/run_b0.py` through `run_b4.py`.
Analysis/plots: `experiments/analyse_b3b4.py`, `experiments/plot_b3b4.py`.

Do not rerun training to regenerate tables. Read retained CSVs; analysis scripts
can overwrite their output files, so use a separately named extension or archive
before changing an analysis. Validate checkpoint metadata and hashes before reuse;
do not overwrite existing scored checkpoints or rescore test for model selection.
Use synthetic fixtures for new runner tests, not real experimental test labels.

Keep the user informed of findings and decisions. Work independently on authorized
tasks; do not add repeated permission gates or expand grids merely to obtain a
positive result. If a bounded experiment reaches a genuine bottleneck, report it.

## Suggested first message to the next agent

> Read `handoff.md`, the older `docs/CODEX_HANDOFF.md`, the complete research log,
> current configs and final result summaries. Work independently without Claude.
> B0–B4 are finished and frozen. Verify the repository before code changes, then
> prepare a concrete follow-up protocol on effective noise and matched-budget
> controls toward a research paper. Preserve all baseline/superseded artifacts;
> select only on validation and do not evaluate a new test protocol before freeze.

## Continuation update: E1 completed on 2026-10-02

The user subsequently authorized independent pursuit of direction 2. The earlier
"no follow-up method selected or implemented" proposal status above is historical.
E1-Full/E1-Two/E1-FixedB/E1-DPPop are now implemented, bounded-search trained,
frozen and scored. Original B0–B4 artifacts remain unchanged.

Start the next continuation with `docs/extensions/E1_HANDOFF.md` and
`docs/extensions/EFFECTIVE_NOISE_RESULTS.md`, then the latest research-log entries.
FixedB has no positive adjusted primary contrast and eight negative contrasts;
DP popularity is the stronger point reference at epsilon <=4. The measured
bilinear energy share is small in this regime. Do not revive the proposal as
though it has not been tested or reinterpret the negative result as a win.
MovieLens-1M, a published private recommender comparison and communication-budget
controls remain unfinished research, with concrete priorities in the E1 handoff.
