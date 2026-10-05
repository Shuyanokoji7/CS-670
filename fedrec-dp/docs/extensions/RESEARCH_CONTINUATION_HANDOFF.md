# Research continuation handoff — 2026-10-03

## Latest continuation: E3/E4 completed

Read `NOISE_REPLICATION_RESULTS.md`, `RANKING_TAIL_RESULTS.md` and
`../../../CS-670/research/NOVELTY_STATUS.md` for the latest assessment.
Thirty transferred ML-1M training runs and 200 E3 diagnostic probes completed
without failures (2800 observations). Disabling balancing during continuation
removes the large raw/aligned churn gap on both datasets. This narrows the
finding to a diagnostic affected by factor-basis changes, rather than large
intrinsic personalization memory. Both datasets are from MovieLens; overlap
of their user populations was not audited.

E4 completed 50 frozen-state checks and 3200 predetermined ranking pairs.
Maximum variance-matched Gaussian risk error is 7.66859e-6; its .01-error
advancement gate failed. The underlying product-normal law is known prior art.
Do not turn this rejected candidate into a claimed mechanism or new theorem.

ML-1M split: 6035 users, 3706 items, 563206 training positives, 6035 validation
and test targets each. Fingerprint:
`e68ef765cc661f5db1864f9f8b2843bee70c7ad261c116421d2c7102ec71bd54`.
Its validation was evaluated; **its test targets remain unscored**. All six
collaborative validation means remain below DP-popularity means. This is a
transferred-setting point comparison, without tuning or significance testing.

E3/E4 runners and protocols are now frozen. Separate result roots are
`results/extensions/noise_replication_v1/` and `ranking_tails_v1/`.
`experiments/summarize_noise_followup.py` verifies saved evidence without
training/model scoring. E3/E4 synthetic checks: 12/6 cases passed. Report
curation now selects 65 source files, 28 references and nine figures with
PNG/PDF/SVG exports. Raw data, checkpoints and per-pair logs stay outside it.

Next substantive work: faithfully audit/compare published private ALS,
establish metadata-only local personalization, and test a specific residual
or communication-budget hypothesis against those stronger controls. These
are established baseline ideas; novelty needs an additional consequence.
Do not continue similar noise pulses merely to increase experiment counts.
The research direction stays flexible and paper-level novelty unconfirmed.

The following E2-only account is retained as historical context; statements
below that ML-1M is unrun are superseded by this update.

---

The objective is a defensible paper contribution somewhere in private/federated
recommendation. The direction is explicitly flexible. The user authorizes
independent research, experiments and documentation; do not wait for another
method choice. No sub-agent or external communication is part of this work.

## Completed in this continuation

1. Reviewed the existing handoff, E1 results and broader primary literature.
   Added a prior-art audit including PRISM, FLoRG, NoiseCurve and other close
   work. Fixed factors, gauge-aware geometry, Procrustes alignment and generic
   noise propagation cannot be presented as novel by themselves.
2. Declared and ran E2: 120 pulse/reset probes of 30 frozen E1 checkpoints;
   Full, FixedB-r8, Two-r8; epsilon 1/2; five seeds; two draws; alpha .25/1.
   All completed, 1680 observations. No held-out targets scored.
3. Declared E2b after inspecting E2, before E2b outputs: 40 Two-r8 probes with
   pre-round orthogonal alignment to control common-noise coupling. All
   completed, 560 observations. This is disclosed adaptive exploration.
4. Added a synthetic sign-change null example: identical effective matrices
   can have positive raw-coupled distance and zero consistently transformed
   distance under the same marginal noise mechanism. This is known algebra.
5. Created the organized sibling `CS-670/` course-report base: integrated draft,
   45 selected source files, six reproducible figures in three formats, methods,
   revision history, references, pilot results and a saved-evidence verifier.

## Main new finding and what it does not show

At full pulse amplitude, ten rounds after restoring shared state, Two-r8
top-10 set disagreement is 17.8546% / 9.3960% at starting epsilon 1 / 2 with
raw-coordinate coupling, versus .4204% / .1083% with aligned coupling. Full
and FixedB-r8 at epsilon 1 retain .5212% / .3747% disagreement. Each mean uses
five checkpoint seeds after averaging two pulse draws within seed. No probes
were excluded; seed dispersion and score-scale outliers remain documented.

The large raw Two effect was substantially a diagnostic-coupling confound.
The control preserves each marginal mechanism in exact arithmetic, not an
improved training method. Measurable local-state memory remains, with modest
population ranking disturbance. No relevance improvement, new DP guarantee,
unique counterfactual, significance claim or paper-level novelty is established.

## Sources of record

- `docs/extensions/NOISE_MEMORY_PROTOCOL.md` and
  `NOISE_MEMORY_ALIGNMENT_PROTOCOL.md`: frozen pre-output designs.
- `docs/extensions/NOISE_MEMORY_RESULTS.md`: completed pilot interpretation.
- `experiments/probe_noise_memory.py` and `probe_noise_memory_aligned.py`:
  frozen code, reading existing E1 checkpoints. Do not edit these retrospectively.
- `results/extensions/noise_memory_v1/` and `noise_memory_aligned_v1/`:
  original probe JSON, observations, inventories, summaries and source freezes.
- `../CS-670/README.md`: curated course-report entry point.
- `../CS-670/reproducibility/audit.json`: latest bundle/archive verification.
- `../CS-670/references/PRIOR_ART_AUDIT.md`: closest work and reading-depth limits.

The archive is source of record; `CS-670/reproducibility/curate_sources.py`
refreshes explicit report copies without altering originals. Figure generation
reads tables only. Historical B0–B4 and E1 results, source and manifests remain
unchanged. The new verification checks 3932 protected files and all E1 manifests,
and recomputes pilot seed summaries from the original observations.

## Next research decisions

Follow `CS-670/research/RESEARCH_DECISIONS.md`. Prioritize whether the diagnostic
has a distinct and useful contribution, with forward/backward citation tracing
and independent-data replication, before naming a new method. A substantial
follow-up should compare raw/aligned/other defensible couplings, include an
optimizer without balancing, and test whether a fixed diagnostic predicts an
actionable intervention better than norms or clipping rates. Local/shared
learning-rate changes are established controls, not novelty in themselves.

MovieLens-100K is historically exposed. MovieLens-1M, published private ALS,
metadata residuals and equal-total-communication comparisons remain unrun.
Freeze any new protocol before its outputs; give a new extension its own name
and preserve this pilot. Changed training horizons need recalibrated privacy.
Do not expand a grid after test inspection or quietly discard finite outliers.
If the local effect is negligible or prior work already covers the study,
redirect toward another evidence-supported question. Novelty is not guaranteed.

No commit, push, external message or publication was made in this continuation.
