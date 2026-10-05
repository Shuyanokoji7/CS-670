# Research decisions and advancement criteria

The user's objective is a defensible paper contribution; no method or direction
is fixed. Completed baselines remain useful evidence even if the paper question
changes. This document ranks candidates on evidence, overlap and feasibility.

| Candidate | Evidence / overlap | Decision |
|---|---|---|
| Fixed public factor as a new method | E1 has zero positive adjusted primary contrasts; FFA-LoRA, FedASK, linear-parameterization work overlap. | Retain as a baseline and negative result. |
| Generic noise geometry / gauge-aware optimization | Strong direct overlap with GeoDP, PRISM, FedGSA and alignment methods. | Do not make a broad novelty claim. |
| Public metadata plus private residual | Useful way to challenge DP popularity, but public-feature MF and local/global residual learning are established. | Reserve as a later baseline or method only with a specific extra hypothesis. |
| Communication-budget allocation between local and shared learning | Motivated by B3's large total volume; Bietti already studies relative local/global step size. | Worth controlled evaluation; tuning a ratio alone is insufficient. |
| Persistent local state and the validity of noise-attribution measurements | E3 reproduces the coupling effect on ML-1M and removes the large gap by disabling continuation balancing. | Retain the narrow diagnostic finding. Reject large intrinsic memory instability as the method's motivation; broader novelty unconfirmed. |
| Non-Gaussian ranking tails missed by average noise energy | Known distributional theory applies, but E4 finds maximum matched probability error below .000008 in 3200 fixed pairs. | Reject for these checkpoints; preserve the predeclared negative screen. |

## What the next substantial study must answer

1. E3 now answers the independent-data and continuation-balancing control.
   It does not evaluate a model trained from scratch without balancing.
   Another pulse sweep needs a specific new consequence to justify it.
2. Are local-memory effects relevant to ranking quality, beyond label-free
   prediction disagreement? Use a new frozen validation protocol and an untouched
   larger-data test; current ML-100K is already exposed.
3. Can a deployable change, such as a separately controlled local update rate
   or regularizer, reduce the measured effect without harming useful learning?
   Both are established controls, so a gain needs a mechanistic prediction and
   comparison with prior methods, not an algorithm-name novelty claim.
4. Does the diagnostic predict useful intervention choices out of sample better
   than simple clipping rate, user norm, coordinate-noise norm and validation
   loss? This distinguishes an explanatory method from another collection of
   plots. Candidate predictors must be fixed before held-out comparisons.
5. Under equal *total* communication and per-run user privacy, does any learned
   personalized model add value over DP popularity and published private ALS?
   Recompute sigma for every changed training horizon. This is still unrun.

## Stop and redirect rules

If only raw-coordinate coupling produces a large effect, report a measurement
confound, not a large intrinsic failure of personalization. If local effects
remain small across replication, do not build a method around them. If a prior
paper already performs the same intervention/attribution study, use it as a
baseline and seek a distinct question. A course report can faithfully present
negative results; a paper contribution is not guaranteed by experiment volume.

The latest 2026-10-03 output includes independent-data replication and a
rejected alternative candidate. It is not a completed paper or a proven new
algorithm. See [current assessment](NOVELTY_STATUS.md),
[E3 results](E3_RESULTS.md) and [E4 results](E4_RESULTS.md).
