# Methods, statistics and scope for the UGP appendix

## Models and final settings

| Model | Shared representation | Privacy | Final training rule |
|---|---|---|---|
| B0 | Full item matrix, d=64 | None | Centralized BPR, interaction sampling, Adam .001, L2 .01 |
| B0-UW | Full item matrix, d=64 | None | Same B0 settings, user-uniform sampling control |
| B1 | Full item matrix, d=64 | None | q=.1, local SGD lr=5, E=2, L2=1e-5, server lr=1 |
| B2 | Full item matrix, d=64 | Entire user | B1 local settings; T=50, C=1, fixed qN=94.2, Gaussian aggregate noise |
| B3 | Q=AB, r=4/8/16/32 | None | Local rates .625/1.25/1.25/.625; B1 stopping rule |
| B4 | Q=AB, r=4/8/16/32 | Entire user | T=50, C=1; r4 lr=2.5/server=1; other ranks lr=5/server=.5 |
| E1-Full | Full item matrix | Entire user | Same final B2 mechanism; equal four-candidate E1 search |
| E1-Two | Both factors trained | Entire user | Common effective initialization with FixedB; r4/8/16 server=.5, r32 server=1; local lr=5 |
| E1-FixedB | Public orthonormal B, private A | Entire user | Local lr=5, server=1; B seeded independently of interactions |
| E1-DPPop | Bounded sum of user item indicators | Entire user | One release; norm bound sqrt(20); separate analytic Gaussian calibration |

All personalized federated models retain a 64-dimensional local user vector.
B3/B4/E1-Two balance factors by QR/core-SVD while preserving AB. Their
regularization applies to local P and effective touched Q rows, with no extra
factor-only penalty. Equal coordinate clipping thresholds do not imply equal
effective-Q sensitivity or optimization speed across parameterizations.

B0/B0-UW use at most 1000 epochs with patience 100. B1/B3 validate every ten
rounds with patience 100 validations and an 8000-round ceiling; B3 r32/seed123
has a declared 16000-round convergence diagnostic. The final selected checkpoint
for that run is round 8350. B2/B4/E1 always use final round 50. A comparison of
converged B1 with T50 B2 is not an isolated estimate of the effect of noise.

## Formal privacy statement

The protected unit is one complete user under add/remove adjacency. At each
private training round users join independently with q=.1. Each whole shared
update is clipped to norm C; the clipped sum receives N(0,sigma^2*C^2*I), then
is divided by fixed qN and multiplied by the public server learning rate. Empty
rounds are noised and accounted. Local P is never uploaded. The server is
assumed to observe only the noisy aggregate through a secure-aggregation
mechanism; this repository simulates that assumption and implements no
cryptographic protocol.

| Target epsilon | Sigma | Achieved PRV epsilon | RDP cross-check |
|---|---|---|---|
| 8 | .805072021484 | 7.9675 | 9.1198 |
| 4 | 1.15192871094 | 3.9639 | 4.4973 |
| 2 | 1.76044921875 | 1.9933 | 2.2206 |
| 1 | 2.97749023438 | .9901 | 1.0872 |

Delta is 1e-5. PRV is the declared primary accountant; RDP is a separate bound,
not an additional privacy expenditure. Opacus is used for accounting only.
Dimension and C do not alter epsilon at fixed q, sigma, T, delta. Different
training horizons require recalibration. The exact rows are in the results.

Each DPPop user contributes a binary vector of distinct training-positive items,
scaled to norm at most sqrt(20). The add/remove sensitivity of its sum is
sqrt(20). One full-population Gaussian release uses the analytic Gaussian
condition of Balle and Wang; sigma is .600229/1.081162/1.993812/3.730632 for
targets 8/4/2/1. This sigma cannot be substituted into the iterative mechanism.

Claims cover one final training mechanism with fixed hyperparameters. Private
checkpointed user states, model selection, detailed diagnostics, evaluation,
and simultaneous release of all runs are outside this per-run guarantee.
Research RNGs are reproducible rather than cryptographically secure. E2/E2b
are private-state counterfactual probes and make no DP release claim.

## Evaluation and statistical units

The canonical split and full-ranking rules are in [DATASET_REPORT.md](DATASET_REPORT.md).
NDCG@10 is primary. MRR@10 is secondary. HR@10 equals Recall@10 because there
is one held-out relevant target. Cold targets remain in the evaluation.

The original weighting comparison uses seeds 42/123/2026. Final B1-B4 and E1
use those plus 7/99. Tables explicitly identify three versus five seeds.
Error bars on utility figures are training-seed SD, not confidence intervals.

For primary paired baseline/E1 contrasts, first average each user's metric
over the five seeds; then use 100000 common paired user-bootstrap samples,
seed 2026. Report ordinary 95% intervals and Bonferroni simultaneous intervals:
98.75% across four B3 comparisons; 99.6875% across sixteen B4 or sixteen E1
primary comparisons. These intervals condition on the retained training runs.
They are not a joint uncertainty calculation over independent datasets and
training seeds. Secondary contrasts and subgroups are exploratory.

E1 gives each of nine model units the same four learning-rate candidates,
screened on three seeds at six noise/clipping levels. There are 648 search
jobs, eleven failed trajectories, 270 finite final T50 states, and 756 unique
training records because selected cells are reused. Twenty-one popularity
arrays plus the 270 states give 291 one-time test scorings. No test-based grid
expansion is included. FixedB's selected local rate is at the upper grid edge.

E2 averages two pulse replicates within each checkpoint seed before reporting
five-seed summaries. It uses all 120 probes; E2b uses all 40 declared follow-ups.
No user-bootstrap significance or recommendation-relevance claim is made for
their top-10 set disagreement. Their candidate set excludes training positives,
unlike the historical held-out evaluator, and they access no held-out targets.

## Limits that matter for interpreting the conclusions

- Historical B4 had more tuning than B2. Coordinate count is not isolated from
  initialization, clipping, factor scaling and optimization.
- Equal rounds demonstrate payload/volume savings at that horizon. They do not
  establish superiority under an equal total communication budget.
- FixedB is score-equivalent to ordinary rank-r BPR through v=Bp; it does not
  contain semantic public movie features. Failure is not explained simply by a
  lower hard score-rank capacity than Two at the same r.
- Fourteen cold test targets are insufficient for strong cold-item conclusions.
- Huge finite local norms are retained. Score RMSE is sensitive to their scale;
  ranking changes and robust seed summaries must accompany it.
- ML-100K has repeated historical exposure, including an archived accidental
  early regression-test evaluation. It is a development benchmark here, not a
  pristine confirmatory holdout. MovieLens-1M remains unevaluated.
- Published private ALS, deployment traffic, cryptographic aggregation and
  large-data replication are not implemented/completed results.
