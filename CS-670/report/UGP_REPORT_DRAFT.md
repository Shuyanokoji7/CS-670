# Privacy, representation and personalization in federated recommendation

UGP report content draft for CS-670. Updated 3 October 2026. This document
covers completed experiments and an explicitly exploratory follow-up. It is
not a claim of a completed publishable contribution. University front matter
and prescribed formatting have not been supplied.

## Abstract

This project investigates how user-level differential privacy, factorized item
representations and persistent local personalization interact in federated
recommendation. A reproducible MovieLens-100K benchmark compares centralized
Bayesian Personalized Ranking (BPR), federated full-matrix models, and low-rank
models with and without private aggregation. Low rank improves over the private
full-matrix baseline in four of sixteen adjusted historical comparisons, but
does not provide a general advantage. A separately frozen extension compares
training both item factors with fixing one public factor under equal tuning
budgets. The fixed-factor method has no positive adjusted primary comparison.
A user-level private popularity baseline exceeds every collaborative point
estimate at epsilon at most four in this extension. Effective-matrix and score
diagnostics show why coordinate-noise norm alone is inadequate. A subsequent
160-probe exploratory study finds modest persistent local-state disturbance
and a substantial dependence of paired trajectory measurements on factor-basis
noise coupling. Thirty transferred MovieLens-1M training runs and 200 further
probes reproduce the coupling effect and show its dependence on rebalancing.
A separate 3200-pair audit rejects practically consequential non-Gaussian
ranking tails in the sampled states. These findings refine the research
question; they do not yet establish a new algorithm or novelty sufficient
for publication.

## 1. Motivation and research questions

Federation keeps user interactions on clients while sharing model updates.
It does not by itself provide a formal privacy guarantee. This project applies
whole-user clipping and Gaussian perturbation to the aggregate shared update,
while keeping user embeddings local. The practical question is whether useful
personalization survives under that noise and a limited communication budget.

The initial hypothesis was that a smaller shared representation could improve
the privacy–utility tradeoff. For an item matrix Q=AB, however, independent
factor noise changes the effective matrix by

\[
(A+Z_A)(B+Z_B)-AB=AZ_B+Z_A B+Z_A Z_B.
\]

Thus fewer perturbed coordinates do not by themselves determine matrix or
prediction disturbance. The project progressed from baseline comparison to
controlled parameterization experiments and then to a diagnostic study of
local-state propagation. The broader objective remains flexible: establish a
specific, reproducible contribution in private federated recommendation,
rather than defend one predetermined method.

## 2. Related work and contribution boundary

BPR provides the pairwise ranking objective, and federated averaging provides
the shared-update framework. See [Rendle et al.](https://arxiv.org/abs/1205.2618)
and [McMahan et al.](https://proceedings.mlr.press/v54/mcmahan17a.html).
Private recommendation and local/global personalization have substantial prior
work, including [private ALS](https://proceedings.mlr.press/v139/chien21a.html),
[federated private MF](https://www.vldb.org/pvldb/vol15/p900-li.pdf) and
[personalization under joint user-level DP](https://proceedings.mlr.press/v162/bietti22a.html).
The present simulator does not replace comparison with those published methods.

Fixing a low-rank factor and reducing factor-noise amplification are established
in [FFA-LoRA](https://arxiv.org/abs/2403.12313) and
[FedASK](https://arxiv.org/abs/2507.09990). Public item features also have
existing private-recommendation methods, including
[Curmei et al.](https://arxiv.org/abs/2309.11516). These ideas are useful
baselines, not sufficient novelty claims for this project.

The latest search identified close work on private low-rank geometry
([PRISM](https://arxiv.org/abs/2606.00944)), factor alignment
([FLoRG](https://arxiv.org/abs/2602.17095)) and propagation of earlier noise
through later gradients ([NoiseCurve](https://arxiv.org/abs/2510.05416)).
Consequently, a broad claim about understanding noise in model weights would
overlap existing research. A narrower candidate concerns the validity and
usefulness of noise-attribution measurements for persistent personalized
ranking. Its novelty remains unconfirmed. The
[prior-art audit](../references/PRIOR_ART_AUDIT.md) records reading depth and
overlap, and the [bibliography](../references/references.bib) supplies citations.

## 3. Dataset, model and evaluation

MovieLens-100K contains 100000 ratings from 943 users over 1682 items.
Ratings at least four become positive interactions. Retaining users with at
least three positives gives 942 users. Chronological leave-two-out splitting
produces 53491 training positives, 942 validation targets and 942 test targets.
Equal timestamps use a fixed hashed tie breaker. The raw data and final split
have recorded SHA-256 fingerprints. Dataset provenance is described by
[Harper and Konstan](https://files.grouplens.org/papers/harper-tiis2015.pdf);
the exact project split is in the [dataset appendix](../appendices/DATASET_REPORT.md).

Scores have the form s_ui=p_u^T q_i. Full models use 64-dimensional item and
user vectors. Factorized models use A with 1682 rows and r columns, and B
with r rows and 64 columns, for r in {4,8,16,32}. Federated users retain their
own 64-dimensional p_u. BPR encourages an observed positive item to score
above a sampled training negative. Low-rank factor balancing preserves AB;
it does not remove stochastic perturbation of that product.

Evaluation ranks the full catalog after filtering items rated before the
held-out event, including low-rated items; the target is retained. NDCG@10 is
primary, with HR@10 and MRR@10 secondary. Fourteen test targets have no training
positive interactions and remain included. This small subgroup supports only
descriptive cold-item analysis. Activity cohorts contain 319 low-, 310 medium-
and 313 high-activity users, using training-positive counts.

## 4. Experimental stages

| Stage | Purpose | Shared model and privacy |
|---|---|---|
| B0 | Centralized reference | Full item matrix; no DP |
| B0-UW | Sampling-weight control | Centralized BPR with user-uniform sampling |
| B1 | Federated reference | Full shared item matrix; local users; no DP |
| B2 | Private full-matrix baseline | Whole-user clipped/noised shared updates |
| B3 | Nonprivate low-rank baseline | Both factors trained; local users; no DP |
| B4 | Private low-rank baseline | Both factors jointly clipped and noised |
| E1 | Controlled extension | Full, two-factor, fixed-public-factor and DP popularity |
| E2/E2b | Exploratory mechanism probes | Pulse/reset trajectories and a coupling control |
| E3 | Independent-data and balancing controls | ML-1M transfer, validation utility, 200 probes |
| E4 | Ranking-tail candidate screen | Known conditional law versus Gaussian approximations |

B1/B3 use convergence-based stopping. B2/B4 use a fixed 50-round horizon,
participation q=.1 and clipping bound C=1. Therefore their gap from converged
nonprivate models cannot be attributed wholly to noise. Historical B4 received
more tuning than B2; that limitation motivated a separately named E1 extension.
Historical outputs were preserved rather than replaced.

E1 gives each of nine model units four candidate local/server learning-rate
pairs, with three selection seeds and six noise/clipping levels. Its 648 search
jobs include eleven retained failures. Selected settings yield 270 finite final
states across five seeds; reusing eligible search cells gives 756 unique
training records. Hyperparameters and validation-selected ranks were frozen
before extension test scoring. E1 includes no-noise and clipping-only controls.

E1-FixedB draws an orthonormal public basis independently of interactions and
learns only A privately. E1-Two and FixedB share the same initial effective Q
at a given rank. FixedB is score-equivalent to ordinary rank-r BPR under v=Bp;
it does not contain semantic metadata and does not have a smaller hard score
rank than Two at the same r. Optimization and clipping geometry still differ.

## 5. Privacy and statistical scope

The intended privacy unit is an entire user under add/remove adjacency.
Sampled clients' whole shared updates are clipped, Gaussian noise is added to
their sum, and the result is divided by fixed qN=94.2. Empty rounds are noised
and accounted. At delta=1e-5, target epsilons 8,4,2,1 correspond to achieved
primary PRV bounds approximately 7.9675,3.9639,1.9933,.9901. RDP provides a
separate accounting cross-check. Accountant inputs and exact noise multipliers
are retained in the [methods appendix](../appendices/METHODS_AND_LIMITATIONS.md).

The server is assumed to observe only the noisy aggregate. Cryptographic
secure aggregation is not implemented. The guarantee applies to one fixed-
setting training mechanism; private local checkpoints, diagnostic outputs,
model selection, evaluation and joint release of many runs are outside it.
E2 counterfactuals make no DP release claim.

Final B1–B4 and E1 utility means use five training seeds. Primary paired
comparisons average each user's metric across seeds before 100000 paired
user-bootstrap samples. Bonferroni intervals cover the declared four- or
sixteen-contrast families. They condition on these training runs and this
dataset. Plot error bars show seed standard deviations, not those intervals.
E2/E2b are descriptive pilots with no hypothesis-test significance claim.

## 6. Completed baseline results

The matched three-seed comparison gives test NDCG@10 of .085972 for B0,
.091745 for B0-UW and .087539 for B1. Overall paired intervals in the weighting
decomposition include zero, so these point differences do not establish a
clean causal federation or sampling benefit. The five-seed converged B1 mean
is .089008. B3 means at ranks 4,8,16,32 are .059468,.071010,.076333,.068074;
all four primary adjusted comparisons favor B1.

| Historical model | Epsilon 8 | Epsilon 4 | Epsilon 2 | Epsilon 1 |
|---|---:|---:|---:|---:|
| B2 Full | .052271 | .042169 | .025428 | .011461 |
| B4 r4 | .038544 | .037846 | .029079 | .015976 |
| B4 r8 | .045385 | .042653 | .034485 | .018162 |
| B4 r16 | .043628 | .038274 | .033993 | .016685 |
| B4 r32 | .044022 | .044489 | .036504 | .015390 |

Entries are five-seed test NDCG@10 means. B4 has four positive adjusted primary
comparisons against B2: ranks 8/16/32 at epsilon 2 and rank 8 at epsilon 1.
All ranks lose at epsilon 8; no difference is detected at epsilon 4. This is
conditional evidence, with historical tuning confounds, rather than a general
law that smaller rank tolerates privacy noise better.

Communication also requires care. Rank-8 two-factor exchange costs 111744
bytes per participating client-round versus 861184 for the full model under
the simulated dense float32 upload-plus-download accounting. Yet smaller
payload does not guarantee smaller total traffic to a selected checkpoint:
B3 rank 32 uses mean 180.33 GB versus B1's 93.91 GB. The project has not
completed an equal-total-communication utility study. See
[baseline tables](../results/baselines/) and [Figure 1](../figures/01_nonprivate_controls.png).

## 7. E1: effective noise and a strong popularity comparator

| E1 method, rank selected on validation | Epsilon 8 | Epsilon 4 | Epsilon 2 | Epsilon 1 |
|---|---:|---:|---:|---:|
| Full | .052271 | .042169 | .025428 | .011461 |
| Two | .051158 (r32) | .045807 (r32) | .040841 (r8) | .025136 (r4) |
| FixedB | .048978 (r32) | .038959 (r32) | .025286 (r16) | .011958 (r16) |
| DP popularity | .049548 | .049428 | .049507 | .048389 |

The primary E1 family compares FixedB with Two at matched rank and epsilon,
not the differently selected ranks in this presentation table. Of sixteen
adjusted intervals, zero favor FixedB, eight favor Two and eight include zero.
At epsilon 2 every FixedB rank is worse; rank 8's difference is −.018662
with adjusted interval [−.027695,−.010368]. Fixing a factor was therefore not
an effective standalone improvement in this controlled study.

DP popularity bounds each user's binary training-item contribution to norm
sqrt(20), then releases one analytically calibrated Gaussian-noised sum. Its
epsilon-1 NDCG .048389 retains about 97% of its bounded no-noise value .049885.
Every E1 collaborative point estimate at epsilon at most four is below the
corresponding DP-popularity mean. This is a point-estimate statement, not a
family-wide significance claim. Bounding reweights users, which also explains
why its no-noise reference differs from historical raw popularity (.044292).

For independent factor noise with SD tau, conditional on the signal factors,

\[
\mathbb E\|\Delta Q\|_F^2
=\tau^2(M\|B\|_F^2+d\|A\|_F^2)+Mdr\tau^4.
\]

Expected score disturbance additionally depends on local user geometry.
These moments, detailed in the [theory appendix](../appendices/EFFECTIVE_NOISE_THEORY.md),
are elementary and numerically checked. In selected epsilon-1 Two runs the
bilinear term accounts for only about .2–.3% of expected matrix perturbation
energy. It does not explain the failures by itself. Linear amplification,
clipping and local-state scales require attention. Large finite local vectors
remain included, making raw score RMS particularly sensitive to unstable runs.

Fixing B saves only 3.67% of coordinates relative to Two at the same rank
because the item factor A is much larger. Most payload savings relative to
Full come from low rank itself. These results weaken both the proposed
standalone method and a simple coordinate-count explanation. See
[Figure 3](../figures/03_effective_noise_utility.png) and
[Figure 4](../figures/04_noise_and_clipping.png).

## 8. E2/E2b: exploratory local-state and coupling study

E2 perturbs 30 existing round-50 states for Full, FixedB-r8 and Two-r8, using
two pulse amplitudes and two pulse draws: 120 probes. After one exposure
round, it separately restores shared or local state to an unperturbed reference,
then continues ten rounds with matched future random streams. No held-out
target is scored. A further 40 Two-r8 probes control factor orientation before
coupling future noise. This adaptive E2b follow-up was declared separately.

Immediately after a shared reset, Q is identical but P can differ. The remaining
score difference is exactly dP Q^T. At epsilon 1 and full pulse amplitude,
population top-10 set disagreement ten rounds later is .5212% for Full and
.3747% for FixedB-r8. This supplies evidence of modest persistent local-state
effects in the chosen checkpoint population.

Raw-coordinate Two-r8 disagreement is much larger: 17.8546% at epsilon 1 and
9.3960% at epsilon 2. Orthogonal alignment reduces these paired diagnostic
distances to .4204% and .1083%, respectively. This does not improve either
branch's marginal noise mechanism or demonstrate better relevance. The same
Gaussian arrays in different factor bases induce different effective-matrix
perturbations, so the original distance mixed local dynamics with coupling
choice. A synthetic sign-change example reproduces this issue even when the
initial effective matrices are identical.

All 160 probes completed and are retained. Synthetic correctness checks cover
zero pulses, exact resets, RNG matching, score-decomposition closure and product
preservation. The [results note](../research/E2_RESULTS.md),
[mathematical interpretation](../research/NOISE_ATTRIBUTION_NOTE.md) and
[Figure 6](../figures/06_coupling_control.png) document the evidence and limits.
The aligned coupling is a useful control, not a unique causal definition.

## 9. E3/E4: replication and rejection of a second candidate

E3 applies the same split procedure to MovieLens-1M, retaining 6035 users,
3706 items and 563206 training positives. Full, FixedB-r8 and Two-r8 use the
frozen E1 rates at epsilon 1/2 with five seeds and no retuning. All 30 runs
complete. Full-population validation NDCG@10 means at epsilon 1/2 are
.019533/.024624 for Full, .013661/.016482 for FixedB, .015254/.015440 for Two,
and .026124/.026161 for DP popularity. These are descriptive validation
comparisons. ML-1M test targets remain unscored.

Two hundred predeclared probes compare raw/aligned coupling with/without
SVD balancing during continuation, using 128 fixed observer users per dataset.
For Two-r8 at epsilon 1, shared-reset top-10 disagreement ten rounds later is
18.6641% raw versus .4219% aligned on ML-100K, and 5.2344% versus .0625% on
ML-1M. Disabling balancing gives identical raw/aligned churn for every seed:
means .3438% on ML-100K and .0703% on ML-1M. The epsilon-2 control agrees
qualitatively. The larger effect therefore depends strongly on factor-basis
changes during balancing. This rejects a broad interpretation as severe
intrinsic local-memory instability. Disabling balancing only in continuation
does not evaluate training from scratch without it. Sampled validation
differences are small and do not establish a useful intervention. See
[E3 results](../research/E3_RESULTS.md) and
[Figure 7](../figures/07_balancing_replication.png).

E4 asks whether small bilinear variance can nevertheless hide consequential
non-Gaussian ranking tails. Its conditional margin law follows established
product-normal theory ([Gaunt](https://arxiv.org/abs/2408.04101)); generic
noisy-score ranking stability also has prior work
([Urmian et al.](https://arxiv.org/abs/2609.29453)). A constructed example
shows that variance alone need not characterize tails, but the real-state
screen gives a negative result. Across 50 checkpoints, 16 users per dataset
and four fixed rank pairs, the maximum absolute variance-matched Gaussian
risk error is .000007669. None of the 1600 primary pairs exceeds the declared
.01 threshold. All six synthetic cases pass, including independent noise
simulation checks. No validation/test labels enter E4. This candidate is
rejected for these states; its exact probability calculation is not presented
as a new method or theorem. See [E4 results](../research/E4_RESULTS.md) and
[Figure 9](../figures/09_ranking_tail_screen.png).

## 10. Limitations and next research stage

MovieLens-100K has repeated historical exposure and cannot serve as a pristine
confirmatory holdout. Two related MovieLens datasets, five training seeds and
small cold-item subgroups limit generalization. Equal tuning does not match
optimization speed or effective sensitivity. E2 uses oracle counterfactual access, two pulse
replicates and no relevance evaluation. None of its resets is a proposed
deployable private algorithm.

The next substantial study should test whether a clearly specified diagnostic
predicts useful interventions on independent data, beyond simple clipping
rates and vector norms. It should include alternative couplings and optimizers,
published private recommendation baselines, DP popularity and equal total
communication with recalibrated privacy budgets. MovieLens-1M validation is
now exposed, while its test has not been scored. Public metadata and communication-budget allocation remain
alternative directions if the current candidate lacks a distinct contribution.
The [research decisions](../research/RESEARCH_DECISIONS.md) define advancement
and stop criteria.

## 11. Reproducibility and project outcome

B0–B4 and E1 are frozen historical evidence. The report folder contains curated
final tables, seed-level pilot summaries, nine reproducible figures, methodology,
material iteration history, source hashes and verified references. Superseded
sweeps and full private-state checkpoints remain in the research archive.
Figure generation reads saved tables and does not rescore models. The original
E1 audit records 203 passing tests; later E2/E2b/E3/E4 checks are separately recorded.

The completed work establishes a controlled benchmark and rejects several
overly broad explanations: low rank is not uniformly better, fixing a factor
is insufficient, the bilinear noise term is not dominant in these selected
runs, and paired trajectory distance can depend strongly on factor-coordinate
coupling. E3 localizes that dependence to rebalancing in the tested continuations,
and E4 rejects consequential one-step non-Gaussian ranking risk in its fixed
screen. A publishable contribution remains a research objective, with the
next step determined by evidence rather than commitment to one method.
