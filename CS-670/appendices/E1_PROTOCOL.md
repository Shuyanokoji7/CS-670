# Effective noise study E1: protocol v1

Declared 2026-10-02 before extension implementation or experimental runs. The
user authorized independent pursuit of direction 2. B0–B4 stay historical and
frozen. All new artifacts live under `results/extensions/effective_noise_v1/`
and `checkpoints/extensions/effective_noise_v1/`. The split, evaluator, targets,
candidate construction and BPR loss are unchanged. The previously exposed
MovieLens-100K test is a development benchmark, not a pristine holdout.

## Question and estimands

Does removing noise multiplication improve personalized recommendation under a
fixed user-level privacy guarantee, after controlling tuning budget and measuring
the disturbance in the effective item matrix and predictions?

E1-Full uses Q directly. E1-Two uses Q=AB, one joint clip over both factors,
independent Gaussian coordinate noise and balancing after every server step.
E1-FixedB uses a public, data-independent, orthonormal B (BBᵀ=I), trains/noises
only A, and keeps p_u local in R^64. A fixed latent basis preserves trainable
item representations. A random fixed item basis would constrain item geometry
and answers a different question. Public B uses seed 314159, is generated once
from a 64-dimensional orthogonal matrix and is nested across ranks.

Ranks are 4/8/16/32. For each rank E1-Two and E1-FixedB start with identical P
and identical Q to float tolerance: A entries have standard deviation
0.01 sqrt(64/r), and B is public orthonormal. E1-Two balances its initial factors;
E1-FixedB preserves B. E1-Full keeps the established full-rank initialization.
This new initialization is declared explicitly; E1-Two is not a B4 replication.

**Recommendation-specific caveat:** fixed orthonormal B with unconstrained local
p_u is equivalent in scores to ordinary rank-r matrix factorization with local
v_u=Bp_u. It can represent every rank-r score matrix. Learning B does not confer
a larger hard score-matrix rank at the same r; it changes optimization and
regularization geometry. This is a baseline adaptation, not a new fixed-factor
algorithm. Even matched Q initialization cannot isolate the bilinear term's
causal utility effect from these changes.

## Finite experiment budget and selection

All mechanisms use q=.1, T=50, E=2, effective-row L2=1e-5, fixed qN and C=1.
Epsilon targets 8/4/2/1 reuse the exact stored B2 sigmas only after full-key and
epsilon verification (PRV primary, RDP cross-check, delta=1e-5).

Each of nine model units (one full plus four ranks for each factor method) gets
exactly four candidate (local lr, server lr) pairs: (1.25,1), (2.5,1), (5,.5),
(5,1). No boundary expansion or fallback. Every candidate runs on seeds
42/123/2026 at **all six levels**: unclipped/no noise, C=1/no noise, eps8/4/2/1.
Eligibility requires all 18 jobs to be finite and reach T=50. Select the largest
three-seed **final-round validation NDCG@10 at eps4**; exact ties prefer server
lr nearer 1, local lr nearer 5, then smaller lr. This is 648 bounded tuning jobs.
If a unit has no eligible candidate, report that bottleneck, retaining failures.

Selected settings run all levels on five seeds 42/123/2026/7/99; exact completed
jobs are reused after content/configuration/hash verification. No checkpoint
selection or early stopping. At most 12 one-thread workers; profile first.
Record validation-selected rank for each factor method and level. If any selected
five-seed inventory fails, stop before test and report the bottleneck; do not
invent another grid. Freeze configs, selections, checkpoint hashes and code in
a dated snapshot before any extension test evaluation. Score each retained
checkpoint once using the frozen evaluator, with reuse checks preventing repeat
scoring. Historical test CSVs are read, not regenerated.

## Mechanism diagnostics

Each noisy step is compared with its own **same-state, same-client-update,
noiseless server counterfactual**. These are instantaneous shocks, not a claim
about the total noise accumulated over training. For signal-updated factors
A_s,B_s and server-step noise Z_A,Z_B:

    delta Q = Z_A B_s + A_s Z_B + Z_A Z_B.

Record each term's squared Frobenius norm, total delta-Q norm, clipping fraction,
pre-clip norm, effective signal-step norm and theoretical conditional expected
energies. Full and fixed methods have linear noise only. Use float64 diagnostics
on the actual float32 Gaussian draws. Report float32 addition/balancing residual.
At rounds 1/10/25/50 measure all-user/catalog score RMSE, score relative error,
and pairwise margin RMSE for 256 data-independent item pairs. P is held fixed
at its just-updated local value in both branches; neither branch retrains users.
Report low/medium/high activity (train positives <=22, 23–61, >=62), norm scaling
and train-cold item rows. Local P and these diagnostics are private research
artifacts, not covered DP server outputs. Mean/permuted-P ablations are evaluated
on validation only after training. Cold validation/test groups (6/14 targets)
are descriptive and too small for strong subgroup conclusions.

Synthetic Monte Carlo verifies the expected energy formula and a gauge sweep
(A→cA,B→B/c) shows that coordinate count and AB can stay fixed while noise
energy changes. This analysis is planned independently of utility results.

## User-level DP popularity

E1-DPPop is one full-population release (q=1,T=1), **not** 50-round federated
training. Each user contributes a binary vector over unique TRAIN positives,
scaled to L2 norm at most sqrt(20), fixed before data inspection. Sum those
vectors and add isotropic Gaussian noise with std sigma sqrt(20). Add/remove
whole-user sensitivity is sqrt(20), with public catalog/population conventions.
Use the analytic Gaussian mechanism to solve sigma at the same eps/delta targets;
also report the existing RDP accountant at q=1,T=1 as a looser cross-check.
No private-data-dependent count truncation or epsilon reuse from B2. Five noise
seeds, no tuning, validation first, same freeze/test gate. Keep raw unbounded
popularity as its existing non-private point reference; include the bounded,
no-noise count scorer as a separate control. Scores may be negative; clipping
them to zero is omitted to avoid gratuitous ranking ties.

## Statistics and communication

Primary family: 16 E1-FixedB(r,eps) minus E1-Two(r,eps) NDCG@10 contrasts.
Each user's metric is averaged over five seeds first; 100000 common paired
user-bootstrap draws (seed 2026) give ordinary 95% and simultaneous Bonferroni
99.6875% intervals, conditional on these training runs. Also report five-seed
SD and each seed's difference. Secondary contrasts: fixed versus full and
DPPop, clipping-only versus unclipped, DP versus clipping-only, and selected
ranks. Secondary intervals and groups are exploratory, with no primary claims.
Also read saved frozen B4 per-user CSVs for secondary historical FixedB-versus-B4
contrasts. Different initialization and historical tuning remain confounds; do
not score or retrain B4 to obtain this comparison.
MRR@10 is secondary; HR@10=Recall@10 is one metric. Negative findings remain.

Tensor communication includes dense float32 private parameters each round.
Full: 8Md bytes/client/round. Two: 8r(M+d). FixedB: 8Mr; public seed/basis can
be cached, report explicit initial basis provisioning separately. Report total
bytes over realized participation, not just payload ratios. Same-T comparisons
do not establish an advantage at equal communication budget or at convergence.

## Prior art and claim limits

[FFA-LoRA](https://arxiv.org/abs/2403.12313) already freezes a random factor and
analyzes amplified DP noise. [FedASK](https://arxiv.org/abs/2507.09990) addresses
both-factor noise and aggregation through double sketching. They concern LoRA
fine-tuning rather than local-user BPR; their results/privacy assumptions cannot
be transferred without verification. [From Bilinear to Linear](https://arxiv.org/abs/2609.26091)
also proposes a linear shared parameterization. These establish that linearizing
private factor learning is prior art. [Public item features](https://arxiv.org/abs/2309.11516)
and [personalization under joint DP](https://proceedings.mlr.press/v162/bietti22a.html)
are additional relevant foundations; metadata residuals remain a later method.

Potential contribution: connect effective noise, pairwise margins, clipping,
local personalization and dense tensor communication in a controlled recommender
study. A positive score alone does not establish novelty. MovieLens-1M independent
replication and a communication-budget study remain necessary before a strong
paper claim; this protocol does not silently add either dataset or claim completion.
The final training mechanisms, conditional on fixed hyperparameters and assuming
the server observes only the noised aggregate, are user-level DP. Local checkpoints,
selection, research diagnostics, evaluation and the joint release of all runs are
outside that per-run guarantee. Noise is seeded for reproducibility, not cryptographic.
