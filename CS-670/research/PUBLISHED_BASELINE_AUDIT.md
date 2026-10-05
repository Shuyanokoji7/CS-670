# Published baseline implementation audit

3 October 2026. This note prepares the next controlled comparison; none of
these published baselines has yet been implemented in this project. The
completed E3/E4 experiments remain frozen. Novelty is not inferred from a
different algorithm name or a missing citation in an earlier search.

## Primary-source findings

[Private ALS](https://proceedings.mlr.press/v139/chien21a/chien21a.pdf)
alternates local user solves with private item solves. Item updates perturb
both a normal-equation matrix and a vector, and bound the number of item
contributions per user. Its practical implementation differs from the
analysis-oriented pseudocode in resampling and orthogonalization. The paper
also addresses item-frequency skew. The main algorithm, privacy section,
practical-modification discussion and
[supplementary privacy proof](https://proceedings.mlr.press/v139/chien21a/chien21a-supp.pdf)
were inspected. A faithful reproduction must specify which variant it uses.

[Private MF with public item features](https://arxiv.org/html/2309.11516)
uses collective factorization of user feedback and public item features.
Private sufficient statistics receive noise, while public-feature statistics
do not. User embeddings remain individually personalized. Its stated trusted
server has raw-data access; secure aggregation is discussed as a possible
extension. It evaluates a rating-prediction task with richer metadata, so
those numbers are not directly comparable with our implicit top-10 ranking.
Algorithm 1, privacy scope and evaluation were inspected.

[Multi-Task DP Under Distribution Skew](https://arxiv.org/pdf/2302.07975)
already studies allocating a user's privacy budget among differently sized
tasks through adaptive weights, with recommendation experiments. Algorithm 1,
Theorem 3.3 and its proof compose the matrix and vector releases. Public
versus privately estimated task counts require separate treatment. These
passages were inspected; the entire convergence analysis was not audited.
This blocks claiming item-frequency-aware privacy allocation as a new idea.

## Requirements for our comparison

1. Derive sensitivity for the exact implemented joint release, including every
   matrix/vector statistic and any popularity/count preprocessing. Do not
   borrow BPR's q=.1 noise multiplier for a different sampling mechanism.
   Matrix storage matters: independent upper-triangle noise is not the same
   coordinate mechanism as independent full-matrix noise followed by averaging.
2. Use our fixed positive threshold and candidate filter. Specify an implicit
   objective and its treatment of unobserved items; changing the objective
   makes this a published-method comparison, not an optimizer-only ablation.
3. Preserve public item catalogs and keep local personalization private.
   Catalog sparsity and user-item routing must not leak through a supposed
   secure-aggregation abstraction. Describe the trust model concretely.
4. Count actual sufficient-statistic uploads and item-factor downloads.
   Fewer ALS rounds alone do not establish a lower communication budget.
5. Freeze tuning limits and the final configuration before using ML-1M test.
   Compare DP popularity and metadata-only local personalization as well as
   collaborative methods. Retain every failed or unstable configuration.

Implementing these methods would strengthen the evidence base. A possible
paper claim must concern a further, demonstrated mechanism or useful tradeoff
that those methods do not already establish. The present audit does not
announce an error in a prior paper or a novel privacy result.
