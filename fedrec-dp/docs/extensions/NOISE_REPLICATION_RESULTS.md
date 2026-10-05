# E3: independent-data replication and balancing control

Completed 2026-10-03. Frozen protocol: `NOISE_REPLICATION_PROTOCOL.md`.
Source freeze: 2026-10-03T17:36:48Z. No original baseline or E1/E2 artifact
was changed. This study tests a diagnostic interpretation; it does not propose
a new optimizer or claim a deployable private intervention.

## Data, execution and checks

The official GroupLens MovieLens-1M archive passed its published MD5 check
(`c4d9eecfca2ab87c1945afe126590906`) and ZIP CRC check. The unchanged split
functions retain 6035 users, 3706 items and 563206 training positives, with 6035
validation and 6035 test targets. Split fingerprint:
`e68ef765cc661f5db1864f9f8b2843bee70c7ad261c116421d2c7102ec71bd54`.
The item catalog comprises items appearing in ratings. Raw and processed files
are local research inputs, excluded from the curated report. Dataset provenance:
[GroupLens archive](https://files.grouplens.org/datasets/movielens/ml-1m.zip),
[original README](https://files.grouplens.org/datasets/movielens/ml-1m-README.txt),
[Harper and Konstan](https://files.grouplens.org/papers/harper-tiis2015.pdf).

All 30 transferred training runs and 200 diagnostic probes completed with no
failures. Twelve synthetic cases passed, including balanced-round identity,
zero-pulse and both-reset equality, matching RNG streams and exact reset
identities. Maximum observed relative product error from alignment was
3.9126854174e-8. The saved-evidence audit recomputed every seed summary from
2800 raw observations and all 30 validation means from 181050 saved user ranks.

ML-1M uses the E1-selected settings without tuning: 50 rounds, q=.1, C=1,
local_lr 5, E=2, dimension 64, rank 8; server_lr 1 for Full/FixedB and .5 for Two.
The accountant depends on q,T,sigma,delta, not N, so the same E1 noise
multipliers apply at the same target epsilon. Division by qN changes the
per-coordinate update SD on the larger population. Delta remains 1e-5.

## Transferred validation utility

Full-population validation NDCG@10, mean ± standard deviation over five
training/noise seeds. These are point estimates, without a superiority test.

| Method | Target epsilon 1 | Target epsilon 2 |
|---|---:|---:|
| Full | .019533 ± .004151 | .024624 ± .001777 |
| FixedB-r8 | .013661 ± .004574 | .016482 ± .004476 |
| Two-r8 | .015254 ± .005194 | .015440 ± .004567 |
| DP popularity | .026124 ± .000168 | .026161 ± .000104 |

All collaborative means remain below DP popularity. This does not establish
that a tuned collaborative method cannot win, nor compare model convergence:
settings and the 50-round horizon were transferred without retuning. The
ML-1M test targets were not scored. Validation results are now exposed.
Both datasets come from MovieLens; user/catalog overlap was not audited.
A second dataset does not establish independence of sampled populations or
generalization to a different recommendation service.

## Main diagnostic result

The table reports top-10 set disagreement ten rounds after restoring the
shared state. The remaining perturbation at the reset is in local P. Each
entry first averages two pulse replicates within checkpoint seed, then five
seeds. The observer sample has 128 fixed users per dataset; ALL users remain
eligible for federated updates. Values below are percentages of top-10 entries,
not percentages of affected users or relevance loss.

| Dataset | Epsilon | SVD, raw | SVD, aligned | No balancing, raw | No balancing, aligned |
|---|---:|---:|---:|---:|---:|
| ML-100K | 1 | 18.6641% | .4219% | .3438% | .3438% |
| ML-100K | 2 | 5.2031% | .0859% | .0781% | .0781% |
| ML-1M | 1 | 5.2344% | .0625% | .0703% | .0703% |
| ML-1M | 2 | 3.3438% | .0078% | .0156% | .0156% |

For the SVD/raw cells, seed SDs are 3.6991, 5.1075, 4.1984, 2.5386 percentage
points, respectively. Keep this considerable variation visible. Raw versus
aligned churn is identical at the saved precision for every seed without
balancing. Their score distances differ slightly through float32 arithmetic.

The new control **supports a dependence on balancing-induced factor-basis
changes**, rather than a large general failure of persistent personalization.
E2 and E3 use different declared diagnostic random streams and observer
populations; differences between their numerical means are not failed reruns.
ML-1M Full/FixedB epsilon 1 shared-reset disagreement is .1328%/.0859%.
Aligned Two-r8 on ML-1M retains score RMS about 3.6% of the initial pulse after
ten rounds, with very small observed top-10 disturbance.

Balancing is disabled only during the diagnostic continuation, starting from
the same balanced checkpoint. This is not an independently trained model
that never balances. Raw and aligned coupling specify different joint laws;
neither uniquely identifies the causal effect of an individual noise draw.
The alignment is an oracle control, not a way to obtain better marginal
accuracy. Removing balancing changes the continuation algorithm itself.

## Relevance and decision

In the 128-user sample, mean validation NDCG changes for SVD/aligned Two at
epsilon 1/2 are .000040/.000226 on ML-100K and .000018/0 on ML-1M. With no
balancing they are 0/.000027 and 0/0. These small, discrete estimates have
limited resolution and do not establish either equivalence or improvement.
SVD/raw differences have mixed signs and large seed variation. No tuning or
test evaluation followed these measurements.

The predeclared broad-instability interpretation is rejected. A narrower
diagnostic validity result survives independent-data and balancing controls.
Its paper value still requires evidence of a scientific consequence beyond
our own measurement setup, and comparison with established factor alignment,
gauge-aware optimization and stochastic coupling literature. More pulse
sweeps alone will not meet that requirement.

Results are in `results/extensions/noise_replication_v1/`: the compact endpoint
tables support the report, while original JSON, observations and validation
ranks remain in the archive. Counterfactuals and private-state diagnostics
are not DP releases. Epsilon denotes only the starting training run.
