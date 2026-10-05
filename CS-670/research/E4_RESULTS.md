# E4: ranking-tail candidate did not pass the advancement gate

Completed 2026-10-03, under `RANKING_TAIL_PROTOCOL.md`, frozen at
2026-10-03T17:49:38Z. No model was trained and no validation/test target was
read or scored. All 50 declared states, 3200 pairs and six synthetic cases
completed, with no tied, zero-user or numerical-failure records.

## Known probability law, checked in this setting

At a fixed factor state and fixed user vector, independent Gaussian noise on
both factors makes an item-pair margin a sum of products of Gaussian variables.
The protocol expresses it as a scaled difference of independent noncentral
chi-square variables. This is an application of established distributional
theory, not a new theorem; see [Gaunt](https://arxiv.org/abs/2408.04101).

The six synthetic cases compare numerical integration with 200000 direct noise
draws each, check mean/variance/third cumulant within six Monte Carlo standard
errors, and verify rank-one closed forms, zero-mean symmetry and rotation/
user-scale invariance. The integration code uses a square-root substitution
to avoid the rank-one density singularity. These tests do not constitute a
formal numerical error certificate for all parameter settings.

A predeclared illustration, rank 1 with standardized means a=b=4, has bilinear
variance share 1/33=3.03%. Its true sign-flip probability is .0000633405, versus
.0026743853 under a Gaussian with the same variance, approximately 42.2 times
larger. Thus low average interaction energy alone is not a general theorem
about accurate tail probabilities. This constructed case is not evidence
that our trained models operate in that regime.

## Real-state result

The screen uses 40 ML-100K Two-factor states (ranks 4/8/16/32, epsilon 1/2,
five seeds) and 10 ML-1M Two-r8 states. Sixteen users per dataset and the four
predeclared score-rank pairs (10,11),(1,11),(10,100),(1,100) give 3200 records.
Only training positives are excluded. These are label-free ranking-stability
pairs, not a held-out relevance evaluation.

| Diagnostic | Result |
|---|---:|
| Largest absolute error, variance-matched Gaussian | .00000766859 |
| Largest absolute error, linearized Gaussian | .0000726162 |
| Primary pairs, ranks(10,11) or(1,11) | 1600 |
| Primary fraction with absolute matched error >.01, ML-100K | 0% |
| Same fraction, ML-1M | 0% |
| Largest quadrature error estimate | 2.64884e-9 |
| Largest omitted integration tail mass | 2.00000e-12 |

The largest matched error is under .000767 percentage points. Pairwise
bilinear variance shares are much smaller here than in the illustration:
group means range from about .0011% to .0750%. This is a different quantity
from E1's whole-matrix energy fraction; do not equate the two.

The predeclared gate required >.01 absolute error on at least 5% of primary
pairs in BOTH datasets. It failed decisively in this sample. The more elaborate
conditional law therefore does not justify a new ranking method for these
checkpoints. No extra amplitudes, users, ranks or selectively chosen pairs
were added after inspection. The result is limited to a single fresh noise
draw at round 50; it does not explain all cumulative training dynamics.

## Prior-art and research decision

[Urmian, Liu and Khalil](https://arxiv.org/abs/2609.29453) already study
recommendation stability under noisy scores. Their explicit sub-Gaussian
assumptions differ from the non-Gaussian factor-product law; that observation
does not refute their results. Neither generic margin stability nor the known
product-normal identity supplies standalone novelty here.

Retain this as a documented rejected candidate and a useful check on the
bilinear explanation. Do not describe it as a discovery that non-Gaussian
tails cause our observed utility failures. The source JSON, per-pair table,
seed summaries and numerical checks are archived under
`results/extensions/ranking_tails_v1/`; only compact evidence is curated.
