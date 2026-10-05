# E4: can second-moment noise diagnostics miss ranking risk?

Declared 2026-10-03 before E4 numerical outputs. This is a bounded candidate
screen, not a new algorithm or an established novelty claim. E1/E2/E3 remain
unchanged. No training, validation labels or test labels are used here.

## Motivation and prior-art boundary

Small bilinear noise energy does not logically imply an accurate Gaussian tail
approximation. This is a probability question about a single additional noise
draw at a frozen state, not about cumulative training error. FFA-LoRA already
identifies noise amplification; PRISM studies effective-update geometry.
Gaunt's product-normal distribution work supplies the relevant distributional
identity, so neither that identity nor its application by substitution is a new
probability theorem. Urmian et al. already study recommendation stability under
noisy scores and explicit sub-Gaussian assumptions. Our bilinear setting has a
different conditional distribution. That does not invalidate their assumptions.

Primary sources checked:
- https://arxiv.org/abs/2403.12313
- https://arxiv.org/abs/2606.00944
- https://arxiv.org/abs/2408.04101
- https://arxiv.org/abs/2609.29453

## Conditional law to verify

Write v=A_i-A_j, u=Bp, sx=sqrt(2)*tau and sy=tau*||p||, where tau is the
actual per-coordinate update-noise SD after server scaling. With independent
standard normal vectors X,Y, the perturbed pair margin is

    L=(v+sx X)'(u+sy Y).

For a=v/sx, b=u/sy, r=rank, k=sx^2 sy^2, and m=v'u,

    L =d sqrt(k)/2 * (U-V),
    U ~ noncentral-chi-square(r, ||a+b||^2/2),
    V ~ noncentral-chi-square(r, ||a-b||^2/2), independently.

The exact conditional flip probability for m>0 is P(U<=V). Its numerical
evaluation will integrate the known noncentral-chi-square CDF against the
other density. Report quadrature error estimates and omitted tail mass;
these are numerical diagnostics, not a formal floating-point certificate.

    Vlin = sx^2 ||u||^2 + sy^2 ||v||^2
    Var(L-m) = Vlin + r*k
    third cumulant(L-m) = 6*k*m

Comparators: the linearized Gaussian risk Phi(-m/sqrt(Vlin)) and the
variance-matched Gaussian risk Phi(-m/sqrt(Vlin+r*k)). The mean bilinear
energy fraction is r*k/(Vlin+r*k). FixedB and Full single-noise pair margins
are exactly Gaussian conditional on a fixed user vector and deterministic
state; this statement does not cover subsequent local adaptation.

Multiplying p by a positive scalar leaves ranking and this risk invariant.
Use normalized p for numerical stability, retain its original norm separately.
This elementary scale invariance is not a novelty claim.

## Frozen real-state screen

Use all E1 Two-factor checkpoints at ranks4/8/16/32, epsilon1/2 and five
seeds42/123/2026/7/99 (40 states), plus E3 ML-1M Two-r8 states at both levels
and the same seeds (10 states). The E1 states are already explored. E3 was
trained before this protocol; no E4 rank-risk outputs have been inspected.

Select sixteen observer users per dataset with RNG[6720,dataset_code] (100
or1000). At each state order all items excluding training positives, using
float64 scores and stable index ties. Evaluate these four rank pairs exactly
as specified: (10,11), (1,11), (10,100), (1,100), with one-based ranks.
Total3200 state-user-pair records. No pairs are selected by observed error.
Ties or zero user vectors are flagged and retained rather than silently
replaced. Hash every checkpoint and freeze code/protocol before reading real
states. No post-hoc user, rank, epsilon or noise-amplitude expansion.

Primary screen: absolute probability error of the variance-matched Gaussian
on pairs (10,11) and (1,11). Report mean, maximum, and fraction above .01,
first averaging within state and then displaying all five training seeds.
Secondary: the two wider-margin pairs, linearized risk, bilinear fraction,
and normalized margin. Relative error is descriptive only for exact risk
>=1e-6; huge relative errors at negligible risk do not justify advancement.

## Numerical checks and advancement

Before real states, check zero-mean symmetry, rank-one closed-form sign
probabilities, and independent direct Gaussian Monte Carlo for predeclared
ranks1/4/8 and fixed deterministic vector patterns. Include rank-one a=b=4
as a mathematical illustration selected before computation, not evidence
that fitted recommendation states inhabit that regime. Check second/third
moments, and invariance under orthogonal rotation and positive user scaling.
Use 200000 Monte Carlo draws per synthetic case; MC uncertainty is distinct
from five-training-seed variation. Retain failures; no adaptive MC extension.

Advance only if the realistic near-boundary screen finds consequential,
replicated error (>.01 absolute probability error for at least 5% of primary
pairs in both datasets), with numerical checks passed. This is a pragmatic
research gate, not a statistical significance threshold. If this gate fails,
record a negative result: the rare-tail caveat does not justify a new method
in these checkpoints. A passing result would still require a stronger novelty
audit and a useful ranking decision, not merely a closed-form calculation.

Private states and per-user diagnostics are internal analyses, not DP outputs.
Epsilon labels identify the checkpoint, not a privacy guarantee for this audit.
