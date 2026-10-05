# Mathematical interpretation of the E2 probe

These are elementary identities and local approximations used to interpret
the pilot. They are not claimed as new theorems, privacy guarantees or a
complete theory of nonlinear/clipped BPR training.

## Separate instantaneous noise from training-state propagation

Write scores as S=PQ^T, with private local user matrix P and shared item
matrix Q. For two states differing by dP,dQ,

\[
\Delta S=P\Delta Q^T+\Delta P Q^T+\Delta P\Delta Q^T.
\]

The first term measures the shared-state difference at fixed reference users;
the second the local-state difference at fixed reference items; the third
their interaction. Their squared norms do not add: inner-product cross terms
can be positive or negative. Labeling the three energies as percentages of
causal damage would be incorrect.

E1 conditions on the current signal update and local users to analyze one
fresh Gaussian draw. E2 instead perturbs a state and follows subsequent
training. Even with future shared noise coupled identically, local gradients
can remember an earlier perturbation. Neither experiment alone decomposes the
model's total historical loss of utility into independent noise contributions.

Let p be the flattened persistent local state and w the shared parameters.
Away from clipping boundaries, a local first-order description of a round is

\[
\begin{bmatrix}\Delta p_{t+1}\\\Delta w_{t+1}\end{bmatrix}
\approx
\begin{bmatrix}J_{pp,t}&J_{pw,t}\\J_{wp,t}&J_{ww,t}\end{bmatrix}
\begin{bmatrix}\Delta p_t\\\Delta w_t\end{bmatrix}.
\]

Common fresh additive noise cancels in shared *coordinates* for Full/FixedB;
the Jacobian still contains local updates, participation, aggregation and
clipping derivatives. A shared-only pulse (0,z) creates local response
approximately J_pw z. Resetting shared state then leaves (J_pw z,0), which can
feed back into shared parameters through J_wp in later rounds. This explains
what the reset intervention measures. It provides no contraction bound:
Jacobians vary over rounds; clipping is nonsmooth at its boundary; local SGD
is nonlinear. Those qualifications matter for the large finite local states.

Partial participation localizes the immediate response to selected users.
Population means can therefore obscure larger exposed-user effects. E2
records both cohorts without using held-out relevance labels. This is an
experimental interpretation, not a novel claim that personalization has state.
Related optimization dynamics and personalization appear in
[NoiseCurve](https://arxiv.org/abs/2510.05416) and
[Bietti et al.](https://proceedings.mlr.press/v162/bietti22a.html).

## A null example for factor-coordinate coupling

Suppose Q=AB with A of size M by r and B of size r by d. The factors
(-A,-B) represent exactly the same Q. Let ZA,ZB be independent matrices of
iid N(0,tau^2) entries. Under identical raw arrays,

\[
Q_1=(A+Z_A)(B+Z_B),\qquad
Q_2=(-A+Z_A)(-B+Z_B).
\]

Both constructions have the same marginal distribution, but their paired
difference is

\[
Q_1-Q_2=2(AZ_B+Z_A B),
\]

and thus

\[
\mathbb E\|Q_1-Q_2\|_F^2
=4\tau^2\{M\|B\|_F^2+d\|A\|_F^2\}.
\]

For fixed P, the corresponding expected squared score distance is

\[
4\tau^2\{M\|PB^T\|_F^2+\|A\|_F^2\|P\|_F^2\}.
\]

Instead couple the second state with (-ZA,-ZB). It then produces exactly Q1,
while retaining the same marginal isotropic Gaussian law. Therefore a paired
distance may be positive for identical initial effective states, purely
because of the chosen joint coupling. The bilinear ZA ZB terms cancel in this
example; it does not rely on bilinear amplification.

The [synthetic script](../reproducibility/coupling_null_example.py) checks both
identities using 10000 draws and no user data. Its
[output](../results/noise_memory/coupling_null_example.json) reports empirical
means and Monte Carlo SE. This idealized noise-step example explains a
possible confound; it is not a quantitative model of every E2 trajectory.

## Why orthogonal alignment is a suitable control, with limits

For an orthogonal R, (AR,R^T B) preserves Q. Joint Euclidean clipping and
isotropic noise are invariant under this transformation, and scalar SGD on
an effective-Q loss is equivariant in exact arithmetic. Rotations chosen from
past states before the next fresh draw can therefore change the cross-branch
coupling without changing either branch's conditional Gaussian mechanism.

E2b solves the ordinary orthogonal Procrustes problem against the reference
pre-round factors. This controls a representational nuisance. It does not
optimize transport distance, prove a unique counterfactual, or produce a
deployable denoiser: the reference itself uses unavailable private states.
Nonorthogonal rescaling is different; it generally changes clipping/noise
geometry and need not preserve the marginal mechanism.

Alignment and gauge-aware low-rank learning already appear in
[FLoRG](https://arxiv.org/abs/2602.17095),
[FedRot-LoRA](https://arxiv.org/abs/2602.23638) and
[PRISM](https://arxiv.org/abs/2606.00944). The potentially useful research
question is whether an explicitly coupled, personalized-ranking diagnostic
adds reproducible and actionable evidence beyond these established tools.
