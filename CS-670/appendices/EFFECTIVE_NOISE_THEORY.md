# Effective noise in locally personalized BPR

This note derives conditional moments, not a new privacy theorem or a ranking
utility guarantee. Fix the signal-updated factors A∈R^(M×r), B∈R^(r×d), and the
just-updated local user matrix P∈R^(N×d). Let independent Z_A,Z_B have iid
N(0,tau²) entries, where tau=server_lr sigma C/(qN). Then

    D = (A+Z_A)(B+Z_B) - AB = Z_A B + A Z_B + Z_A Z_B.

The cross inner products between the three terms have expectation zero by
independence and centering. Consequently

    E ||D||_F² = tau² [M ||B||_F² + d ||A||_F²] + M d r tau⁴.
    E ||P Dᵀ||_F² = tau² [M ||P Bᵀ||_F² + ||A||_F² ||P||_F²]
                    + M r tau⁴ ||P||_F².

The first expression measures disturbance to the item matrix; the second measures
score disturbance with the local user state held fixed. A small item-matrix
disturbance can still produce large score changes when personalized user norms
are large. Conditioning on P means this is a one-step shock calculation:
past noisy rounds already affected P and the factors.

For two distinct items i,j and a particular local p_u, the pairwise margin shock
has mean zero and variance

    2 tau² ||B p_u||² + tau² ||A_i-A_j||² ||p_u||²
    + 2 r tau⁴ ||p_u||².

The bilinear shock need not be Gaussian. Do not convert this variance into a
Gaussian tail bound for two-factor margin flips. If the clean margin m is nonzero,
Chebyshev gives the weak bound P(|shock|>=|m|) <= min(1, Var(shock)/m²).
This helps explain why raw coordinate noise norm is an incomplete ranking proxy.

With a fixed public B only the first term remains:

    D = Z_A B,
    E ||D||_F² = M tau² ||B||_F²,
    E ||P Dᵀ||_F² = M tau² ||P Bᵀ||_F².

If BBᵀ=I, the map A→AB is an isometry in Frobenius norm, so the effective
noise norm equals the A-noise norm. The two-item margin shock is Gaussian with
variance 2 tau² ||B p_u||². At a fixed nonzero clean margin, its sign-flip
probability is Phi(-|m|/(sqrt(2) tau ||B p_u||)). This describes that immediate
shock, not cumulative ranking performance, candidate-dependent top-K utility or
generalization.

## Scores and capacity with local personalization

For fixed orthonormal B, define v_u=Bp_u. Scores are A_iᵀv_u. Every vector
v_u∈R^r has a preimage p_u=Bᵀv_u; therefore every rank-at-most-r user-item
score matrix remains representable. The irrelevant orthogonal component of p_u
does not affect scores. With B1's effective-row L2, ||A_i B||²=||A_i||² and
the minimum user penalty over preimages is ||v_u||². Local gradient descent
projected through B matches ordinary rank-r BPR, up to float precision.

Jointly learning B retains rank-at-most-r scores but changes the local latent
metric and shared optimization path. FixedB is therefore a useful linear baseline,
not proof that a restriction-free optimizer was obtained. This equivalence also
explains why the method is not a novel public-feature recommender: the public
basis contains no movie information.

## Gauge, clipping and aggregation

The transformation A→cA, B→B/c preserves AB and the coordinate count, but the
linear expected energy becomes

    tau² [M ||B||²/c² + d c² ||A||²].

Its minimum over positive scalar c occurs at c⁴=M||B||²/(d||A||²), not in
general at ||A||=||B||. Equal-norm balancing removes an extreme scale imbalance,
but does not minimize this matrix-size-weighted expression or the score-weighted
expression. Balancing **after** a noisy update preserves its noisy AB; it cannot
remove an already realized shock. The tau⁴ term persists under any scalar gauge.

For scores with P held fixed the minimizing scalar instead satisfies
c⁴=M||PBᵀ||²/(||A||²||P||²). The optimum can differ across private user groups.
Using local P to choose a shared scaling rule would require its own privacy
treatment; these diagnostic formulas do not make P public post-processing.

This synthetic gauge observation must not be read as an optimization prescription
without accounting for signal updates and clipping. Joint factor clipping bounds
||[delta A,delta B]||, whereas FixedB clips ||delta A|| and Full clips ||delta Q||.
An identical C does not imply identical bounds on effective-Q client influence.
Report clipping-only controls and effective signal-step norms, and avoid attributing
all utility changes to the bilinear noise term.

Separate factor averaging also differs from averaging client products:
mean(A_u) mean(B_u) - mean(A_u B_u) is a cross-client covariance term. FixedB
removes that discrepancy as well. E1 does not vary this discrepancy independently,
so it remains a mechanism confound, already identified in
[FFA-LoRA](https://arxiv.org/abs/2403.12313) and
[FedASK](https://arxiv.org/abs/2507.09990).

## Popularity calibration

Let h_u be the binary TRAIN item vector and x_u=h_u min(1,sqrt(20)/||h_u||).
Then ||x_u||<=sqrt(20), so adding/removing an entire user changes sum(x_u)
by at most sqrt(20). One isotropic Gaussian release with std sigma sqrt(20)
uses the exact Gaussian-mechanism condition

    delta = Phi(1/(2 sigma)-epsilon sigma)
            - exp(epsilon) Phi(-1/(2 sigma)-epsilon sigma).

E1 solves this equation numerically following the
[analytic Gaussian mechanism](https://arxiv.org/abs/1805.06530), rather than
using a small-epsilon approximation outside its range. Whole-user clipping matters:
adding iid noise to unbounded raw interaction counts would not yield this guarantee.
Each seed/epsilon run has a per-run guarantee; simultaneous release of all runs
would need additional composition. Population/catalog and hyperparameters are
treated as fixed public inputs, consistent with the project's threat model.

## Rectangular recommendation factors and communication

Unlike a square LoRA block, recommendation factors here are strongly rectangular:
A has 1682r entries, B only 64r. Fixing B reduces the private coordinate count and
dense repeated payload by just 64/(1682+64)=3.67% **relative to Two at the same
rank**. The large reduction relative to a full 1682×64 matrix comes from reducing
rank, which Two already does. This distinction prevents presenting fixed-B
communication savings as the approximate halving reported in a different geometry.

Generating B locally from its public seed can avoid transmitting its entries at
all. If explicitly provisioned as float32, B costs 4dr bytes per client once;
report this separate setup volume, and never include local user vectors in upload.
