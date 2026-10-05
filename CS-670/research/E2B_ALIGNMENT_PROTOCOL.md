# E2b: factor-coordinate coupling control

Declared 2026-10-03 after inspecting E2, before E2b outputs. This is a disclosed
adaptive diagnostic follow-up; it is not a new primary hypothesis test.

E2 Two-r8 trajectories show much larger delayed divergence than Full/FixedB.
Balancing can select different factor rotations after small changes. Matching
future Gaussian arrays in raw coordinates therefore need not match effective
matrix perturbations. This can inflate a paired trajectory-distance diagnostic
without increasing either branch's marginal noise variance.

Repeat the 40 E2 Two-r8 probes, keeping all checkpoints, streams, amplitudes,
reset rules and observations. Before each coupled round, orthogonally align
each branch's factors to the reference's PRE-ROUND factors. Solve

    min_R ||A_branch R - A_ref||_F^2 + ||R^T B_branch - B_ref||_F^2,
    R^T R = I.

If U S V^T = SVD(A_branch^T A_ref + B_branch B_ref^T), use R=U V^T,
then A_branch <- A_branch R, B_branch <- R^T B_branch. This leaves AB
unchanged in exact arithmetic. Identity states bypass the SVD. Local scalar SGD,
whole-factor Euclidean clipping, and independent isotropic Gaussian noise are
orthogonally equivariant in exact arithmetic. Thus this changes the cross-branch
noise coupling, rather than giving a variance-reduced marginal mechanism.
Alignment is chosen before the current round's fresh noise. Numerical product
preservation, orthogonality, zero-pulse equality and RNG matching are checked.

Use the frozen E2 observer; record the maximum relative product-preservation
error and rotation size. Compare paired distances under the two couplings,
especially after the shared reset. No utility/relevance metric is introduced.
No branch is selected for being stable or favorable; retain every outcome.

Interpretation must distinguish this diagnostic coupling from a deployable
algorithm. The reference contains counterfactual/private states. Alignment is
also prior art: FLoRG (https://arxiv.org/abs/2602.17095) and FedRot-LoRA
(https://arxiv.org/abs/2602.23638) address rotations in federated low-rank models;
PRISM (https://arxiv.org/abs/2606.00944) addresses gauge-dependent private LoRA.
The narrower possible contribution is a careful audit of intervention-based
noise attribution in personalized recommendation. No first-of-its-kind claim
is justified by this search or this control alone.
