# E2: shared-noise and local-state memory diagnostic

Declared 2026-10-03 before any E2 real-data output. This is an exploratory
mechanism probe, not a new DP training algorithm or a confirmatory utility study.
The research direction remains open. A positive probe is not sufficient novelty.

## Question and prior art boundary

Does a perturbation to shared recommendation parameters leave consequential
changes in persistent local user vectors after the shared state is restored?
How much do those local changes subsequently affect shared training and rankings?

Noise-induced trajectory drift is established (NoiseCurve,
https://arxiv.org/abs/2510.05416). Local/global step-size geometry under user-level
joint DP is established (Bietti et al.,
https://proceedings.mlr.press/v162/bietti22a.html). E2 investigates a narrower
intervention in a personalized BPR simulator. Neither the existence of a coupled
dynamical system nor the score decomposition below is claimed as a new theorem.

## Fixed design

- Read E1 final round-50 checkpoints for Full, FixedB rank 8 and Two rank 8,
  at epsilon targets 1 and 2, all five seeds 42/123/2026/7/99. These checkpoint
  choices are exploratory choices informed by completed E1, not an untouched
  holdout selection. Full and FixedB are the primary diagnostic examples;
  Two is secondary because factor balancing makes common-noise coupling depend
  on the implemented factor coordinate convention.
- Preserve each checkpoint's selected local/server rates, clipping, regularizer,
  public basis, local epochs and q. No search or model selection.
- Apply a fresh independent Gaussian pulse to the shared coordinates, with
  standard deviation alpha * server_lr * sigma * C / (q*N). Use alpha in
  {0.25, 1}, two pulse replicates per checkpoint. This is a perturbation of an
  already private state, not replay or removal of an actual E1 noise draw.
- Pulse streams: seed tuple [training_seed, 6702, replicate]. Future clients,
  negatives and Gaussian draws: [training_seed, 6703/6704/6705, replicate].
  Common random numbers couple every branch within a probe and both amplitudes.
- Perform one ordinary local/shared round, allowing local P to react to the
  pulse. Then construct four branches: reference (no initial pulse), retained
  pulse, shared reset (copy reference shared state, keep perturbed P), local
  reset (copy reference P, keep perturbed shared state). Also use a both-reset
  branch as an exact arithmetic/RNG check, not as an experimental method.
- Continue all branches for ten rounds with the same future client samples,
  negative samples and Gaussian coordinate draws. Observe the immediate pulse,
  the exposure round before/after resetting, and 1/4/10 rounds after resetting.
- Total: 3 methods x 2 levels x 5 checkpoint seeds x 2 pulse draws x 2 amplitudes
  = 120 probes. Failed/nonfinite trajectories remain in the inventory. No
  replacement seeds, exclusions, expansion or early stopping.

## Measurements and interpretation

For a branch (P+dP,Q+dQ) and its current reference (P,Q), calculate in float64:

    delta S = P dQ^T + dP Q^T + dP dQ^T.

Record matrix and score RMS differences, all three score-component energies,
their cross inner products, and numerical closure error. Component energies
do not sum to total energy; they are not causal percentages. Immediately after
the shared reset dQ=0, so the remaining score difference is exactly dP Q^T.
Later shared differences in this branch can arise from its changed local state.

Report top-10 set disagreement on TRAIN-unseen catalog items (deterministic item
index tie breaking), and sign changes on 256 public random distinct item pairs.
These measure prediction stability, not relevance or NDCG. No validation or test
target is evaluated. Report exposed users separately from others, and training
activity cohorts using E1 thresholds. Local P norms and finiteness are retained.

Normalize delayed score differences by the immediate pulse score RMS within
each probe; report absolute differences too. A larger ratio alone can reflect
a smaller initial denominator. Reset branches are interventions, not an additive
causal partition and not deployable denoisers. Future noise cancellation is
exact in Full/FixedB shared coordinates, but differs in effective Q for Two.

Average the two pulse replicates within each checkpoint seed before summarizing
across the five seeds. Report mean, median, range and seed SD; do not turn 120
correlated probes or individual users into 120 independent training runs.
No hypothesis-test significance claim is planned for this screening pilot.

## Correctness and privacy

Checkpoint hashes must match E1 final records. Reuse original local-update and
aggregation code. Check zero-pulse branch equality, both-reset equality, shared
reset dQ=0, score-decomposition closure, and identical RNG states across branches.
Checks use synthetic data before the real probe. The original E1 and B0-B4
outputs and test scores are never rewritten.

The starting E1 mechanism has its documented per-run DP guarantee. The E2
counterfactual states, private local P, unnoised differences and diagnostic
outputs are research artifacts outside that guarantee. Continuing with E1 sigma
does not retain the original epsilon after additional releases. E2 makes no
privacy or utility claim for the intervention branches; it does not propose
releasing them. A deployable intervention would need fresh privacy accounting.

## Advancement gate

Continue this candidate only if the local-state branch has measurable prediction
consequences that reproduce across checkpoint seeds. Even then, require a
larger independent dataset, sensitivity to local step size and participation,
an intervention achievable without counterfactual access, and comparison with
existing personalization/denoising methods before claiming a paper contribution.
If effects are negligible or mainly numerical instability, retain the negative
result and reconsider other candidates. The report must distinguish this pilot
from completed E1 hypothesis tests.
