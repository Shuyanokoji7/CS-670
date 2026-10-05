# E2/E2b: local-state memory and a noise-coupling confound

Completed 2026-10-03 (Asia/Kolkata). Status: exploratory mechanism study on
previously inspected MovieLens-100K checkpoints. No new relevance evaluation,
DP release guarantee, improved recommender or paper-level novelty is claimed.

## Main result

A perturbation of shared item parameters changes persistent local user vectors.
Restoring the shared state therefore leaves a small, measurable prediction
difference. However, the very large delayed differences initially observed for
the two-factor model depended strongly on how future Gaussian draws were paired
across the counterfactual trajectories. They cannot be interpreted as an
intrinsic amount of damage caused by local personalization.

## Design and completeness

E2 used 30 frozen E1 round-50 states: Full, FixedB-r8 and Two-r8; epsilon labels
1 and 2; five training seeds. Each received two independent shared Gaussian
pulses at amplitudes .25 and 1 times its own E1 per-round coordinate noise SD.
One exposure round preceded the intervention. We either retained the pulse,
restored reference shared parameters while keeping changed local P, or restored
reference P while keeping changed shared parameters. All branches continued
for ten rounds using common client, negative-sampling and future noise streams.

All 120 E2 probes completed, producing 1680 observations. After inspecting E2,
a separately declared E2b control repeated all 40 Two-r8 probes, producing
560 observations. It orthogonally aligned each branch to the reference's
pre-round factor basis before applying matched future coordinate noise. There
were no failed or excluded probes. E2b is an adaptive follow-up, not a
preplanned primary contrast. Protocols, code and input hashes are recorded.

The epsilon labels describe the starting checkpoints. Additional diagnostic
rounds do not retain that epsilon for a hypothetical release. Resets use
counterfactual private states and are not deployable denoisers.

## Residual disturbance after restoring shared state

The table uses amplitude 1 and lag 10 after the shared reset. Average the two
pulse replicates within each checkpoint seed, then summarize five seed values.
Top-10 disagreement is 100 times one minus the overlap fraction of the two
sets, averaged over users. It is neither the percentage of users with any
change nor a drop in NDCG. Candidate sets exclude training-positive items only.

| Model / coupling | Starting epsilon | Score RMS difference, mean | Ratio to initial pulse RMS, mean / median | Top-10 disagreement %, mean ± seed SD | Range of seed disagreement % |
|---|---:|---:|---:|---:|---:|
| Full | 1 | .011703 | .105239 / .104054 | .5212 ± .0708 | .4087–.5732 |
| Full | 2 | .004531 | .076944 / .075459 | .3206 ± .0465 | .2707–.3822 |
| FixedB-r8 | 1 | .005825 | .084118 / .082905 | .3747 ± .0588 | .2760–.4246 |
| FixedB-r8 | 2 | .002337 | .060556 / .059305 | .2038 ± .0750 | .1115–.3132 |
| Two-r8, raw coordinates | 1 | 178.264576 | 911.196442 / 1.065232 | 17.8546 ± 12.0456 | .4512–29.4002 |
| Two-r8, raw coordinates | 2 | .081387 | 1.456748 / 2.015653 | 9.3960 ± 6.3167 | .0743–14.3471 |
| Two-r8, aligned coordinates | 1 | .041172 | .225161 / .085824 | .4204 ± .0330 | .3822–.4512 |
| Two-r8, aligned coordinates | 2 | .002367 | .040665 / .040587 | .1083 ± .0376 | .0743–.1645 |

Immediately after the shared reset, the effective item difference is exactly
zero. The residual score difference is therefore exactly the changed local
vectors multiplied by the reference item matrix. Ten rounds later, the
aligned/Full/FixedB effects remain concentrated in users selected during the
exposure round. At epsilon 1, mean exposed-user disagreement is 4.7166% for
Full, 3.2102% for FixedB-r8 and 3.4678% for aligned Two-r8; the corresponding
unexposed-user means are .0154%, .0333% and .0560%.

Reducing pulse amplitude from 1 to .25 reduces the lag-10 shared-reset score
RMS by approximately a factor of four for Full, FixedB and aligned Two. This
is consistent with a locally linear response in these settings. Two amplitudes
do not establish a general scaling law. Differences in models' own noise SD,
server step sizes and learned scales prevent a causal ranking of architectures
from this table alone.

## Why the control changes the conclusion

For any orthogonal R, (AR)(R^T B)=AB. Raw identical Gaussian arrays added in
two different factor bases need not induce similar changes in AB. SVD balancing
can choose different signs or rotations after small state changes. Thus common
random numbers in factor coordinates need not represent a matched effective
perturbation. Full and FixedB have no analogous changing factor basis here.

E2b changes the *joint coupling* between diagnostic branches. Under the scalar
SGD, Euclidean clipping and isotropic Gaussian mechanism used here, orthogonal
transformation preserves each branch's marginal mechanism in exact arithmetic.
It does not reduce its marginal noise variance. The maximum measured relative
change in AB caused by alignment across the real probes was 3.91326e-8. The roughly 42-fold and 87-fold
reductions in paired shared-reset disagreement are therefore evidence about
the diagnostic, not improvements in recommendation quality or privacy.

The aligned coupling is a controlled alternative, not a uniquely correct
counterfactual or a proven optimal coupling. Residual coupled distances depend
on the intervention and common-random-number convention. Finite precision is
also present. Alignment does not eliminate every unstable trajectory: the
aligned Two epsilon-1 score-ratio mean .2252 versus median .0858 shows remaining
scale sensitivity. Report the seed-level values and bounded ranking metric,
not just a large raw mean. All outcomes remain in the archive.

## Correctness and evidence boundary

Six E2 and two E2b synthetic check cases verify zero-pulse equality, both-reset
equality through ten rounds, shared-reset matrix equality, matched RNG streams
and the exact score decomposition, including its cross terms. Real runs check
checkpoint hashes, finiteness, decomposition closure, reset equality and RNG
matching. The both-reset branch is continued through all rounds in the
synthetic checks; it is checked at construction in the real probes.

A separate post-pilot synthetic null example starts with (A,B) and (-A,-B),
which have identical AB. Raw matched noise creates a positive expected paired
matrix distance; sign-transformed noise gives exactly identical outputs. In
10000 float64 draws, expected/observed squared distance was
6.345313/6.341142 (Monte Carlo SE .012443), and aligned distance was exactly
zero. This verifies elementary algebra and supplies no additional novelty.
The portable script and its output are in CS-670/reproducibility and
CS-670/results/noise_memory respectively.

No held-out targets were scored and no B0–B4/E1 training or result was changed.
This pilot uses only five existing training seeds and two pulse replicates per
state, and it shares random streams between amplitudes. It supplies descriptive
mechanistic evidence, not independent confirmatory samples or a causal
percentage decomposition of the final model's error.

## Prior art and next decision

Noise propagation through optimization is established in
[NoiseCurve](https://arxiv.org/abs/2510.05416); local/global private
personalization in [Bietti et al.](https://proceedings.mlr.press/v162/bietti22a.html);
factor alignment in [FLoRG](https://arxiv.org/abs/2602.17095) and
[FedRot-LoRA](https://arxiv.org/abs/2602.23638); and private low-rank gauge
geometry in [PRISM](https://arxiv.org/abs/2606.00944). Neither alignment nor
persistent optimization effects are claimed as inventions.

The remaining candidate is narrower: an audit of how noise-attribution
measurements interact with local personalization and ranking. Exact prior-art
coverage is not settled. Advance only with an independent dataset, alternative
couplings/optimizers, and evidence that the diagnostic predicts useful,
deployable interventions beyond simple norms and clipping rates. If local
effects remain small or the same study already exists, redirect. The research
direction remains flexible.

Canonical evidence: results/extensions/noise_memory_v1 and
results/extensions/noise_memory_aligned_v1. The curated report contains
aggregate tables, seed tables, inventories and reproducible figures; full
per-probe observations remain in the archive.
