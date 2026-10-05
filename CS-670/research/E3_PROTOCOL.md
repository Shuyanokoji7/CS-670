# E3: independent-data and balancing-control replication

Declared 2026-10-03 before preprocessing outcomes, training or E3 probes.
This is a bounded diagnostic continuation of E2. The research direction remains
open. Replicating an effect is not itself a paper-level novelty claim.

## Data and privacy scope

Download the official GroupLens MovieLens-1M archive, verify its published MD5
and record SHA-256 for the archive and extracted inputs. Keep data outside the
report folder, with its original research-use license. Reuse the frozen pure
split functions: rating >=4, at least three positives per user, chronological
leave-two-out, hashed timestamp ties with split seed 2026, catalog consisting
of items appearing in raw ratings. Save to a separately named extension path.
Do not modify MovieLens-100K or the canonical preprocessing/evaluator code.

No test target is scored. Validation is used for descriptive relevance changes,
with the canonical candidate exclusion and pessimistic tie rules. It is not
used to change the settings below. Epsilon labels apply only to the initial
50-round private training runs. Counterfactual continuations and private-state
diagnostics are not DP releases; their extra rounds do not retain that epsilon.

## Frozen training transfer

Train Full, FixedB-r8 and Two-r8 on ML-1M at targets epsilon 1 and 2, for seeds
42/123/2026/7/99: thirty runs. Transfer E1 settings without tuning: dimension64,
rank8, init_std .01, public basis seed314159, q=.1, local_lr5, local_epochs2,
reg1e-5, C1, T50. Full/FixedB server_lr1; Two server_lr.5. Use the exact E1
sigma at the same q,T,delta (2.97749023438 and 1.76044921875); verify against
the stored accounting rows. Training uses the unchanged E1 update code.
Keep every finite/nonfinite result. No replacement seeds or rate fallback.

Record one full-population validation evaluation of each completed state, using
float64 score products and the unchanged ranking function. Include the fixed
sqrt(20)-bounded user-level DP-popularity baseline at both levels and all five
seeds. No selection or claimed significance against popularity is planned.

## Primary diagnostic grid

Use Two-r8 round50 states from both datasets, both epsilon levels and all five
seeds. At each state use two pulse draws, alpha1 only. Compare raw-coordinate
and orthogonally aligned common-noise coupling, each with either:

- original SVD balancing during the diagnostic continuation; or
- balancing disabled for the pulse and continuation, starting from the SAME
  balanced checkpoint. This isolates continuation sensitivity; it is not an
  independently trained optimizer or an unchanged marginal mechanism versus
  the balanced variant.

Total: 2 datasets x 2 levels x 5 seeds x 2 replicates x 2 coupling modes x
2 balancing modes = 160 Two probes. Add forty secondary ML-1M Full/FixedB
probes (2 methods x 2 levels x 5 seeds x 2 replicates), using their native
coordinate coupling. No same-coordinate factor rotation is applied to FixedB.

Within each probe, perturb the shared state by independent Gaussian noise with
SD server_lr*sigma*C/(qN). After one exposure round construct retained,
shared-reset and local-reset branches and continue ten rounds. Client,
negative and Gaussian streams match across branches and coupling/balancing
controls for each dataset/seed/replicate. Alignment uses only pre-round states.
Use RNG seed codes 6710/6711/6712/6713 for pulse/clients/negatives/noise,
with dataset code100 or1000, training seed and replicate appended.

## Observations and uncertainty

Fix a uniform sample of 128 user indices per dataset with public RNG seed6708
and the dataset code before training outputs. These are diagnostic observer
users; ALL eligible users still participate in federated updates. Scores use
float64 products. Measure pulse lag-1, before-reset lag0, and after-reset
lags0,1,4,10. Record sampled-user score RMS, top-10 set disagreement excluding
training positives, and validation NDCG@10 for reference and branch. Validation
uses all previously rated items before its target, including low ratings.
Record exposed/unexposed observer cohorts separately and global P/Q norms.

The 128-user sample gives limited relevance resolution and cannot establish a
small population utility effect. First average two draws within training seed,
then report five-seed means, median, SD and range. Do not treat users, branches,
rounds or paired replicates as independent training runs. No significance test
or post-hoc sample expansion is planned. A finite-sample change in mean utility
under different couplings does not disprove equal marginal laws; theory and
sampling error must guide interpretation.

## Checks, failure policy and advancement

Before real runs check zero-pulse equality, exact both-reset continuation,
RNG matching, product-preserving orthogonal alignment, exact shared/local resets,
and reuse of the original round under the balanced setting. A synthetic check
uses each balancing/coupling combination. Freeze protocol/code/data hashes
before real outputs. Hash every starting checkpoint. Retain failures and any
partial observations, without pretending they are completed probes.

If the gap only occurs with balancing, report sensitivity to factor-basis
changes, not general local-state instability. If it fails to reproduce on the
larger dataset, stop the broad generalization claim. Even successful replication
must show a useful scientific consequence beyond fixing our own diagnostic;
factor alignment, invariant geometry and common random numbers are established.
The distinctness audit and alternative research candidates continue alongside
this bounded experiment.
