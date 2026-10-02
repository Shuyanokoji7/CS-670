# Research Log

Append-only. Each entry records decisions *before* results that could motivate
them are seen. Changing a frozen decision requires a new dated entry with the reason.

---

## 2026-09-28 — Phase 0: research foundation

> Superseded in part by the entry **"Phase 0b"** below: the tie-break, candidate filtering and
> baseline numbers in this entry are no longer current.

### Research Question

Can reducing the dimensionality of privacy-sensitive shared representations improve the
privacy–utility–communication trade-off in differentially private federated recommender
systems?

### Hypothesis

**H1:** Under the same formal DP guarantee, a lower-dimensional shared representation may
outperform a full-dimensional one in sufficiently noisy privacy regimes.

H1 is **falsified** (on this benchmark) if, at every tested privacy level, the full-dimensional
shared representation matches or beats every lower dimension on validation-selected, test-reported
NDCG@10 beyond seed-to-seed variability. The comparison must be at an **identical (ε, δ)**, computed
by the same accountant with the same clipping, sampling, rounds and noise multiplier. Dimension is
not a privacy knob under the Gaussian mechanism, so any "privacy gain" from lower dimension
would have to show up as utility at fixed ε, not as a smaller ε.

### Threat Model

See `docs/THREAT_MODEL.md`. Summary: one user = one client. Raw history `D_u` and user vector
`p_u` stay on-device. Honest-but-curious server. **User-level DP** (neighbouring datasets differ
by one user's entire contribution). Secure aggregation is **assumed/simulated, not
cryptographically implemented**.

### Dataset Decision

MovieLens-100K (official GroupLens release, `u.data` SHA-256 pinned in `src/data.py`).

- Stable, fixed release, so results are comparable across the project and with the literature.
- Small (943 users × 1682 items): full-ranking evaluation over the whole catalog is cheap, and
  many DP/federated configurations × seeds can be swept.
- Has timestamps, so a chronological split is possible.
- `ml-latest-small` is not used: it is a periodically regenerated snapshot, not a frozen benchmark.
- MovieLens-1M is registered in the loader and reserved for a later generalisation check. It is not
  used now.

### Recommendation Formulation

Implicit-feedback Top-K recommendation, not rating prediction (RMSE). The federated recommender
setting and the downstream utility we care about is *which items to show*. Ranking metrics measure
that directly, and RMSE improvements are known to correlate poorly with Top-K quality. BPR-style
pairwise/implicit models (planned) are the standard fit for this setting.

### Positive Interaction Definition

rating ≥ 4 is a positive. Ratings 1–3 are **not** positives, and they are **not** treated as explicit
negatives. They are simply not positive signal. Rated-but-low items remain in the candidate set at
evaluation time. Result: 55,375 positives out of 100,000 ratings.

### Split Strategy

Per-user chronological leave-two-out over positives: latest positive → test, second-latest →
validation, all earlier → train. Ordering key is `(timestamp, raw item id)`. The item-id
component is a deterministic tie-break only.

- Minimum 3 positives per user (the structural minimum for 1/1/1). A higher minimum was
  considered and **not** adopted: it would remove light users, who are exactly the ones most
  affected by DP noise, and it would bias the study toward heavy users. Only 1 user (raw id 685,
  0 positives) is removed. 2 retained users have a single train positive.
- No item filtering. All 1682 catalog items are candidates. 17 test and 7 validation targets have
  no training positive (cold items). They are kept, because removing them would make the task easier.
- Hyperparameters and dimensionality choices must be selected on **validation only**. Test is
  reported once per final configuration.

### Evaluation Decision

Primary metric: **full-ranking NDCG@10**.

- Full ranking: the held-out item is ranked against every catalog item the user has not positively
  interacted with in training (test additionally excludes the user's validation item, a known past
  positive at test time). "1 positive + 99 sampled negatives" is not used. Sampled metrics
  are easier, depend on the negative sample, and can even reorder models (Krichene & Rendle, 2020).
- NDCG@10 is position-sensitive within the top 10, unlike HR@10.
- Score ties with the target are counted as ranked above it (pessimistic). Constant or coarse scorers
  therefore cannot be rewarded. On popularity this changes NDCG@10 by ≤ 0.001 (checked).
- Per-user metric means are reported with standard errors. With 942 users, differences smaller than
  roughly 2 SE should not be interpreted.
- The same protocol, code path (`src/evaluate.py`) and candidate sets are used for every model.

Sanity baselines (seed 42):

| Model | Split | NDCG@10 | HR@10 | MRR@10 |
|---|---|---|---|---|
| Random | validation | 0.0033 | 0.0074 | 0.0021 |
| Random | test | 0.0045 | 0.0085 | 0.0034 |
| Popularity (train counts) | validation | 0.0314 | 0.0616 | 0.0222 |
| Popularity (train counts) | test | 0.0288 | 0.0584 | 0.0199 |

Random HR@10 agrees with its closed-form expectation (≈ 0.0062) within 1 SE. Any learned model,
including DP ones reported as useful, should beat popularity. A DP model that falls to popularity
level has learned no personalisation.

### Limitations

- MovieLens-100K is small (942 evaluated users). Metric standard errors are large relative to
  plausible effect sizes (popularity NDCG@10 SE ≈ 0.004), so multiple seeds are required.
- rating ≥ 4 as "positive" is a modelling assumption. Low ratings still indicate exposure/consumption.
- Secure aggregation is simulated/assumed, not cryptographically implemented.
- One held-out test item per user, so HR@10 and Recall@10 are identical and are not independent
  evidence.
- Per-user leave-last-out is not a global temporal split. One user's training data can be later in
  wall-clock time than another user's test item, so global popularity/trend information from the
  "future" can leak across users. This is standard practice but optimistic.
- Timestamp ties: 305 users (32%) have validation and test positives with the *same* timestamp, and
  316 users have their validation positive tied with their last train positive. MovieLens timestamps
  record when ratings were entered (often in bulk), not when movies were watched. For these users
  "chronological" order is decided by the raw item-id tie-break. The split is deterministic but the
  temporal semantics are weaker than they appear.
- No DP guarantee is claimed in this phase. DP is not implemented.

---

## 2026-09-28 — Phase 0b: final protocol changes before B0 (FROZEN for B0–B4)

Both changes were made **before any learned model existed**. They are corrections to the
protocol, not responses to model results. After this entry, the split and evaluation protocol are
frozen for every method B0–B4. Split content fingerprint (SHA-256 over
train/validation/test/history):
`6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989`.
It is pinned in `tests/test_data_split.py`, so any change to the split fails the test suite.

### 1. Timestamp ties: fixed pseudo-random tie-break (replaces raw-item-id order)

**Equal timestamps contain no reliable internal chronology.** MovieLens timestamps record when a
rating was entered, often many in the same second, not when a movie was watched. Ordering tied
interactions by item id imposed an arbitrary but *systematic* order (low ids → earlier), which could
correlate with item properties such as catalogue age. Ties are now ordered by a fixed pseudo-random key:

    tie_key = first 8 bytes of SHA-256("{user_id}:{item_id}:{split_seed}"),   split_seed = 2026

- Every user's **all** ratings (not only positives) are placed in one total order by
  `(timestamp, tie_key)`. The positive split and the "rated before" relation (below) both use it.
- `split_seed` is stored in `configs/dataset.yaml` and is immutable. It is the only seed preprocessing
  reads. The model-training `seed` is never used by preprocessing, and a test rebuilds the split
  under several training seeds to check that it is identical.
- SHA-256 is used, not Python's salted `hash()`, so the order is identical across runs, machines and
  Python versions. A test pins a known key value.
- Effect: 315 of 942 users have a different validation and/or test item than under item-id order
  (284 validation, 205 test). A test confirms that changes occur **only** for users whose split boundary
  is a timestamp tie.

### 2. Candidates: exclude every previously *rated* item, not only previous positives

"Positive" (rating ≥ 4, used for training and targets) is now distinguished from "previously
observed" (any rating). A movie rated 1–3 has been seen and should not be re-recommended as unseen.

- Validation candidates = catalog − every item rated strictly before the validation event. The target
  is kept. The future test item remains a candidate.
- Test candidates = catalog − every item rated strictly before the test event (all training-period ratings,
  the validation item, and low ratings between validation and test). The target is kept.
- "Before" = the frozen total order above. Ratings after an event never filter its candidates. A rating in
  the same second as the target counts as before only if the tie-break places it before, the same rule
  that defines the split.
- Effect: 43.6 (validation) / 45.1 (test) additional low-rated items removed per user on average.
  Mean candidates: 1581.6 / 1579.1 (was 1625.2 / 1624.2).

Unchanged: MovieLens-100K, rating ≥ 4, per-user chronological leave-two-out (no global time split),
min 3 positives (942 users; raw user 685 removed), all 1682 catalog items, cold targets kept (6
validation / 14 test now), full ranking, pessimistic score ties, NDCG@10 primary, HR@10 = Recall@10,
MRR@10, no retraining on train+validation, threat model.

### Updated sanity baselines (split_seed 2026; random seed 42)

| Model | Split | NDCG@10 (± SE) | HR@10 = Recall@10 | MRR@10 |
|---|---|---|---|---|
| Random | validation | 0.0023 ± 0.0009 | 0.0064 | 0.0011 |
| Random | test | 0.0038 ± 0.0015 | 0.0085 | 0.0024 |
| Popularity | validation | 0.0464 ± 0.0056 | 0.0839 | 0.0351 |
| Popularity | test | 0.0443 ± 0.0052 | 0.0870 | 0.0314 |

Random matches its closed-form HR@10 expectation (0.0064). Popularity rose from 0.029 to 0.044
test NDCG@10. Removing seen low-rated items takes away popular-but-already-seen movies that previously
occupied top slots.

### Statistical protocol for headline comparisons (decided now, before any model)

- Every headline configuration is trained with **multiple model-training seeds** (≥ 5). Results
  report the mean and spread across seeds. The data split does not vary with these seeds.
- Every method is evaluated on the **identical 942 test users and candidate sets**, so comparisons are
  **paired**. Differences are analysed per user (method A − method B per user, averaged over seeds),
  with paired bootstrap confidence intervals over users (and/or a paired permutation / Wilcoxon test).
  Unpaired comparisons of means ± SE are not used for claims.
- Model selection (including the choice of dimensionality) uses validation only. Test is computed
  once per selected configuration.

---

## 2026-09-28 — Phase B0 — Centralized BPR Baseline

> Superseded in part by **"Phase B0 revision"** below. The selected configuration (L2 = 1e-3), the
> "converged runs only" selection rule, max_epochs = 300 and the result numbers in this entry are no
> longer current. The model, negative sampling, training protocol and diagnostics are unchanged.

Phase 0 was not modified. Split fingerprint before and after B0:
`6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989`. All 43 Phase 0 tests still pass.

### Objective

Build a non-private, non-federated recommendation reference and check that a standard
collaborative-filtering model learns personalised rankings under the frozen full-ranking pipeline.
B0 is the upper reference for B1–B4. It is not meant to be state of the art.

### Model

Standard BPR matrix factorisation (`src/bpr.py`), d = 64: `score(u, i) = p_u · q_i`.
No biases, side features, genres, timestamps or neural layers.

- Parameters: users 942 × 64 = 60,288. Items 1682 × 64 = 107,648. **Total 167,936.**
- Initialisation: N(0, 0.01²) for both embedding tables.
- Loss per batch: `mean softplus(s_ui− − s_ui+) + λ · mean(‖p_u‖² + ‖q_i+‖² + ‖q_i−‖²)`.
  `softplus(−x) = −log σ(x)` is used for numerical stability. The L2 penalty is applied to the embeddings in
  the batch, not as Adam `weight_decay`.

### Training data and negatives (decision)

- **Positives:** only the frozen TRAIN split (rating ≥ 4). Each training positive is visited once per epoch, in
  a fresh random order.
- **Negatives:** one per positive, drawn **uniformly from all catalog items that are not a TRAIN positive of that
  user**, with fresh draws every epoch. The sampler is built from the training positives only. It never reads
  `history`, validation or test.
- Consequences, accepted deliberately:
  - Items the user rated below 4 are ordinary non-positives and can be sampled as negatives. They are not
    up-weighted as "hard" negatives. This does not use their rating or timing, so low ratings that happen after
    the validation/test events cannot leak into training.
  - The user's validation/test target can occasionally be drawn as a negative (probability about 1/1600 per
    draw). Excluding it would require using the held-out label, which is leakage. This is the standard behaviour
    for leave-out BPR.
- A unit test spies on every training batch. It checks that the positives used are exactly the training pairs, that no
  validation/test positive is used as a positive, and that no negative is a training positive.

### Training protocol

- Adam. Batch size 1024 (fixed, not searched). **Max 300 epochs.**
- Validation NDCG@10 after every epoch, using the frozen `src/evaluate.py`.
- **Early stopping** with patience 30 on validation NDCG@10. The best-validation weights are restored.
- CPU with 4 threads and `torch.use_deterministic_algorithms(True)`. Runs are bit-reproducible except for the
  wall-clock time column.
- The test set is evaluated **once**, after training, on the restored best-validation checkpoint. The training and
  search code paths never evaluate test.

### Hyperparameter search (validation only, seed 42, d = 64, batch 1024)

| Stage | lr | L2 | Best / run epochs | Stopped by | Val NDCG@10 |
|---|---|---|---|---|---|
| 2 | 5e-4 | 1e-5 | 296 / 300 | max_epochs (ineligible) | 0.0963 |
| 2 | **5e-4** | **1e-3** | **267 / 297** | early stopping | **0.0951 ← selected** |
| 1 | 1e-3 | 1e-4 | 148 / 178 | early stopping | 0.0941 |
| 2 | 1e-3 | 1e-3 | 153 / 183 | early stopping | 0.0939 |
| 1 | 1e-3 | 1e-5 | 150 / 180 | early stopping | 0.0938 |
| 1 | 1e-3 | 0 | 150 / 180 | early stopping | 0.0938 |
| 1 | 5e-3 | 1e-4 | 71 / 101 | early stopping | 0.0906 |
| 1 | 5e-3 | 1e-5 | 71 / 101 | early stopping | 0.0905 |
| 1 | 5e-3 | 0 | 71 / 101 | early stopping | 0.0901 |
| 2 | 1e-3 | 1e-2 | 152 / 182 | early stopping | 0.0895 |
| 2 | 5e-4 | 1e-2 | 155 / 185 | early stopping | 0.0825 |
| 1 | 1e-2 | 1e-5 | 6 / 36 | early stopping | 0.0793 |
| 1 | 1e-2 | 0 | 6 / 36 | early stopping | 0.0793 |
| 1 | 1e-2 | 1e-4 | 6 / 36 | early stopping | 0.0792 |

The full table, including HR/MRR, is in `results/b0_hyperparameter_search.csv`. Per-epoch curves are in `results/raw/b0_search/`.

### Selected configuration

**lr = 5e-4, L2 = 1e-3, batch 1024, d = 64, patience 30, max 300 epochs** (frozen in `configs/b0.yaml`).

Selection rule: the highest validation NDCG@10 among runs that **converged**, meaning they ended by early stopping
and not by hitting `max_epochs`. Exact ties go to the larger L2. The single highest value (lr 5e-4, L2 1e-5) reached
its best at epoch 296 of 300. Its optimum is undefined under the budget and its score depends on the arbitrary
cap, and B1–B4 need a reference with a well-defined stopping point. This rule was added after seeing
validation results (see Problems) but **before any test evaluation**, so the test estimate is not affected by it.

The choice among the top configurations barely matters. The top six lie within 0.0025 validation NDCG@10
of each other, while the per-user standard error of validation NDCG@10 is about 0.0074. These configurations are statistically
indistinguishable on validation.

### Results

Seed 42 (the selection seed): best epoch 267 of 297.

| Model | Split | NDCG@10 | HR@10 = Recall@10 | MRR@10 |
|---|---|---|---|---|
| Random | validation | 0.0023 | 0.0064 | 0.0011 |
| Random | test | 0.0038 | 0.0085 | 0.0024 |
| Popularity | validation | 0.0464 | 0.0839 | 0.0351 |
| Popularity | test | 0.0443 | 0.0870 | 0.0314 |
| B0 BPR-MF (seed 42) | validation | 0.0951 ± 0.0074 (SE) | 0.1858 | 0.0681 |
| B0 BPR-MF (seed 42) | test | 0.0888 ± 0.0070 (SE) | 0.1741 | 0.0628 |

Preliminary seed check (42, 123, 2026; mean ± std across seeds):

| Split | NDCG@10 | HR@10 = Recall@10 | MRR@10 | Best epoch |
|---|---|---|---|---|
| validation | 0.0920 ± 0.0046 | 0.1776 ± 0.0072 | 0.0663 ± 0.0055 | 222 ± 54 |
| test | 0.0856 ± 0.0040 | 0.1699 ± 0.0042 | 0.0602 ± 0.0038 | |

Per seed, test NDCG@10 is 0.0888 (42), 0.0812 (123) and 0.0870 (2026).

### Comparison and correctness diagnostics

- **B0 vs popularity (paired, per user, test NDCG@10, 10,000 bootstrap resamples over users):**
  - seed 42: +0.0445, 95% CI [0.0292, 0.0599];
  - 3-seed mean: +0.0414, 95% CI [0.0271, 0.0557].
  - B0 wins on 173 users, loses on 62, and ties on 707. The ties are mostly users where both methods miss the top 10.
- **Personalisation check** (seed 42 test NDCG@10): with each user's own vector 0.0888; with user vectors randomly
  permuted across users 0.0218; with one mean user vector for everyone 0.0492, which is about popularity
  level. The gain over popularity therefore comes from the per-user vectors.
- **Embeddings are trained:** the final user and item tables have 41.6× and 29.5× the Frobenius norm distance from their
  initialisation.
- **Checkpoint:** reloading `checkpoints/b0_best.pt`, with a safe weights-only load and a fingerprint check,
  reproduces test NDCG@10 = 0.088752 exactly.
- **Toy sanity test** (user 0 likes items {0, 1}, user 1 likes {2, 3}): after training, each user's positives are ranked above
  all other items. This is automated in `tests/test_bpr.py`.

### Problems encountered

1. **The initial L2 grid had no measurable effect.** L2 ∈ {1e-4, 1e-5, 0} gave near-identical curves at every lr, and
   1e-5 and 0 were *exactly* tied. With the per-example penalty used here, λ ≤ 1e-4 is far below the BPR term, so
   stage 1 effectively searched lr only. A second, final stage was declared (lr {1e-3, 5e-4} × L2 {1e-2, 1e-3,
   1e-5}, 5 new runs). It showed that L2 starts to matter only at 1e-2, where it hurts.
2. **Patience 10 stopped runs on noise.** Validation NDCG@10 fluctuates by about ±0.003 between epochs (942 users), while
   the underlying improvement late in training is about 0.0002 per epoch. With patience 10, runs stopped while still
   improving, which also biased the search against smaller learning rates. Patience was raised to 30 and max epochs to 300,
   and **the whole search was re-run** under that rule. The patience-10 results are kept in
   `results/raw/b0_search/b0_hyperparameter_search_patience10.csv` for transparency. No test evaluation had
   happened at that point.
3. **The best learning rate kept landing on the grid edge.** Lower lr trains more slowly but ends slightly higher, and
   lr 5e-4 with L2 1e-5 hit the 300-epoch cap. I did not extend further (lr 2.5e-4, more epochs), because that would be
   score-chasing for gains well within validation noise. This is handled by the convergence rule above.
4. **The stopping point is seed-sensitive:** the best epoch was 162, 238 and 267 for the three seeds. Even in the selected run,
   validation NDCG was still drifting up very slowly when patience ended it (plot in
   `results/plots/b0_validation_ndcg.png`). B0 is therefore "converged under the declared rule", not at an exact optimum.
5. **Selection optimism.** Seed 42 was used for the search, so its validation score (0.0951) is optimistically biased.
   Seeds 123 and 2026 score 0.0867 and 0.0941 on validation. Test is lower than validation for all three seeds
   (mean 0.0856 vs 0.0920).
6. **Checkpoint bug (fixed).** `torch.__version__` was stored as a `TorchVersion` object, which the safe
   `weights_only=True` loader rejects. It is now stored as a string, and all checkpoints were regenerated by re-running the
   seeds. The metric outputs were byte-identical to the first run.

### Interpretation

Centralised BPR-MF learns meaningful personalised rankings under the frozen protocol. It roughly doubles
popularity's test NDCG@10 (0.0856 vs 0.0443). The paired confidence interval excludes zero by a wide margin, and the gain
disappears when user vectors are shuffled or averaged. Absolute values are modest (about 17% of test targets land in the
top 10 of roughly 1,580 candidates), as expected for strict chronological leave-last-out with full ranking and seen items removed.

### Decision

**Freeze B0** (`configs/b0.yaml`, d = 64, lr 5e-4, L2 1e-3, batch 1024, patience 30, max 300 epochs, the
negative-sampling rule above) as the non-private centralised reference for B1–B4.

- It is correct (54 tests, including leakage spies and the toy sanity check), bit-reproducible, and clearly
  better than popularity.
- Seed-to-seed variation (std 0.004 test NDCG@10) is about the size of the per-user SE, so headline comparisons must use
  ≥ 5 seeds with paired per-user analysis, as already planned.
- B0's lr and L2 are for centralised Adam training. They are **not** automatically valid for federated or DP training,
  which will need their own validation-based tuning under the same protocol.

---

## 2026-09-29 — Phase B0 revision: epoch cap and selection rule

### Why

The 2026-09-28 rule ("highest validation NDCG@10 among runs that stopped early rather than hitting
the epoch cap") rejected the configuration with the highest validation score (lr 5e-4, L2 1e-5,
0.0963) only because it hit `max_epochs = 300`. Hitting a budget does not make a configuration invalid.
It means its optimum was not established under that budget. The honest fix is to give it enough budget and
check whether it converges, not to exclude it.

### Validation-only convergence check (declared before running)

The censored configuration was re-run with the same seed (42), patience (30), frozen split and evaluator,
and `max_epochs = 600`. The test set was not touched.
`python experiments/run_b0.py --config configs/b0.yaml --check 0.0005 0.00001 --max-epochs 600`

Training is deterministic, so the first 300 epochs repeat exactly (verified), and the best validation score can
only be ≥ 0.0963. The only open question was convergence. The pre-declared outcomes were:

- **A:** it early-stops before 600. Plain argmax then selects it.
- **B:** it hits 600 again. The existing configuration stays.

**Result: Case A.** The best epoch stays at 296 (validation NDCG@10 0.096255), and early stopping triggers at epoch
**326**. The old cap cut it off 4 epochs before patience would have confirmed convergence. After epoch 296,
validation NDCG@10 plateaus and dips (0.0939–0.0950 at epochs 300–320, 0.0922 at 326).

### Changes

- `max_epochs`: 300 → **600** for all configurations. Every other search run already early-stopped before 300, so
  re-running the whole search under the 600 cap changed **exactly one row** (lr 5e-4, L2 1e-5: 296/326 instead of
  296/300). The search CSV is regenerated under this single protocol.
- Selection rule: **highest validation NDCG@10** (exact ties → larger L2). The "converged runs only" filter is
  removed. As a guard, if the winner ever hits `max_epochs`, the search raises an error ("raise the budget")
  instead of silently skipping it.
- **Frozen B0: lr 5e-4, L2 1e-5**, batch 1024, d = 64, patience 30, max 600 epochs, same negative sampling.
- The superseded outputs (L2 1e-3: search CSV, per-seed and per-user results, checkpoints, plots) are archived in
  `results/raw/superseded_b0_lr5e-4_l2_1e-3/`, not deleted, because their test numbers were already reported.

### Results (revised B0)

| Model | Split | NDCG@10 | HR@10 = Recall@10 | MRR@10 |
|---|---|---|---|---|
| Random | test | 0.0038 | 0.0085 | 0.0024 |
| Popularity | test | 0.0443 | 0.0870 | 0.0314 |
| B0, seed 42 (best epoch 296 / 326) | validation | 0.0963 | 0.1783 | 0.0717 |
| B0, seed 42 | test | 0.0841 | 0.1688 | 0.0581 |
| B0, seeds 42/123/2026, mean ± std | validation | 0.0935 ± 0.0056 | 0.1769 ± 0.0025 | 0.0685 ± 0.0066 |
| B0, seeds 42/123/2026, mean ± std | test | **0.0844 ± 0.0021** | 0.1674 ± 0.0025 | 0.0594 ± 0.0034 |

- Per seed: test NDCG@10 0.0841 / 0.0825 / 0.0867. Best epochs 296 / 162 / 311. Seed 2026 would also have been censored
  by the old 300 cap.
- Paired B0 (3-seed mean) − popularity on test NDCG@10: **+0.0402, 95% bootstrap CI [+0.0264, +0.0543]**
  (179 users better, 63 worse, 700 equal).
- Personalisation check (seed 42 test NDCG@10): own user vectors 0.0841, permuted user vectors 0.0200, one mean user vector
  0.0514.
- Checkpoint reload with the fingerprint check reproduces test NDCG@10 = 0.084091 exactly.

### Comparison with the superseded configuration (reported, not used for selection)

On test, the revised B0 is marginally *lower* than the superseded one (3-seed mean 0.0844 vs 0.0856). Paired per user:
**−0.0012, 95% CI [−0.0040, +0.0016]**, so there is no detectable difference. The revision does not change the reference level.
It makes the selection follow the pre-stated primary rule (validation argmax) with no exceptions. Switching back
because of this test number would be test-set selection, so it is not done.

### Decision

**B0 frozen** as lr 5e-4, L2 1e-5, max 600 epochs, patience 30 (`configs/b0.yaml`). All other conclusions of the
2026-09-28 entry stand. Lesson for B1–B4: epoch/round budgets must be large enough that the selected
configuration converges by early stopping. The search code now enforces this.

---

## 2026-09-29 — Reproducibility finding (corrects "bit-reproducible")

While preparing the B0 report, the revised B0 was reproduced in a fresh copy (new venv from `requirements.txt`, raw
data only). Preprocessing, the baselines and seeds 123 and 2026 were byte-identical to the main repository, and all
54 tests passed. **Seed 42 was not byte-identical.** Its training loss first differed in the sixth decimal at epoch 13.
The final weights differ by a relative 7.8e-6 (largest single difference 5e-5). The best epoch (296) and all reported
metrics on validation and test were identical to six decimals. One validation user's rank moved from 596 to 595.

Library versions (torch 2.14.0+cpu, numpy 2.2.6), hardware, thread count and
`torch.use_deterministic_algorithms(True)` were the same. The most likely cause, not verified, is float32 MKL kernels
whose rounding depends on memory alignment, which PyTorch's deterministic mode does not control.

The earlier statements in this log that B0 runs are "bit-reproducible" are therefore too strong. The accurate claim is
**reproducible to the reported precision, with byte-identical results in most runs but not guaranteed**. No reported B0
number changes. Action before B1: evaluate MKL conditional numerical reproducibility (`MKL_CBWR`) or an equivalent
setting, and re-test cross-environment bit-identity. No code was changed in this phase.

---

## 2026-09-29 — Phase B1 — Federated BPR Baseline

Phase 0 and B0 were not modified. Before and after B1: all 54 prior tests pass, the raw `u.data` hash is unchanged,
and the split fingerprint is `6faed6fc…6989`. B0 is the revised frozen configuration (lr 5e-4, L2 1e-5).

### Objective

Measure the effect of federated optimisation on utility *before* DP. B1 contains federation, Poisson client
sampling, local training, persistent local user vectors, a shared item model, server aggregation and communication
accounting. It has no clipping, no noise and no low-rank structure.

### Architecture

- One client per retained user (942).
- **Local:** p_u ∈ ℝ⁶⁴ and the user's training positives. p_u persists across rounds, is never averaged, never sent,
  and never reset.
- **Global:** Q ∈ ℝ^{1682×64}.
- Score: `p_u · q_i`, as in B0.
- Initialisation is identical to B0 for the same seed (P from B0's user table, Q from its item table, N(0, 0.01²)).
- Implementation: `src/federated.py` (simulator) and `src/train_federated.py` (training loop). P is stored as one
  array in the simulator, but only the client's own code path reads or writes its row, and tests enforce this.

### Round algorithm (as implemented)

1. **Sampling:** S_t = {u : U_u < q}, with U_u ~ Uniform(0,1) independently (Poisson sampling).
   **Empty rounds are skipped** (Q unchanged), not resampled, so the sampling distribution stays exact for B2's
   accounting. With q = 0.1 there were no empty rounds (min 62 clients per round across all seeds).
2. **Local training:** each u ∈ S_t downloads Q_t, sets Q_u ← Q_t and runs E local epochs. Each epoch is **one
   full-batch SGD step** over all of u's n_u training positives, with fresh uniform negatives (any item that is not a
   training positive of u, as in B0):
   `L_u = (1/n_u) Σ_k softplus(−p_uᵀ(q_{i_k} − q_{j_k})) + λ[‖p_u‖² + (1/n_u) Σ_k (‖q_{i_k}‖² + ‖q_{j_k}‖²)]`
   `p_u ← p_u − η ∇_p L_u,  Q_u ← Q_u − η ∇_Q L_u` (simultaneous update).
   This is B0's per-batch loss applied to one client's triplets. The gradients are hand-derived in numpy, and a unit
   test checks them against PyTorch autograd of `src.bpr.bpr_loss`. L2 applies only to rows used in the loss.
3. **Upload:** the client sends the dense ΔQ_u = Q_u − Q_t and keeps p_u.
4. **Aggregation (equal-user FedAvg):** `Q_{t+1} = Q_t + η_s · (1/|S_t|) Σ_{u∈S_t} ΔQ_u` with η_s = 1.

### The sparse-update question

A client touches only the rows of its positives and sampled negatives, so every other row of ΔQ_u is exactly 0.
- **Untouched rows are not shrunk.** A zero row adds nothing, so an item that no selected client touched keeps its
  embedding bit-for-bit (tested). The only way unrelated items could shrink is L2 or weight decay applied to *all*
  of Q locally, which is not done.
- **Dilution is intended.** An item touched by k of |S_t| clients moves by (k/|S_t|) × (mean of those k updates).
  This is the exact gradient of the equal-user average objective (1/|S_t|) Σ_u L_u, not an artefact.
- **Per-item averaging** (divide each row by the number of clients that touched it) changes the weighting
  from users to items. Under DP it would also need private per-item counts. It was implemented only as a diagnostic
  and was **not eligible for selection**. At the frozen configuration (seed 42, validation only):
  FedAvg 0.0959 vs item-mean 0.0911 (best rounds 1040 vs 820). Equal-user FedAvg is also the better choice
  empirically. (`results/b1_aggregation_diagnostic.csv`)

### Local optimiser and learning-rate scale

Local SGD, with no optimiser state to carry between rounds. A pilot (seed 42, q 0.1, E 1, validation only) showed that
learning rates of 1e-3 to 1e-2, as suggested for Adam, are far too small for SGD here. From N(0, 0.01²) embeddings with a
per-client mean loss, the gradients are about 1e-3. After 200 rounds, lr 0.1 did not move validation NDCG@10
(0.0066 → 0.0046), lr 1 reached 0.0175, and lr 10 reached 0.063. lr 100 diverged (non-finite scores, which the
evaluator rejects). The search grid was therefore set around lr 5–20. Two fixes came out of the pilot: an
overflow-safe sigmoid in the gradient, and explicit divergence handling (a run is marked `diverged` and is ineligible).

### Hyperparameter search (validation only, seed 42; L2 fixed at B0's 1e-5; server lr 1)

Budget: max 8,000 rounds, validation every 10 rounds, patience 50 evaluations (500 rounds). The pilot showed
validation plateaus of about 150 rounds, so a 200-round patience would have repeated B0's early-stopping problem.
Stages were declared in `configs/b1.yaml` before running. Stages 2 and 3 are derived automatically from earlier stages.

| Stage | q | local lr | E | best / run rounds | Val NDCG@10 |
|---|---|---|---|---|---|
| 2 | **0.10** | **5** | **2** | **1040 / 1540** | **0.0959 ← selected** |
| 1 | 0.10 | 10 | 1 | 1110 / 1610 | 0.0958 |
| 3 | 0.10 | 2.5 | 2 | 1770 / 2270 | 0.0951 |
| 1 | 0.10 | 5 | 1 | 1940 / 2440 | 0.0948 |
| 1 | 0.20 | 5 | 1 | 1910 / 2410 | 0.0944 |
| 1 | 0.20 | 10 | 1 | 1110 / 1610 | 0.0942 |
| 1 | 0.05 | 10 | 1 | 1590 / 2090 | 0.0932 |
| 1 | 0.05 | 20 | 1 | 1040 / 1540 | 0.0928 |
| 1 | 0.05 | 5 | 1 | 2120 / 2620 | 0.0921 |
| 2 | 0.10 | 10 | 2 | 920 / 1420 | 0.0918 |
| 1 | 0.10 | 20 | 1 | 790 / 1290 | 0.0918 |
| 2 | 0.10 | 5 | 5 | 710 / 1210 | 0.0917 |
| 1 | 0.20 | 20 | 1 | 400 / 900 | 0.0913 |
| 2 | 0.10 | 10 | 5 | 630 / 1130 | 0.0909 |

- Stage 1: q ∈ {0.05, 0.1, 0.2} × lr ∈ {5, 10, 20}, with E = 1.
- Stage 2: the best q (0.1) with E ∈ {2, 5} × lr ∈ {best, best/2}.
- Stage 3 is run only if the winner's lr is at the edge of the lrs tried for its (q, E). It was (lr 5 at E 2), so lr 2.5
  was tried and was worse.
- **All 14 runs ended by early stopping. None hit the round cap and none diverged.**

### Selected configuration and convergence

**q = 0.1, local SGD lr 5.0, E = 2 local epochs, L2 1e-5, equal-user FedAvg, server lr 1** (frozen in
`configs/b1.yaml`). This is the highest validation NDCG@10.
- The runner-up (lr 10, E 1: 0.0958) is indistinguishable (validation SE ≈ 0.0074).
  The rule, not the margin, decides.
- Convergence check: `--check 0.1 5.0 2 --max-rounds 3000` gives the same best round (1040), stopped at 1540. The result
  does not depend on the 8,000-round budget.
- Across seeds, the best rounds were 1040 / 1020 / 970, with early stopping 500 rounds later. Validation NDCG@10 passes popularity
  (0.0464) by round 30–40 and plateaus at about 0.09 after round ~1000. The selected round is a noisy maximum on that
  plateau, which is the usual optimism of choosing the best of many validation checks. It affects only validation numbers,
  not the test estimate.
- Local loss: 0.69 in the first rounds, about 0.045–0.049 at the best round.
- Mean user-vector norm: 0.08 → about 11.

### Results (test evaluated once per seed, on the restored best-validation checkpoint)

| Model | Split | NDCG@10 | HR@10 = Recall@10 | MRR@10 |
|---|---|---|---|---|
| Random | test | 0.0038 | 0.0085 | 0.0024 |
| Popularity | test | 0.0443 | 0.0870 | 0.0314 |
| B0 centralised (42/123/2026) | test | 0.0844 ± 0.0021 | 0.1674 ± 0.0025 | 0.0594 ± 0.0034 |
| **B1 federated (42/123/2026)** | test | **0.0875 ± 0.0021** | 0.1642 ± 0.0054 | 0.0645 ± 0.0012 |
| B1 federated | validation | 0.0940 ± 0.0018 | 0.1829 ± 0.0031 | 0.0674 ± 0.0015 |

Per seed, test NDCG@10 is 0.0890 (42), 0.0852 (123) and 0.0885 (2026).

### B0 comparison

- **NDCG@10 retention** B1/B0 = 0.0875 / 0.0844 = **1.037** on test (1.005 on validation). The
  **federation-induced utility gap** (B0 − B1)/B0 = **−3.7%** on test, meaning B1 is marginally higher.
- **Paired per-user test NDCG@10**, B1 − B0 (each seed-averaged over 42/123/2026), 10,000 bootstrap resamples:
  **+0.0031, 95% CI [−0.0042, +0.0105]**. B1 is better for 15.1% of users, worse for 12.1%, and tied for 72.8%.
  **No detectable federation-induced change in either direction.** This is not evidence that federation *improves*
  utility.
- HR@10 is slightly lower for B1 (0.1642 vs 0.1674) while MRR@10 is higher (0.0645 vs 0.0594). This is within seed spread.
- A plausible mechanism for parity, not tested causally: B1's client objective averages over each user's own triplets and
  then weights users equally, while B0 weights every interaction equally. The metric averages over users.
- Activity terciles (fixed cut-points on training positives), test NDCG@10 as B0 → B1, paired difference:

  | Group | Train positives | B0 → B1 | Paired difference (95% CI) |
  |---|---|---|---|
  | low | 1–22 | 0.1218 → 0.1314 | +0.0096 [−0.0063, 0.0258] |
  | medium | 23–61 | 0.0667 → 0.0618 | −0.0050 [−0.0161, 0.0064] |
  | high | 62–376 | 0.0639 → 0.0684 | +0.0045 [−0.0049, 0.0138] |

  All three intervals include 0. This is a secondary diagnostic and was not used for tuning. (`results/b1_activity_groups.csv`)

### Communication (simulated float32 tensor volume, not network traffic)

- **Per selected client per round:** download Q_t, 1682 × 64 × 4 = 430,592 B, and upload the dense ΔQ_u, 430,592 B.
  Total **861,184 B (≈ 0.86 MB)**.
- **Participation (seed 42):** 93.8 ± 8.9 clients per round (min 68, max 131, 0 empty rounds). 64.8% of clients were
  sampled at least once by round 10, 99.6% by round 50, and 100% by the best round.
- **Totals, seed 42:**
  - to the best round (1040): **84.15 GB** (42.07 GB down + 42.07 GB up);
  - whole run including the 500-round patience tail (1540 rounds): **124.43 GB**.
  - Seeds 123 and 2026: 82.78 and 78.81 GB to the best round.
- **Sparse diagnostic (not the protocol):** if clients uploaded only touched rows (d floats plus a 4-byte index), the mean
  upload would be about 42.3 KB per client, about 9.8% of the dense upload. The dense protocol is kept, because B2's Gaussian
  noise must cover every coordinate.
- Per-round logs: `results/b1_communication.csv` and `results/raw/b1_communication_seed*.csv`. Plot:
  `results/plots/b1_communication.png`.

### Personalisation sanity check (`checkpoints/b1_best.pt`, seed 42, test NDCG@10)

| User vector | Test NDCG@10 |
|---|---|
| Own local p_u | 0.0890 |
| Random other client's p_v | 0.0202 |
| Mean of all p_u | 0.0502 |

Federation preserved personalisation. The same pattern as B0 (0.0841 / 0.0200 / 0.0514) holds.

### Tests

**71 passing**: the 54 prior tests unchanged, plus 17 in `tests/test_federated.py`. They cover:
- hand-derived gradients against autograd of B0's loss;
- ΔQ_u equals Q_local − Q_t, and it is exactly zero outside touched rows;
- the FedAvg arithmetic on a toy example, plus the item-mean diagnostic;
- untouched item rows unchanged after aggregation;
- the server update equals the mean of the received deltas, which are Q-shaped only;
- selected clients receive the current Q_t;
- selected p_u change and unselected p_u do not;
- a non-participating client's p_u cannot influence Q;
- p_u persists through empty rounds;
- empty rounds are skipped, not resampled;
- seeded client selection and training are reproducible;
- communication counts on a toy example;
- a toy federation in which each client ranks its own items first. This fails if p_u were averaged.
- the split fingerprint is unchanged;
- training positives are exactly the client's train positives, with no held-out item;
- the checkpoint holds all p_u and Q, and a wrong split fingerprint is rejected.

### Reproducibility

- Search and seed runs execute in parallel worker processes with single-threaded BLAS.
- The standalone `--seed 42` run was byte-identical to the multi-seed run's seed-42 result files and bit-identical in P and Q.
- The `--check` rerun reproduced best round 1040 exactly.
- The checkpoint stores Q, all p_u, the configurations, the seed, the round, the validation metrics, the sampling scheme
  (Poisson, q, empty rounds skipped) and the split fingerprint. It is loaded weights-only, with a fingerprint check.

### Problems

1. **Learning-rate scale.** Plain SGD needed learning rates about 1,000× larger than the Adam-style range. Establishing the scale took a pilot (recorded above).
2. **Divergence at lr 100:** gradient overflow and then non-finite scores. Fixed with a stable sigmoid and explicit `diverged` handling.
3. **Patience:** set to 500 rounds after the pilot showed about 150-round plateaus.
4. **A self-matching `pgrep` wait loop** in a verification script hung. This was a tooling issue that did not affect any results. The runs themselves had completed and were verified afterwards.
5. **The noisy maximum on the validation plateau** makes the reported validation numbers optimistic, as in B0.

### Decision

**Freeze B1** (`configs/b1.yaml`: q 0.1, local SGD lr 5, E 2, L2 1e-5, equal-user FedAvg, server lr 1, dense ΔQ).
- **Correct:** locality of p_u, aggregation arithmetic and leakage are enforced by tests. Personalisation survives federation.
- **Stable:** test std 0.0021 across seeds, with the same best-round range.
- **Converged:** all runs early-stopped, and the result is unchanged with a larger budget.
- It shows **no detectable federation-induced utility loss** relative to B0 on this benchmark. B2 can therefore attribute utility changes to DP rather than to federation.
- **Caveat for B2:** aggregation currently divides by the realised |S_t|. DP-FedAvg usually divides by the expected qN so that the sensitivity is fixed. B2 must choose and document this explicitly. The 0.86 MB-per-client dense payload is the full-rank communication reference for B3 and B4.

---

## 2026-10-02 — Revised stopping protocol for B0, B0-UW and B1 (declared BEFORE any new test evaluation)

### Why

To separate the federation effect from the weighting effect in B0 → B1, a centralised user-weighted control
(**B0-UW**) was added: B0 with training examples drawn user-uniformly and then positive-uniformly. In expectation this
optimises (1/N) Σ_u (1/n_u) Σ_k L_uk, the equal-user objective of B1's aggregation, instead of B0's interaction-weighted mean.
It is configured in `configs/b0_uw.yaml`, and B0's own training path was verified bit-identical after adding the option.

Under B0's frozen patience of 30, two of three B0-UW seeds stopped at a transient early peak (epochs 8–10, validation about 0.064)
that later runs climb out of. A **validation-only** check with patience 100 (test not touched) then showed that **B0 itself
had stopped too early for seeds 123 and 2026**:
- best epochs changed from 162 → 309 and 311 → 476;
- 20-epoch-smoothed validation NDCG@10 improved by +0.0076 and +0.0046, so this is not noise;
- seed 42 was unchanged.

B1 used a different stopping rule (50 evaluations = 500 rounds). The B0/B1 comparison therefore mixed in a stopping-rule
asymmetry. Logs: `results/raw/weighting_control/` (archived copy in the superseded folder below).

### Revised protocol (applies to B0, B0-UW and B1; replaces the earlier patience settings)

1. **Patience = 100 validation checks** for every model.
   - B0 / B0-UW: validated every epoch, so 100 epochs; `max_epochs` raised 600 → 1000 so the winner is not censored.
   - B1: validated every 10 rounds, so 1,000 rounds; `max_rounds` stays 8,000.
   - The censoring guard is unchanged: if a selected run ends at the cap, the search fails instead of selecting it.
2. **B0 search:** re-run the original validation search with the **same search space** (stage 1: lr {1e-2, 5e-3, 1e-3} ×
   L2 {1e-4, 1e-5, 0}; stage 2: lr {1e-3, 5e-4} × L2 {1e-2, 1e-3, 1e-5}; 14 configurations), seed 42, and no new
   hyperparameters. Select the highest validation NDCG@10 (exact ties → larger L2).
3. **B0-UW (strict control):** the **selected revised B0** architecture and hyperparameters with
   `example_sampling: user_uniform`, patience 100. It is not retuned. A small learning-rate sensitivity run (lr {1e-3, 5e-4,
   2.5e-4} at the selected B0 L2, seed 42) is kept as a **validation-only diagnostic** and cannot replace the strict control.
4. **B1 search:** re-run the existing validation search (same stages and automatic stage-2/3 derivation rules, same space,
   seed 42), patience 100 evaluations. Select the highest validation NDCG@10 (ties → fewer local epochs).
5. **Freeze all three configurations on validation only.** Only after all three are frozen, train seeds 42/123/2026 for
   each and evaluate each selected checkpoint **once** on test.
6. **Comparisons** (paired per user on test NDCG@10, each model's per-user score averaged over its three seeds, 10,000 user
   bootstrap resamples):
   - B0-UW − B0 (weighting effect, centralised);
   - B1 − B0-UW (federation effect at the same user weighting);
   - B1 − B0 (total).

   Plus the activity-tercile breakdown.
7. All previous B0, B1 and B0-UW headline outputs (results, per-seed and per-user files, search tables, checkpoints, plots,
   configs) are archived unchanged in `results/raw/superseded_stopping_rule_2026-10-02/`. Earlier log entries remain as
   written. Their numbers are superseded by the entry that reports this protocol's results.

### 2026-10-02 — Validation-only selections under the revised protocol (recorded BEFORE any test evaluation)

- **B0 search** (14 configurations, seed 42, patience 100, max 1,000 epochs): the selected configuration is
  **lr 1e-3, L2 1e-2**, with validation NDCG@10 0.0977 (best epoch 417 of 517). All runs ended by early stopping.
  - Runners-up: lr 1e-3 / L2 1e-3 0.0971, lr 1e-3 / L2 1e-4 0.0963, lr 5e-4 / L2 1e-5 0.0963. The last is the old frozen
    B0; it reproduced best epoch 296 and 0.096255 exactly, which serves as a determinism check.
  - The top four lie within 0.0015, which is well inside the validation SE (≈ 0.0075).
  - **Caveat:** L2 1e-2 is the largest value in the declared search space. As declared, the space was **not** extended.
  - Table: `results/b0_hyperparameter_search.csv`; curves: `results/raw/b0_search/`.
- **B1 search** (same stages and space, seed 42, patience 100 evaluations): the selected configuration is **unchanged:
  q 0.1, lr 5, E 2**, with validation NDCG@10 0.0959 (best round 1040 of 2040). The derived stages 2 and 3 were identical to
  before, and all runs ended by early stopping. Table: `results/b1_hyperparameter_search.csv`.
- **B0-UW:** revised B0's lr 1e-3 / L2 1e-2 with `example_sampling: user_uniform`, patience 100. No selection of its own.
  The learning-rate sensitivity diagnostic (validation only) uses lr {1e-3, 5e-4, 2.5e-4} at L2 1e-2.
- **Configs frozen:** `configs/b0.yaml`, `configs/b0_uw.yaml`, `configs/b1.yaml`. Test is evaluated next, once per seed
  (42/123/2026) per model, on the validation-selected checkpoints.

### 2026-10-02 — Results under the revised protocol (B0, B0-UW, B1 frozen; test evaluated once per seed)

All nine seed runs (3 models × seeds 42/123/2026) ended by early stopping. Each test set was evaluated once, on the
validation-selected checkpoint, after all three configurations were frozen.

| Model (config) | Best epochs / rounds | Val NDCG@10 | **Test NDCG@10** | Test HR@10 = Recall@10 | Test MRR@10 |
|---|---|---|---|---|---|
| Random | — | 0.0023 | 0.0038 | 0.0085 | 0.0024 |
| Popularity | — | 0.0464 | 0.0443 | 0.0870 | 0.0314 |
| **B0** centralised, interaction-weighted (lr 1e-3, L2 1e-2) | 417 / 262 / 301 | 0.0981 ± 0.0010 | **0.0860 ± 0.0005** | 0.1720 ± 0.0021 | 0.0601 ± 0.0008 |
| **B0-UW** centralised, user-weighted (same hyperparameters) | 272 / 266 / 177 | 0.0935 ± 0.0018 | **0.0917 ± 0.0019** | 0.1819 ± 0.0058 | 0.0645 ± 0.0011 |
| **B1** federated, user-weighted (q 0.1, lr 5, E 2) | 1040 / 1020 / 970 | 0.0940 ± 0.0018 | **0.0875 ± 0.0021** | 0.1642 ± 0.0054 | 0.0645 ± 0.0012 |

Per-seed test NDCG@10:
- B0: 0.0865 / 0.0857 / 0.0858.
- B0-UW: 0.0932 / 0.0924 / 0.0896.
- B1: 0.0890 / 0.0852 / 0.0885.

B1's selected rounds are the same as under patience 50, so its checkpoints and test results are byte-identical to the
pre-revision run. Only its patience tail is longer.

**Paired per-user test NDCG@10** (each model seed-averaged; 10,000 user bootstrap resamples; `results/weighting_control_paired.csv`):

| Comparison | Mean difference | 95% CI | Improved / degraded / tied |
|---|---|---|---|
| B0-UW − B0 (weighting effect, centralised) | +0.0058 | [−0.0002, +0.0119] | 12.3% / 10.4% / 77.3% |
| B1 − B0-UW (federation effect, same weighting) | −0.0042 | [−0.0109, +0.0023] | 13.2% / 12.2% / 74.6% |
| B1 − B0 (total) | +0.0016 | [−0.0054, +0.0087] | 14.1% / 11.8% / 74.1% |

**Activity terciles (secondary analysis, not used for any decision; `results/weighting_control_activity.csv`):**

| Group | Train positives | B0 | B0-UW | B1 | B0-UW − B0 (95% CI) | B1 − B0-UW (95% CI) |
|---|---|---|---|---|---|---|
| low | 1–22 | 0.1258 | 0.1444 | 0.1314 | **+0.0186 [+0.0063, +0.0314]** | −0.0131 [−0.0291, +0.0023] |
| medium | 23–61 | 0.0705 | 0.0708 | 0.0618 | +0.0003 [−0.0100, +0.0109] | −0.0090 [−0.0191, +0.0004] |
| high | 62–376 | 0.0607 | 0.0588 | 0.0684 | −0.0019 [−0.0098, +0.0056] | **+0.0096 [+0.0028, +0.0168]** |

These are six subgroup tests with no multiplicity correction, so individual intervals that exclude zero are suggestive only.

**Personalisation check** (seed-42 checkpoint, test NDCG@10; own / shuffled p / mean p):
- B0: 0.0865 / 0.0248 / 0.0483;
- B0-UW: 0.0932 / 0.0220 / 0.0507;
- B1: 0.0890 / 0.0202 / 0.0502.

All checkpoints reload, with the fingerprint check, to their saved test scores exactly.

**B0-UW learning-rate sensitivity** (validation only, seed 42, L2 1e-2; `results/b0uw_hyperparameter_search.csv`):
- lr 1e-3: 0.0950 (epoch 272) — the strict control's own setting;
- lr 5e-4: 0.0918 (epoch 335);
- lr 2.5e-4: 0.0636 (epoch 35). It stalls at the early transient peak even with patience 100.

The strict control was not replaced.

**Interpretation**
- **The weighting change matters, and it explains part of B1's parity with B0.** Centralised user weighting improves test
  NDCG@10 by +0.0058, borderline with the CI just touching 0. The gain is concentrated in low-activity users (+0.0186).
  Against the user-weighted centralised control, federation retains **95.4%** of test NDCG@10 (0.0875 / 0.0917). That
  −0.0042 federation effect is not statistically detectable with this design. The earlier wording "no detectable
  federation-induced loss" relative to B0 conflated the two effects.
- **The revised B0 is better than the superseded one** (0.0860 ± 0.0005 vs 0.0844 ± 0.0021 test) and much more stable
  across seeds. Its validation-to-test drop is notably larger than B0-UW's or B1's (0.0981 → 0.0860, vs 0.0935 → 0.0917 and
  0.0940 → 0.0875). Part of this is selection optimism: 14 configurations, seed 42.
- **Caveats:**
  - B0's selected L2 (1e-2) is at the edge of the declared search space, which was deliberately not extended.
  - B0-UW inherits B0's hyperparameters, so it is a strict one-factor control, not an optimised user-weighted model.
  - Three seeds only.

**Decision.** Freeze **B0 (revised)**, **B0-UW (diagnostic control)** and **B1 (unchanged configuration, patience 100)**
under this protocol. B1 → B2 will isolate the incremental effect of DP, because B2 adds DP directly to the frozen B1
protocol. B0 and B0-UW remain the centralised references for interpreting the federation effect. Previous outputs are in
`results/raw/superseded_stopping_rule_2026-10-02/`.

---

## 2026-10-02 — Phase B2 protocol (declared BEFORE any DP run or B1 norm statistics)

- **Mechanism:** the frozen B1 round, then
  ΔQbar_u = ΔQ_u · min(1, C/‖ΔQ_u‖_F) (one norm over the whole shared update, with a 1e-6 safety margin so that the
  float32 result is strictly ≤ C);
  Σ̃_t = Σ_{u∈S_t} ΔQbar_u + N(0, σ²C² I_D);
  Q_{t+1} = Q_t + η_s Σ̃_t / (qN), with qN = 94.2.
  p_u is local and is never clipped or uploaded. Empty Poisson rounds still receive noise and count as accountant steps.
  Implemented in `src/privacy.py` as a subclass of B1's simulator. With C = ∞, σ = 0 and B1's realised denominator it
  reproduces B1 bit-for-bit (tested on toy data and on the real split).
- **Privacy unit:** one user, with add/remove-one-user adjacency. Each round is a Poisson-subsampled Gaussian mechanism
  (rate q = 0.1, noise multiplier σ, L2 sensitivity C of the clipped sum).
- **Horizon:** exactly **T = 1000** private rounds (B1's selected rounds were 970–1040). There is no early stopping and no
  checkpoint selection. The reported model is the round-T model. Validation is logged every 10 rounds as a diagnostic only.
- **Accounting:** δ = 1e-5. **Primary accountant: Opacus PRVAccountant**; cross-check: Opacus RDPAccountant. Both are reported
  for every configuration, and the primary is not switched after seeing numbers. σ is solved per target
  ε ∈ {8, 4, 2, 1} by bisection to the smallest σ with ε_PRV ≤ target, within 1%.
- **Clipping grid rule:** from the non-private B1 mechanics (seed 42, T = 1000 rounds, realised denominator, no test data),
  pool all per-client ‖ΔQ_u‖_F and take C ∈ {p25 (aggressive), p50 (moderate), p90 (loose)}, rounded to 2 significant
  figures. This selection step uses the private data non-privately and is outside the DP guarantee.
- **C selection:** at the reference ε ≈ 4 (seed 42), choose C by the **final-round (T) validation NDCG@10**. Freeze it for all ε.
- **Server learning rate:** keep η_s = 1 (B1). The **failure criterion** is that the selected-C ε ≈ 4 run's final validation
  NDCG@10 is below the popularity baseline's validation NDCG@10 (0.0464). Only if that holds, run η_s ∈ {0.5, 1, 2} at ε ≈ 4
  (validation only), select by final validation NDCG@10, and freeze before the sweep.
- **Matched no-DP control (B1-DPReady):** B1 + T = 1000 fixed rounds + qN denominator, with no clipping and no noise. Not tuned.
- **Sweep:** no-DP control and ε ∈ {8, 4, 2, 1}, seeds 42/123/2026, the same frozen C and η_s. Each round-T model is evaluated
  once on test.
- **Claim scope:** "The final B2 training mechanism, conditional on fixed hyperparameters and under the stated
  secure-aggregation assumption, is accounted as user-level (ε, δ)-DP." This does not cover baseline development,
  hyperparameter selection, validation monitoring, or the reproducible (non-cryptographic) PCG64 noise RNG.

### 2026-10-02 — B2 pre-sweep results (validation only; recorded BEFORE the sweep's test evaluation)

- **B1 client-update norms** (non-private B1 mechanics, seed 42, 1000 rounds, 93,915 updates; `results/b2_b1_update_norm_stats.csv`):
  mean 1.477, p25 1.048, p50 1.489, p75 1.895, p90 2.381, p95 2.747, max 9.25. Updates are smaller and more spread out in
  rounds 1–100 (median 0.99). By the declared rule, **C grid = {1.0, 1.5, 2.4}**.
- **Accounting** (`results/b2_accounting.csv`; q 0.1, T 1000, δ 1e-5):

  | Target ε | σ | ε (PRV, primary) | ε (RDP, cross-check) |
  |---|---|---|---|
  | 8 | 2.0647 | 7.937 | 8.564 |
  | 4 | 3.5617 | 3.968 | 4.290 |
  | 2 | 6.4339 | 1.995 | 2.159 |
  | 1 | 11.9836 | 0.999 | 1.081 |

  RDP is uniformly about 8% looser. The two are directionally consistent.
- **Sanity run** (ε ≈ 4, C = 1.5, seed 42): all mechanism checks pass (`results/b2_sanity_checks.csv`), but utility collapses:
  final validation NDCG@10 is 0.0092 (round 0: 0.0066; popularity 0.0464).
- **Clip search** (ε ≈ 4, final-round validation NDCG@10): C = 1.5 → 0.0092, C = 2.4 → 0.0089, C = 1.0 → 0.0081.
  **C = 1.5 selected and frozen.** For every C, 77–81% of clients are clipped (more than B1's distribution predicts, because
  updates grow under noise) and the per-round ‖clipped sum‖ / ‖noise‖ ≈ 0.0085.
- **The failure criterion was met** (0.0092 < 0.0464), so the server-lr check was run (C 1.5, ε ≈ 4): η_s 0.5 → 0.0105,
  1.0 → 0.0092, **2.0 → 0.0124**. Selected per the rule: **η_s = 2.0, frozen**. The differences are noise-level and the failure is
  not cured; the per-round SNR is unchanged (≈ 0.0086).
- Because B2's η_s now differs from B1's, the sweep runs **two** no-DP controls:
  - **B1-DPReady** (η_s = 1, qN, T = 1000): B1 → B1-DPReady is the protocol effect;
  - **matched no-DP** (η_s = 2, qN, T = 1000): B1-DPReady → matched is the η_s effect, and matched → B2 is the pure DP effect.

---

## 2026-10-02 — Phase B2 — User-Level DP Federated BPR

### Objective

Measure the incremental utility cost of adding formal user-level DP to the frozen federated system (B1).

### Mechanism

Frozen B1 round (Poisson q = 0.1, local full-batch SGD lr 5, E 2, L2 1e-5, persistent local p_u), then:

- **Clip** each client's whole shared update to Frobenius norm C, with a 1e-6 margin so that ‖·‖ ≤ C holds in float32.
- **Sum** the clipped updates; **add** N(0, σ²C² I_D) with D = 1682 × 64 = 107,648.
- **Update:** Q ← Q + η_s · (sum + noise) / (qN), with qN = 94.2.
- **Empty rounds** are noised and accounted.
- **Horizon:** exactly T = 1000 rounds; the round-T model is reported and there is no checkpoint selection.

Code: `src/privacy.py` (a subclass of B1's simulator) and `src/train_dp_federated.py`.

### Privacy unit and accounting

- One user, with add/remove-one-user adjacency. The sensitivity of the clipped sum is C. Each round is a Poisson-subsampled
  Gaussian mechanism (q, σ), composed over T rounds.
- Primary accountant: Opacus PRV; cross-check: Opacus RDP. δ = 1e-5.
- Achieved ε (PRV / RDP): 7.94 / 8.56 (σ 2.065), 3.97 / 4.29 (σ 3.562), 1.99 / 2.16 (σ 6.434), 1.00 / 1.08 (σ 11.98).
- C does not enter ε (tested).

### Clipping study (validation only)

- From B1: median client-update norm 1.49 (p25 1.05, p90 2.38). The declared rule gave the grid {1.0, 1.5, 2.4}.
- Results at ε ≈ 4: 0.0081 / 0.0092 / 0.0089. **C = 1.5 was frozen.**
- The failure criterion was met, so η_s ∈ {0.5, 1, 2} was checked: 0.0105 / 0.0092 / 0.0124, and **η_s = 2 was frozen by
  the rule** (noise-level differences).

### Matched no-DP controls (test, seeds 42/123/2026; paired per user, seed-averaged)

- **B1-DPReady** (η_s 1, qN, T = 1000): 0.0882 ± 0.0027.
  vs B1 frozen: **+0.0007 [−0.0014, +0.0028]**, so the protocol change has no material effect.
- **Matched no-DP** (η_s 2, qN, T = 1000): 0.0884 ± 0.0046.
  vs B1-DPReady: **+0.0002 [−0.0035, +0.0040]**, so η_s has no material effect.

### Main privacy sweep (test NDCG@10, mean ± std over 3 seeds; C 1.5, η_s 2, q 0.1, T 1000)

| Level | ε (PRV) | σ | NDCG@10 | HR@10 = Recall@10 | MRR@10 | Retention vs matched no-DP | Paired DP effect (95% CI) |
|---|---|---|---|---|---|---|---|
| matched no-DP | ∞ | 0 | 0.0884 ± 0.0046 | 0.1674 ± 0.0060 | 0.0648 ± 0.0040 | 1.000 | — |
| ε ≈ 8 | 7.94 | 2.065 | 0.0222 ± 0.0050 | 0.0456 ± 0.0074 | 0.0153 ± 0.0043 | 0.251 | −0.0662 [−0.0794, −0.0541] |
| ε ≈ 4 | 3.97 | 3.562 | 0.0193 ± 0.0058 | 0.0368 ± 0.0086 | 0.0141 ± 0.0050 | 0.218 | −0.0691 [−0.0823, −0.0571] |
| ε ≈ 2 | 1.99 | 6.434 | 0.0104 ± 0.0024 | 0.0202 ± 0.0055 | 0.0074 ± 0.0014 | 0.118 | −0.0780 [−0.0911, −0.0661] |
| ε ≈ 1 | 1.00 | 11.98 | 0.0054 ± 0.0022 | 0.0124 ± 0.0043 | 0.0034 ± 0.0017 | 0.061 | −0.0830 [−0.0962, −0.0708] |

- Reference points: popularity 0.0443, random 0.0038, B1 frozen 0.0875 ± 0.0021.
- **Every private level is below popularity.** ε ≈ 1 is close to random.
- Per-seed results are in `results/b2_privacy_sweep.csv`; per-user results in `results/b2_per_user_test.csv`; paired
  comparisons in `results/b2_paired.csv`.

### Diagnostics (means over rounds and seeds; `results/b2_diagnostics.csv`, `results/b2_summary.csv`)

| Level | Pre-clip median norm | Fraction clipped | Mean shrink factor | ‖clipped sum‖ | ‖noise‖ (theory σC√D) | SNR per round |
|---|---|---|---|---|---|---|
| no DP | 1.22 | 0 | 1 | 13.6 | 0 | — |
| ε ≈ 8 | 3.80 | 0.75 | 0.53 | 14.2 | 1016.2 (1016.1) | 0.0140 |
| ε ≈ 4 | 8.72 | 0.85 | 0.34 | 15.1 | 1752.9 (1752.9) | 0.0086 |
| ε ≈ 2 | 20.3 | 0.89 | 0.21 | 15.6 | 3166.5 (3166.4) | 0.0049 |
| ε ≈ 1 | 44.4 | 0.91 | 0.15 | 15.7 | 5897.9 (5897.7) | 0.0027 |

- **The noise accumulates in Q as a random walk.** At round T (seed 42), the mean item-vector norm is 16.4 / 28.4 / 51.4 / 95.9 for
  ε ≈ 8 → 1, against a predicted pure-noise row norm of η_s σC√T / (qN) · √d = 16.6 / 28.7 / 51.8 / 96.5. The no-DP value is 0.85.
- **Consequence:** the item embeddings are noise-dominated. Clients then compute larger gradients against noisy Q, so update
  norms inflate (median up to 44), most clients are clipped hard, and the clipped signal (‖sum‖ ≈ 15) is about
  1/70 – 1/370 of the per-round noise.
- **Noise-dimension diagnostic for B4:**
  - D = 107,648 and E‖Z‖² = Dσ²C², i.e. a noise norm of about σC√D = 328 σC.
  - The per-coordinate noise std of the update is η_s σC/(qN). It is independent of D, and so is ε.
  - Reducing the shared dimension reduces the total noise norm and the per-item noise norm (∝ √d), but **not ε**.

### Communication

Unchanged from B1: 861,184 B per selected client per round (dense Q down, dense clipped ΔQ_u up; float32). About 81.1 GB per
1000-round run. Noise is added after (assumed) secure aggregation and does not change the tensor size. **B2 is the full-rank DP
communication reference for B4.** DP does not reduce communication.

### Tests

**92 passing**: 73 prior, plus 19 in `tests/test_privacy.py`. The new tests cover:
- the Frobenius norm of the whole update;
- clipping (unchanged below C, bounded above C, one global scale);
- a toy end-to-end example in which a giant update contributes at most C;
- the noise shape and an empirical std of σC on the sum;
- the qN denominator;
- empty rounds being noised and counted;
- noise-seed reproducibility, and change under a different seed;
- ε monotone in σ, T and q, and independent of C;
- the solver hitting its target within 1%;
- σ = 0 giving no finite guarantee;
- p_u never clipped or uploaded;
- unselected p_u unchanged;
- **bit-identity with B1** when C = ∞, σ = 0 and the denominator is realised (toy and real split);
- the split fingerprint;
- pinned MD5 hashes of B1's results;
- checkpoint privacy metadata and rejection of a mismatched split or privacy configuration.

### Problems encountered

1. **Float32 clipping.** A clipped update came out at 1.000000024·C. Fixed with a 1e-6 margin plus a runtime assertion. The
   sensitivity bound now holds exactly in float32.
2. **Accounting precision.** σ was first stored to 6 decimals, so recomputed ε differed in the last digits and a sanity check
   failed. σ is now stored at full precision. ε was always computed by the accountant, never typed in.
3. **Utility collapse.** It appeared already in the sanity run and was robust to C and η_s. The validation-only server-lr check
   (pre-declared failure rule) did not cure it. η_s = 2 was frozen per the rule, although the differences were noise-level.
4. **The control had to be split in two** (B1-DPReady with η_s = 1, and matched no-DP with η_s = 2) once η_s changed.
   Both are negligible relative to B1.
5. **Analysis-script bug.** A renamed column broke plotting after all training had finished. It was fixed and the analysis was
   re-run from saved files, with no retraining and no new test evaluation.
6. **RDP warning.** For some (σ, T) pairs in the tests, RDP reports that its optimal order is the largest α, so the RDP bound is
   loose there. PRV is primary and was declared before the sweep.

### Limitations

- The DP claim is narrow: **the final B2 training mechanism, conditional on fixed hyperparameters and under the stated
  secure-aggregation assumption, is accounted as user-level (ε, δ)-DP.** Several things are outside the guarantee:
  - the clipping grid (from non-private B1 statistics);
  - the choice of C and η_s (on validation);
  - validation monitoring;
  - all non-private baselines.
- Secure aggregation is simulated/assumed. The noise RNG is a reproducible PCG64 stream, not cryptographically secure.
- With only N = 942 clients (about 94 per round) and about 100 participations per user, the user-level noise needed for ε ≤ 8 is
  large relative to the clipped aggregate. These results describe this small-population regime and do not speak about DP
  federated recommendation in general.
- Three seeds; one dataset; no per-ε tuning of C or η_s (by design).

### Decision

**Freeze B2** (`configs/b2.yaml`: C 1.5, η_s 2, q 0.1, T 1000, δ 1e-5, PRV primary, σ from `results/b2_accounting.csv`).
- **The mechanism is verified:** clipping, noise scale, accounting, locality, and bit-identity with B1 when DP is off.
- **The controls are clean:** protocol and η_s effects are negligible.
- **The privacy–utility result is stable across seeds and monotone in ε.**

It shows a **severe** DP cost in this regime (6–25% NDCG@10 retention). The diagnostics locate the cause in noise accumulating in
the 107,648 shared coordinates, which is precisely the quantity that B3/B4's lower-dimensional shared representation changes.
Whether that helps at a fixed ε is now an open, testable question, not a premise.
