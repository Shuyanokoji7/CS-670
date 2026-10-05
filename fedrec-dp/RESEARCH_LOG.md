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

> **Superseded in part (annotation added 2026-10-02, under Codex supervision).** The "Decision: Freeze B2" at the end of
> this entry is **withdrawn**. B2 was reopened by the later entry "B2 robustness check: privacy-aware training horizon"
> because T = 1000 was inherited from B1's non-private convergence. The T = 1000 results below are the **provisional /
> initial B2** result and are archived in `results/raw/superseded_b2_T1000/`. The entry text below is preserved unchanged.

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

---

## 2026-10-02 — B2 robustness check: privacy-aware training horizon (protocol declared BEFORE any run)

**Why.** T = 1000 was inherited from B1's *non-private* convergence. At a fixed ε, fewer private rounds means fewer
compositions and therefore a smaller σ. The T = 1000 result may be confounded by the horizon.

**Bug found during verification (results unaffected).** B2 checkpoints were named `b2_{level}_seed{seed}.pt`. Both the
matched no-DP control (η_s 2) and B1-DPReady (η_s 1) are level `nodp`, so B1-DPReady's checkpoints overwrote the matched
control's. Reloading them reproduces B1-DPReady's saved test scores exactly. All reported B2 numbers came from result and
per-user files whose names include η_s, so they are correct. The runner now names every output with T, C and η_s.

**Archive.** All initial B2 outputs (fixed T = 1000) are in `results/raw/superseded_b2_T1000/` (139 files, with a known-issue
note).

**Protocol (validation only until the final freeze entry):**
1. **T sweep:** T ∈ {100, 250, 500, 1000} at target ε = 4, δ 1e-5, q 0.1, C 1.5, η_s 2, local lr 5, E 2, full-rank Q,
   seed 42.
   - σ is **re-solved for every T** (PRV primary, smallest σ with ε_PRV ≤ 4 within 1%), with an RDP cross-check.
   - Each run is exactly T rounds with no early stopping. The metric is the **final-round (T) validation NDCG@10**.
2. **Boundary rule (applied once only):** if the best T is 100, add T = 50; if the best T is 1000, add T = 1500. σ is re-solved
   in both cases. 250 or 500 need no extension.
3. **Seed rule:** if the top two T differ by < 0.003 validation NDCG@10 on seed 42, run both on seeds 123 and 2026 and choose
   by the mean over seeds 42/123/2026. Otherwise select the seed-42 winner provisionally and document the uncertainty.
4. **C recheck:** at the selected T, the existing grid C ∈ {1.0, 1.5, 2.4} only (ε ≈ 4, η_s 2, seed 42, validation). σ is
   determined by (ε, q, T), not by C. Select by highest validation NDCG@10.
   - The optional "prefer the smaller C when the top two are within 0.003" is **not** applied, because it would conflict with
     the pre-declared argmax rule whenever the smaller C is not the top scorer. The existing rule is preserved.
5. **η_s recheck** {0.5, 1, 2} only if the new T **materially changes the utility curve**, defined here as: the selected
   T ≠ 1000 **and** its seed-42 validation NDCG@10 exceeds the T = 1000 value by more than 0.003. In that case, recheck at
   the selected T and C (ε ≈ 4, seed 42, validation) and select by highest validation NDCG@10. No new values.
6. **Freeze** T, C and η_s in a dated log entry, then set `privacy.protocol_frozen: true`. The runner refuses any test
   evaluation unless the protocol is marked frozen. Only then re-solve σ for ε ∈ {8, 4, 2, 1} at the final T and run the
   sweep: matched no-DP control (same T, qN, η_s), B1-DPReady (η_s 1) if η_s ≠ 1, and the four ε levels, on seeds
   42/123/2026.

---

## 2026-10-02 — B2 robustness: audit and protocol-test notes (Codex supervising; written BEFORE the T sweep)

- **Read-only audit (accepted by the supervisor):**
  - 92 tests passed;
  - `u.data` SHA-256 `06416e59…a490` matches `src/data.py` and `data_summary.json`, and all 23 raw files match the
    session-start snapshot;
  - the split fingerprint `6faed6fc…6989` matches;
  - all B1 output hashes match;
  - no experiment was running.

  The repository is a git working tree. Commit `3d13c47` (user) predates the robustness setup.
- **Runner changes (not yet exercised on real T-sweep runs):**
  - `t_sweep` takes an injectable run function, returns `(rows, selected T, decision)`, and writes
    `results/b2_t_sweep.csv` and `results/b2_t_selection.csv`. The boundary rule (one expansion) and the top-two seed rule
    (< 0.003, seeds 123/2026, 3-seed mean) are applied in code.
  - **σ precision fix:** `sigma_for` returned the solver's full-precision σ on a fresh solve but the 12-significant-digit
    cached value afterwards (difference about 1e-12). σ is now rounded to the stored precision *before* ε is computed and
    stored, so every run uses exactly the accounted value. No reported number is affected: the T = 1000 σ values were
    already read from the cache.
- **New tests** in `tests/test_b2_protocol.py` (12). They run in temporary directories with an RDP copy of the config, and a
  module guard fails if any file under `results/` or `checkpoints/` is created or modified. They cover:
  - σ is freshly solved and cached per T, and the stored ε equals the accountant's ε for that T;
  - the required σ is monotone increasing in T at fixed ε;
  - exactly T steps are executed and accounted, including all-empty rounds (which are still noised);
  - a validation-only `run_one` never evaluates test (spies on all three evaluator references) and writes no checkpoint;
  - unfrozen protocol: both direct test jobs and `sweep` are refused before any training;
  - `run_tag` distinguishes T, C and η_s (the old collision);
  - frozen test jobs write distinct `b2_{level}_T{T}_C{C}_slr{η_s}_seed{s}.pt`, with T in the checkpoint metadata;
  - T-selection rules with fake run results: lower-edge expansion to 50 happens once only (even if 50 is best), upper-edge
    expansion to 1500, no expansion for an interior best, the seed rule with 3-seed-mean selection when the gap is < 0.003,
    and no seed rule when the gap is ≥ 0.003.

  The suite now has **104 passing** tests.
- **Next (validation only):** `python experiments/run_b2.py --t-sweep` with PRV σ re-solved per T at ε = 4, C 1.5, η_s 2,
  q 0.1, δ 1e-5, local lr 5, E 2, exact final-round evaluation and no early stopping. Stop after T selection, before the C
  recheck, the freeze, or any test evaluation.

### 2026-10-02 — B2 T-sweep results (validation only; NO test evaluation; C recheck / freeze NOT yet done)

Command: `python experiments/run_b2.py --t-sweep --workers 6`, exit 0. Outputs: `results/b2_t_sweep.csv`,
`results/b2_t_selection.csv`, `results/raw/b2_t_sweep_log.txt`, `results/raw/b2_runs/*_T{50,100,250,500,1000}_*`.
Settings: target ε 4 (PRV, σ re-solved per T), δ 1e-5, q 0.1, C 1.5, η_s 2, local lr 5, E 2, exactly T rounds, final-round
validation NDCG@10.

| Stage | T | Seed | σ | ε PRV | ε RDP | Val NDCG@10 | HR@10 | MRR@10 | Clipped | Median pre-clip ‖Δ‖ | ‖clipped sum‖ | ‖noise‖ | SNR | Final item norm (pure-noise pred.) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grid | 100 | 42 | 1.3953 | 3.968 | 4.399 | 0.0229 | 0.0520 | 0.0142 | 0.44 | 1.33 | 14.52 | 686.7 | 0.0212 | 3.56 (3.56) |
| grid | 250 | 42 | 1.9308 | 3.983 | 4.342 | 0.0158 | 0.0340 | 0.0104 | 0.62 | 1.97 | 14.62 | 950.1 | 0.0154 | 7.70 (7.78) |
| grid | 500 | 42 | 2.5880 | 3.980 | 4.315 | 0.0202 | 0.0361 | 0.0153 | 0.78 | 4.10 | 15.07 | 1273.6 | 0.0118 | 14.53 (14.74) |
| grid | 1000 | 42 | 3.5617 | 3.968 | 4.290 | 0.0124 | 0.0276 | 0.0078 | 0.85 | 8.73 | 15.03 | 1752.8 | 0.0086 | 28.37 (28.70) |
| boundary | 50 | 42 | 1.1519 | 3.964 | 4.497 | 0.0230 | 0.0488 | 0.0154 | 0.36 | 1.10 | 13.65 | 567.0 | 0.0241 | 2.10 (2.08) |
| seed rule | 50 | 123 | 1.1519 | 3.964 | 4.497 | 0.0370 | 0.0637 | 0.0290 | 0.34 | 1.09 | 13.80 | 566.7 | 0.0244 | 2.08 (2.08) |
| seed rule | 50 | 2026 | 1.1519 | 3.964 | 4.497 | 0.0271 | 0.0467 | 0.0212 | 0.36 | 1.12 | 13.71 | 566.8 | 0.0242 | 2.09 (2.08) |
| seed rule | 100 | 123 | 1.3953 | 3.968 | 4.399 | 0.0261 | 0.0510 | 0.0187 | 0.41 | 1.29 | 14.49 | 686.7 | 0.0211 | 3.55 (3.56) |
| seed rule | 100 | 2026 | 1.3953 | 3.968 | 4.399 | 0.0232 | 0.0446 | 0.0170 | 0.45 | 1.34 | 14.66 | 686.5 | 0.0213 | 3.55 (3.56) |

The T = 1500 σ (4.2919, ε PRV 3.996) was pre-solved but not run, because the upper boundary was not triggered.

**Selection, applied literally by code:**
1. The grid best on seed 42 was T = 100, the lower edge, so the one-time boundary expansion added T = 50.
2. The top two on seed 42 were T = 50 (0.0230) and T = 100 (0.0229), a gap of 0.0002, below 0.003. Both were therefore run on
   seeds 123 and 2026. Three-seed means: **T = 50: 0.0291**, T = 100: 0.0241.
3. **Selected T = 50.**

**Caveats:**
- T = 50 is at the edge of the explored range. The declared rule allows one expansion only, so smaller T was not tested.
- The seed-42 grid is non-monotone (250 is below 500), which is consistent with seed noise at this scale.
- Every value is still well below popularity's validation NDCG@10 (0.0464).
- Final item norms track the pure-noise random-walk prediction at every T. That is consistent with accumulated Gaussian
  perturbation dominating Q, but it does not establish causation.
- At T = 50 the matched no-DP control will train for only 50 rounds (B1 round-50 validation was about 0.05). Any B1 → control
  horizon effect must be reported separately and not attributed to DP.

**Next, pending supervisor approval:** C recheck at T = 50 (validation only), then the conditional η_s check, the freeze entry,
and only then test evaluation.

### 2026-10-02 — Audit correction: pre-freeze real-data test-path regression checks; cache-key fix

- **What happened.** The first version of `test_frozen_test_jobs_write_distinct_T_C_slr_outputs` redirected outputs and
  config to a temporary directory, but still loaded the **real** frozen MovieLens split. Its three `evaluate_test=True`
  jobs (ε≈4, no-DP η_s 2, no-DP η_s 1; **T = 2 rounds**; seed 42) therefore evaluated the **real held-out test labels**
  before the final B2 freeze. This happened in 3 pytest sessions, i.e. **9 test evaluations of 2-round models**:
  - the first file-only run (its temporary directory was auto-deleted by pytest's retention policy);
  - the file rerun;
  - the full 104-test suite.
- **Not used.** These test metrics were never printed, inspected or used, for T, C or η_s selection or for anything else.
  The T sweep and all selections used validation only.
- **Evidence.** Preserved unopened in `results/raw/audit_pre_freeze_test_regression/`: copies of the two retained sessions'
  temporary directories, `MANIFEST.sha256` (40 files) and a README. While copying, my own shell bug created a stray
  directory. It was verified byte-identical to the two preserved copies (`diff -rq`) and then removed, and the manifest
  was rebuilt.
- **Accurate statement for the freeze entry:** no **experimental** test evaluation has been performed under the revised B2
  protocol. The only real-test evaluations since B2 was reopened are these 9 accidental regression-test evaluations of
  2-round models. They are disclosed here and were not used.
- **Fix.**
  - Every runner test now uses a **synthetic** split: 30 users × 40 items, built by the real split code. `load_all` is
    replaced in the isolation fixture, and an evaluator spy on all three evaluator references asserts that every
    evaluation, including the test path, receives the synthetic split.
  - The isolated test config sets T, C and η_s explicitly (T 3, C 1.5, η_s 2, RDP), so production config updates cannot
    break the filename assertions.
  - A check of the other test files found no model evaluation on the real test labels. Their "test" references are Phase 0
    split-structure checks, or a hand-made toy split.
- **σ-cache key fix.**
  - `cached_accounting_row()` now matches the **full key**: target ε, T, q, δ, primary accountant and cross-check
    accountant. The stored ε must also lie in [target·(1−tol), target].
  - Rows for other keys are kept and not reused. Conflicting rows for the same key raise an error.
  - `sigma_for`, `accounting()` and `sanity()` all use this one function. Previously, `accounting()` and `sanity()` filtered
    on (T, target ε) only.
  - All existing cache rows are PRV / RDP, q 0.1, δ 1e-5, and still match, so the completed T sweep's σ values are unchanged
    and valid. The T sweep was **not** rerun.
- **Tests:** `tests/test_b2_protocol.py` now has 21 tests:
  - the synthetic-split validity check;
  - cache-key mismatch (primary, cross-check, q, δ) forcing a fresh solve while keeping the old row;
  - full-key reuse without solving;
  - an out-of-tolerance row not being reused;
  - conflicting rows raising;
  - `accounting()` using full-key rows.

  The full suite has **113 passing** tests.

### 2026-10-02 — B2 C and η_s rechecks at the selected T = 50 (validation only; NOT frozen; no experimental test evaluation)

The supervisor accepted T = 50 under the declared boundary and seed rules. Before any overwrite, the T = 50 validation run
files and the T-selection outputs were archived to `results/raw/archive_before_C_lr_recheck_2026-10-02/` (20 files plus
`MANIFEST.sha256`). The T = 1000 archive and the full T-selection table are preserved. Settings: target ε 4 (PRV σ 1.1519,
ε 3.964, RDP 4.497), q 0.1, δ 1e-5, T 50, local lr 5, E 2, seed 42, final-round validation NDCG@10.

**C recheck** (existing grid only, η_s 2; `results/b2_clip_search_T50.csv`):

| C | Val NDCG@10 | HR@10 | MRR@10 | Clipped | Median pre-clip ‖Δ‖ | ‖clipped sum‖ | ‖noise‖ | SNR | Item norm (pure-noise pred.) |
|---|---|---|---|---|---|---|---|---|---|
| **1.0** | **0.0332** | 0.0711 | 0.0218 | 0.51 | 1.02 | 10.41 | 378.0 | 0.0275 | 1.41 (1.38) |
| 1.5 | 0.0230 | 0.0488 | 0.0154 | 0.36 | 1.10 | 13.65 | 567.0 | 0.0241 | 2.10 (2.08) |
| 2.4 | 0.0105 | 0.0244 | 0.0064 | 0.09 | 1.24 | 16.50 | 907.1 | 0.0182 | 3.33 (3.32) |

- **Selected C = 1.0** (exact argmax).
- **Caveat:** 1.0 is the smallest value in the declared grid, which was not expanded.
- The C = 1.5 rerun was byte-identical to the T-sweep run (result, per-user and rounds files; history identical apart from
  wall time).

**η_s recheck** (condition met: the selected T's seed-42 validation 0.023047 exceeds T = 1000's 0.012391 by more than 0.003),
existing values {0.5, 1, 2} only, at C 1.0 (`results/b2_server_lr_check_T50.csv`):

| η_s | Val NDCG@10 | HR@10 | MRR@10 | Clipped | ‖clipped sum‖ | SNR | Item norm (pure-noise pred.) |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.0213 | 0.0414 | 0.0151 | 0.25 | 7.28 | 0.0193 | 0.36 (0.35) |
| **1.0** | **0.0344** | 0.0669 | 0.0247 | 0.37 | 9.14 | 0.0242 | 0.71 (0.69) |
| 2.0 | 0.0332 | 0.0711 | 0.0218 | 0.51 | 10.41 | 0.0275 | 1.41 (1.38) |

- **Selected η_s = 1.0** (exact argmax). Its margin over η_s 2 (0.0012) is well within seed-to-seed variation, so this
  selection should not be read as a reliable difference.
- The η_s 2 / C 1.0 rerun was byte-identical to the C-recheck run.
- Since η_s = 1.0 equals B1's value, the matched no-DP control (T 50, qN, η_s 1, no clipping, no noise) **is** the B1-DPReady
  control at T = 50, and the sweep will run a single control.

**Selected (pending the freeze entry): T = 50, C = 1.0, η_s = 1.0.**
- Every configuration is still below popularity's validation NDCG@10 (0.0464).
- Final item norms track the pure-noise random-walk prediction in every run. That is consistent with accumulated Gaussian
  perturbation being a major component of Q, but not conclusive.
- `configs/b2.yaml` now has T 50, C 1.0, server_lr 1.0, and `protocol_frozen: false`.

---

## 2026-10-02 — FINAL B2 protocol FROZEN (privacy-aware horizon) — written BEFORE any experimental test evaluation

Reviewed and accepted by the supervisor (Codex).

**Frozen protocol**
- User-level DP with add/remove-one-user adjacency.
- Poisson client sampling, q = 0.1. Empty rounds are noised and accounted.
- **T = 50** private rounds, exactly. No early stopping and no checkpoint selection.
- Whole-client clipping, **C = 1.0**. Gaussian noise N(0, σ²C² I) on the clipped sum; fixed denominator qN = 94.2.
- **η_s = 1.0.** Local full-batch SGD lr 5, E 2, L2 1e-5; d = 64.
- δ = 1e-5. Accounting: PRV primary, RDP cross-check. Target ε ∈ {8, 4, 2, 1}.
- σ is re-solved at T = 50 with `--accounting`. For ε≈4 the verified T = 50 solve is reused (σ 1.15192871094, ε_PRV 3.964),
  since C and η_s do not affect ε.

**Selection logic** (validation only, final-round NDCG@10, all rules declared beforehand):
- **T** (`results/b2_t_sweep.csv`, `results/b2_t_selection.csv`): the grid {100, 250, 500, 1000} at ε≈4 put T = 100 best on
  seed 42, the lower edge, so the one-time boundary expansion added T = 50. The seed-42 top two (T 50: 0.0230, T 100: 0.0229)
  were separated by 0.0002, below 0.003, so the 3-seed means decided: **T 50 0.0291** vs T 100 0.0241.
- **C** (`results/b2_clip_search_T50.csv`): exact argmax over the existing grid {1.0, 1.5, 2.4}: **1.0 (0.0332)** vs
  0.0230 / 0.0105.
- **η_s** (`results/b2_server_lr_check_T50.csv`): the condition was met (0.023047 vs 0.012391, a gap of more than 0.003).
  Exact argmax over {0.5, 1, 2}: **1.0 (0.0344)** vs 0.0213 / 0.0332.

**Caveats**
- T 50 and C 1.0 lie at the edges of their explored grids. **There will be no further grid expansion.**
- The η_s margin over 2.0 (0.0012) is not a reliable difference.
- C and η_s were selected on seed 42 only.

**Controls and comparisons (declared now)**
- The matched no-DP control uses the same T 50, qN and η_s 1.0, with no clipping and no noise. Because η_s = B1's, this is
  the B1-DPReady control at T = 50.
- B1 → matched control is reported as a protocol/horizon effect, **never** as a DP effect. Matched control → B2 is the DP effect.
- Retention = DP / matched no-DP. Popularity (0.0443) is the external reference.
- The initial T = 1000 B2 is retained as the **superseded fixed-1000-round result** (`results/raw/superseded_b2_T1000/`).

**Test-set status**
- **No experimental test-set evaluation has been performed under this revised B2 protocol.**
- Disclosed exception: before the freeze, a regression test made 9 accidental real-data test evaluations of 2-round
  models. They were never inspected or used (`results/raw/audit_pre_freeze_test_regression/`; see the audit-correction entry).

**Next:** `--accounting` at T 50, then exactly one `--sweep --seeds 42 123 2026` (matched control plus ε 8/4/2/1, test
evaluated once per run), then analysis.

### 2026-10-02 — FINAL B2 results (frozen privacy-aware protocol; one sweep; test evaluated once per run)

> **Correction (supervisor review, same day):** the interpretation bullet "The privacy-aware horizon materially changed the
> B2 result … confounded by the horizon" over-attributes the change to T alone. The final protocol changed T, C **and** η_s
> together, so the improvement is not attributable to T alone. See the clarification entry below. Entry text preserved.

Commands, exact, from `fedrec-dp/`, also logged in `results/raw/b2_final_command_log.txt`:
- `.venv/bin/python experiments/run_b2.py --accounting`
- `.venv/bin/python experiments/run_b2.py --sweep --seeds 42 123 2026 --workers 15` (exit 0, log
  `results/raw/b2_final_sweep_log.txt`)

The sweep ran 15 runs: the matched no-DP control and ε≈8/4/2/1, on seeds 42/123/2026. Each ran exactly T = 50 rounds, with
no per-ε tuning, no early stopping and no checkpoint selection.

**Accounting (T 50, q 0.1, δ 1e-5)**

| Target ε | σ (stored, full precision) | ε PRV (primary) | ε RDP (cross-check) |
|---|---|---|---|
| 8 | 0.805072021484 | 7.9675 | 9.1198 |
| 4 | 1.15192871094 | 3.9639 | 4.4973 |
| 2 | 1.76044921875 | 1.9933 | 2.2206 |
| 1 | 2.97749023438 | 0.9901 | 1.0872 |

ε was recomputed independently from each stored σ and matched to about 1e-10. All values lie in [0.99·target, target].
For ε≈4, the verified T = 50 solve was reused.

**Test results** (mean ± std over seeds; HR@10 = Recall@10 exactly, because there is one held-out item per user)

| Level | NDCG@10 | HR@10 = Recall@10 | MRR@10 | Retention vs matched no-DP |
|---|---|---|---|---|
| matched no-DP (T 50, qN, η_s 1) | 0.0582 ± 0.0021 | 0.1090 ± 0.0034 | 0.0429 ± 0.0021 | 1.000 |
| ε≈8 | 0.0544 ± 0.0007 | 0.1030 ± 0.0021 | 0.0397 ± 0.0005 | 0.935 |
| ε≈4 | 0.0456 ± 0.0010 | 0.0839 ± 0.0046 | 0.0340 ± 0.0020 | 0.783 |
| ε≈2 | 0.0274 ± 0.0012 | 0.0502 ± 0.0067 | 0.0205 ± 0.0003 | 0.470 |
| ε≈1 | 0.0115 ± 0.0008 | 0.0226 ± 0.0012 | 0.0082 ± 0.0011 | 0.198 |

- Validation means: no-DP 0.0535, ε8 0.0480, ε4 0.0396, ε2 0.0226, ε1 0.0103.
- Against popularity (0.0443; compared on means only): ε8 +0.0101, ε4 +0.0013, ε2 −0.0169, ε1 −0.0328. The matched no-DP
  control is +0.0139 above it.

**Paired per-user test NDCG@10** (seed-averaged; 10,000 user bootstrap resamples; `results/b2_paired.csv`)

| Comparison | Mean difference | 95% CI | Improved / degraded / tied |
|---|---|---|---|
| matched no-DP (T 50) − B1 frozen: **protocol/horizon effect** | −0.0293 | [−0.0443, −0.0143] | 10.0% / 21.2% / 68.8% |
| ε≈8 − matched no-DP: DP effect | −0.0038 | [−0.0094, +0.0017] | 10.0% / 10.3% / 79.7% |
| ε≈4 − matched no-DP: DP effect | −0.0126 | [−0.0201, −0.0056] | 9.3% / 11.3% / 79.4% |
| ε≈2 − matched no-DP: DP effect | −0.0308 | [−0.0407, −0.0216] | 6.8% / 12.4% / 80.8% |
| ε≈1 − matched no-DP: DP effect | −0.0467 | [−0.0582, −0.0360] | 4.1% / 13.3% / 82.6% |

At ε≈8 the interval crosses zero, so **no reliable utility difference from the matched control was detected**.

**Diagnostics** (means over rounds and seeds; `results/b2_summary.csv`, `results/b2_diagnostics.csv`)

| Level | Clipped | Median pre-clip ‖Δ‖ | Shrink factor | ‖clipped sum‖ | ‖noise‖ (σC√D) | SNR | Final item norm (pure-noise pred.) |
|---|---|---|---|---|---|---|---|
| no DP | 0 | 0.39 | 1 | 11.89 | 0 | — | 0.111 (—) |
| ε≈8 | 0.32 | 0.68 | 0.86 | 8.86 | 264.1 (264.1) | 0.034 | 0.500 (0.484) |
| ε≈4 | 0.38 | 0.81 | 0.83 | 9.32 | 377.9 (377.9) | 0.025 | 0.705 (0.692) |
| ε≈2 | 0.45 | 0.98 | 0.80 | 9.82 | 577.5 (577.6) | 0.017 | 1.066 (1.057) |
| ε≈1 | 0.54 | 1.17 | 0.77 | 10.26 | 976.8 (976.9) | 0.011 | 1.789 (1.788) |

D = 107,648. Communication is 861,184 B per selected client per round, about 4.04 GB per 50-round run (T = 1000: about 81 GB).

**Interpretation (qualified)**
- **The privacy-aware horizon materially changed the B2 result.** At the same ε≈4, test NDCG@10 rose from 0.0193 (initial,
  T 1000, C 1.5, η_s 2; superseded) to 0.0456. The initial result was therefore confounded by the horizon inherited from
  non-private B1. It remains archived as the **superseded fixed-1000-round B2 result**.
- **The short horizon has its own large cost.** The non-private control at T = 50 is 0.0293 below frozen B1. Final B2 utility
  is limited both by DP noise and by the short training horizon that a fixed ε budget favours.
- **The DP effect grows with privacy strength.** It is not reliably detected at ε≈8 and is clear at ε ≤ 4.
- **Item-vector norms remain close to the pure-noise random-walk prediction at every ε.** That is consistent with
  accumulated Gaussian perturbation dominating the *magnitude* of Q. Utility nonetheless survives at ε 8/4, so norm dominance
  alone does not determine ranking utility. The diagnostics are consistent with accumulated high-dimensional Gaussian
  perturbation being a major cause of the remaining DP loss, and B4 will test this directly.

**Limitations**
- T 50 and C 1.0 lie at the edges of their explored grids (no further expansion by rule).
- C and η_s were selected on a single seed; T was confirmed on three seeds by the seed rule.
- Three seeds per final configuration; one dataset.
- The popularity comparison uses means, not a paired test.
- Privacy claim scope: the final B2 training mechanism, conditional on fixed hyperparameters and under the stated
  secure-aggregation assumption, is accounted as user-level (ε, δ)-DP. Not covered: hyperparameter/T selection on validation,
  the non-private statistics, the simulated secure aggregation, and the reproducible PCG64 RNG.
- Disclosed exception: 9 accidental pre-freeze real-data test evaluations of 2-round models by a regression test, never used.

**Decision.** B2 final = this frozen protocol and these results. B3/B4 are not started, and no authorization has been relayed.

### 2026-10-02 — Clarification of the FINAL B2 conclusions (after supervisor review; no new experiments)

- **Initial vs final.** The revised protocol (T 1000 → 50, C 1.5 → 1.0, η_s 2 → 1) materially improves the private point
  estimates compared with the superseded fixed-1000-round B2. At ε≈4, test NDCG@10 is **0.045590 vs 0.019306**. All three
  settings changed together, so this change is **not attributable to T alone**.
- **Horizon/protocol effect, reported separately from DP.** The final matched no-DP control scores **0.058196**. B1 →
  matched control: **−0.029344, 95% CI [−0.044349, −0.014269]**.
- **DP effects vs the matched control:**
  - **ε≈8:** 0.054411, 93.5% retention, DP effect **−0.003785 [−0.009443, +0.001674]**. The interval includes 0: **no
    reliable DP loss detected** at ε≈8.
  - **ε≈4:** 0.045590, 78.3% retention, DP effect **−0.012606 [−0.020120, −0.005582]**.
  - **ε≈2:** 0.027378, 47.0% retention, −0.0308 [−0.0407, −0.0216].
  - **ε≈1:** 0.011526, 19.8% retention, −0.0467 [−0.0582, −0.0360].
- **Popularity** (0.044292): point estimates only. Popularity per-user test outputs are not saved, and no rescoring was
  done, so **no paired or significance claim** is made against popularity.
  - ε≈8 is above it (+0.010119).
  - ε≈4 is near it (+0.001298).
  - ε≈2 (−0.016914) and ε≈1 (−0.032766) are below it.
  - The initial-protocol statement "every private level falls below popularity" applies **only** to the superseded
    T = 1000 B2, not to final B2.
- **Accounting at T 50:** PRV / RDP, with RDP looser by a different amount at each level:

  | Target ε | ε PRV | ε RDP | RDP looser by |
  |---|---|---|---|
  | 8 | 7.9675 | 9.1198 | 14.5% |
  | 4 | 3.9639 | 4.4973 | 13.5% |
  | 2 | 1.9933 | 2.2206 | 11.4% |
  | 1 | 0.9901 | 1.0872 | 9.8% |

  The "about 8%" figure in the initial-B2 entry described T = 1000 only.
- **Diagnostics.** Noise norms and item-vector norms close to the pure-noise random-walk prediction are consistent with
  large accumulated Gaussian perturbation. They do **not** prove that dimensionality is the cause.
- **Implication for B4.** Final B2 retains 93.5% at ε≈8 and is near popularity at ε≈4. A fair B4 comparison, at the same ε,
  δ, q and T-selection rules and with the same accountant, is therefore **more demanding** than the superseded T = 1000
  result suggested.

### 2026-10-02 — Final documentation/artifact corrections (no training, no tests)

- `docs/B2_RESULTS.md`: the realised noise norm now "closely" matches σC√D; it previously said "exactly". Per-round means
  differ slightly.
- `docs/CODEX_HANDOFF.md` (the previous copy is preserved in `docs/archive/`):
  - the Environment block, Phase summary (adds a FINAL B2 row) and Verification snippet now reflect 113 tests and final B2;
  - §4 is labelled as historical;
  - the weighting bullet now reads: point estimates suggest a weighting contribution, but all overall CIs include zero, so
    no reliable difference is detected;
  - dense transmission is described as *this project's* dense reference protocol, not a universal requirement.
- Freeze snapshot `results/raw/b2_freeze_snapshot_20261002T172616/`: the original `SNAPSHOT_SHA256.txt` is preserved. It is
  not a reliable check manifest, because it contains a self-hash and hashes of live repository files. A new
  `MANIFEST.sha256` (relative paths, excluding itself) and `NOTE_MANIFEST.md` were added and verified with `sha256sum -c`.
  The freeze timestamp (17:26:16) and the snapshot contents are unchanged.

---

## 2026-10-02 — B3/B4 protocol (low-rank shared item matrix), reviewed and approved by the supervisor — written BEFORE implementation

**Authorization:** the user authorised B3 and B4 (relayed by the Codex supervisor). No commits or pushes.

**Research-development history (disclosed).**
- The test split has already been exposed during development: B0/B0-UW/B1/B2 test evaluations, including 9 accidental
  pre-freeze B2 regression-test evaluations, and B3 test scoring will follow.
- There is **no pristine holdout**, and **no DP claim covers the full research pipeline**. DP claims are limited to each final
  training mechanism, conditional on fixed hyperparameters, under the simulated secure-aggregation assumption.

**Model (B3 and B4).**
- score(u, i) = p_u · q_i. p_u ∈ R^64 is local: never clipped or uploaded.
- Shared item matrix Q = A B, with A ∈ R^{1682×r} and B ∈ R^{r×64}, r ∈ {4, 8, 16, 32}. D_r = r(1682+64) = 6,984 / 13,968 /
  27,936 / 55,872, against full D = 107,648.
- Dense payload per selected client per round: 2·4·D_r B = 55,872 / 111,744 / 223,488 / 446,976 B, against 861,184 B.
- **Objective:** exactly B1's BPR objective and regulariser, on p_u and the effective touched rows q_i = A_i B. The existing
  regularised g_Q = ∂L/∂q is chained through A and B (∂A_i = g_Q,i Bᵀ, ∂B = Σ A_iᵀ g_Q,i).
  - There are **no factor-only ‖A‖ or ‖B‖ penalties** and no double-counted L2.
  - Untouched effective rows get no local regularisation shrink. The globally shared B can affect every item, by design.
- **Initialisation:** A, B ~ N(0, s_r²) with s_r = (1e-4/r)^{1/4}, so Var(q_ij) = 1e-4, matching B1/B2's N(0, 0.01²).
  p_u is as in B1 for the same seed.
- **Balancing:** QR plus core-SVD balancing (‖A‖_F = ‖B‖_F, AB preserved to float tolerance), once at initialisation and
  after every server step. In B4 this is **DP post-processing** of the released A and B.
- Changing the factorisation and the optimiser path means **no claim of pure dimensionality causation**.

**B3 (no DP; capacity and communication vs frozen B1).**
- B1's convergence protocol: Poisson q 0.1, E 2, η_s 1, equal-user FedAvg ÷ realised |S_t|, validation every 10 rounds,
  patience 100 evaluations, max 8,000 rounds, balancing every round.
- Per-rank finite, validation-only local lr grid {1.25, 2.5, 5, 10, 20}, seed 42, with a **single edge rule**: one ×2 or ÷2
  extension if the argmax is at an edge. Argmax of best-round validation NDCG@10; ties go to the lr closer to 5 in log scale.
- Failures and non-finite trajectories are recorded explicitly as `diverged`.
- If an otherwise useful run hits the 8,000-round cap, this is **reported before any extra search** is invented.
- BLAS/OpenMP are forced to one thread per worker. Runtime and memory are profiled, and workers sized from the measurements.
- B3 is frozen (dated entry plus verified snapshot) **before** B3 test scoring. Final seeds: {42, 123, 2026, 7, 99}.

**B4 (low-rank user-level DP; matches final B2).**
- Poisson q 0.1, exactly T = 50, δ 1e-5, PRV primary / RDP cross-check, target ε {8, 4, 2, 1}, B2's T = 50 σ.
- ONE whole-client L2 clip of the concatenated [vec ΔA, vec ΔB] to C. Gaussian noise N(0, σ²C² I) on **all** D_r shared
  coordinates. Fixed qN = 94.2, then η_s, then balancing (post-processing). Empty rounds are noised and accounted.
- Local lr = B3's per-rank selection.
- Per-rank, validation-only selection at ε≈4, seed 42, final-round validation NDCG@10:
  - C grid = quantiles {p25, p50, p90} (2 significant figures) of pooled ‖[ΔA, ΔB]‖ from that rank's B3 mechanics (selected
    lr, seed 42, realised denominator, no DP, rounds 1–50). Argmax; ties to the smaller C.
  - Then η_s ∈ {0.5, 1, 2}. Argmax; ties to 1.
  - No expansion.
- A per-rank matched no-DP control at T 50 (same lr, η_s, qN; no clipping or noise).
- **Headline:** *optimised low-rank versus frozen full-rank at matched privacy*. This is **not** isolated coordinate-count
  causation.
- **Pre-declared fixed-settings diagnostic:** every rank at ε≈4 with local lr 5, C 1, η_s 1, T 50, on the same five seeds.
  This separates gains that need rank-specific tuning from those under the inherited B2 settings. Identical cells are reused.
- **Rank selection before any B4 test exposure:**
  1. Train the frozen per-rank final jobs in validation-only mode and retain the exact T = 50 checkpoints.
  2. Compute five-seed validation means.
  3. Document the validation-selected rank per ε, with a snapshot.
  4. Only then score those same checkpoints on test, once.
- B4 is frozen before test scoring.

**Supplemental replication.**
- B1 and B2 at seeds 7 and 99: exact frozen-protocol replication only (no retuning), written to new, separately tagged paths.
  Existing configs, results and checkpoints stay unchanged.
- The 3-seed cores remain as reported. The five-seed extension is reported separately.

**Statistics** (test NDCG@10; each user's score averaged across the five training seeds first; seed SD reported).
- 100,000 common paired user-bootstrap resamples, seed 2026, batched for memory. The bootstrap is **conditional on these training
  runs**.
- **B3:** B3(r) − B1, with Bonferroni-simultaneous 98.75% CIs across the 4 ranks.
- **B4 primary family:** 16 contrasts, B4(r, ε) − B2(ε), with simultaneous 99.6875% CIs plus ordinary 95% CIs.
  - Positive cells support improvement **only in those evaluated settings**.
  - A lack of positive evidence does not prove there is no benefit.
- **Secondary:** the validation-selected rank per ε versus B2; each rank's DP effect (versus its own control) and capacity
  effect; the fixed-settings diagnostic.
- **Popularity:** point estimates only.
- **Wording:** ε is never attributed to dimension. Noise diagnostics are "consistent with", never proof.

**Tests:** synthetic data only (the 30×40 fixture, patched `load_all`, evaluator spy, module guard). No real test calls in tests.

### 2026-10-02 — B3 implementation notes (before the B3 validation search)

- **New modules:** `src/lowrank.py` (factors, balancing, chained local update, one simulator for B3 and B4) and
  `src/train_lowrank.py` (the B3 convergence loop and the B4 fixed-T loop). They reuse B1's `bpr_grads` and B2's clip and
  noise functions unchanged. No B1/B2 file was edited.
- **Wording correction (supervisor):** no L2 term is applied directly to unseen effective rows, but their scores can still change
  locally and globally through the shared B.
- **Tests:** `tests/test_lowrank.py` has 15 tests, all on synthetic data. The full suite has **128 passing** tests.
- **Synthetic toy finding:** at lr 2.0 (the full-rank toy's value), the factorised model diverged by about round 20, because
  the effective step on q grows with the factor norms. lr 0.5–1.0 learned the rankings, so the toy test uses lr 1.0.
- **Profiling** (real split, 50 rounds, validation-free, `results/b3_profile.csv`):
  - 0.035–0.054 s per round and about 0.05 s per validation pass, balancing included;
  - peak RSS 250–290 MB per worker;
  - **at lr 5, rank 8 diverged at round 48**; ranks 4/16/32 did not within 50 rounds.

  Divergence is recorded as a search outcome.
- **Supervisor review details adopted:**
  - B3 rows and checkpoints store `best_round`, `rounds_run`, communication up to the best round, and total communication
    separately;
  - only early-stopped runs are eligible for the ordinary argmax (diverged and cap-hit runs are excluded);
  - any cap-hit run above the converged argmax is reported before any change to the search.
- **Search launch:** `.venv/bin/python experiments/run_b3.py --search --workers 16`. One BLAS/OpenMP thread per worker.
  Validation only.

### 2026-10-02 — B3 search attempt 1 crashed (robustness bug); identical search rerun

- **What happened.** The first `run_b3.py --search` run exited with code 1 before aggregating its results. In the rank-32 lr
  5 and lr 20 cells, A, B and P were finite, but the score matrix P(AB)ᵀ overflowed float32. The frozen evaluator correctly
  refused the non-finite scores with a `ValueError`, which the B3 loop did not treat as divergence. Of the 20 grid cells,
  18 had completed. Their per-run histories are archived unchanged in `results/raw/b3_search_attempt1_crashed/`, together
  with the crash log and a MANIFEST.sha256.
- **Fix.** Non-finite scores at evaluation are now recorded as `diverged`, in both low-rank loops, and a synthetic test was
  added. Total communication now uses `sim.comm_totals`, so it is correct even when a run diverges between evaluations.
- **Rerun.** The **identical** pre-declared search (same grid, same edge rule, seed 42, validation only), with no new values.
  Training is deterministic, so the 18 completed cells serve as a reproducibility check.

### 2026-10-02 — B3 search corrective rerun: outcome (validation only; NOT frozen; no test exposure)

- **Exit 0.** All 20 grid cells and 4 edge cells (lr 0.625, one ×½ expansion per rank) finished. The tables are
  `results/b3_lr_search.csv` and `results/b3_selection.csv`, and the histories and round logs are in `results/raw/b3_search/`.
- **Determinism.** The 18 cells completed in attempt 1 reproduce exactly: 18 of 18 histories (all columns except wall
  time) and 18 of 18 per-round logs are identical.
- **What failed and why.** 16 cells diverged:
  - 14 hit non-finite parameters inside a round;
  - r32 at lr 5 (round 60) and r32 at lr 20 (round 10) hit non-finite *scores* with finite factors (float32 overflow).
    These are the 2 cells that crashed attempt 1, and they are now correctly recorded as diverged.

  The divergence type is classified from the saved round logs, with no new runs.
- **Per-worker persistence.** This rerun did not write the per-worker `result_*.csv` files. The process had loaded
  `run_b3.py` before that code was added, so the aggregated CSV is this run's only result table. Every later run writes
  these files.
- **Selection** (early stopping only, seed 42):

  | Rank | Selected lr | Validation NDCG@10 | Best round |
  |---|---|---|---|
  | 4 | 1.25 | 0.0729 | 1430 |
  | 8 | 1.25 | 0.0852 | 3330 |
  | 16 | 1.25 | 0.0913 | 3920 |
  | 32 | **0.625** | 0.0859 | 5640 |

  For rank 32, the expanded-edge value beat lr 1.25 (0.0854) by 0.0005 on a single seed. It is **at the expanded edge**,
  and the single-edge rule allows no further expansion, so this is flagged for review. No run hit max_rounds.
- **Round-50 validation summary** (from existing histories, no new runs). At round 50 every selected lr is close to its
  initialisation:

  | Rank | Selected lr | Val NDCG@10 at round 50 | Best val NDCG@10 |
  |---|---|---|---|
  | 4 | 1.25 | 0.0040 | 0.0729 |
  | 8 | 1.25 | 0.0055 | 0.0852 |
  | 16 | 1.25 | 0.0028 | 0.0913 |
  | 32 | 0.625 | 0.0025 | 0.0859 |

  Several larger lrs reach 0.03–0.05 by round 50 before diverging later: r4 at lr 5 reaches 0.032 and diverges at 56;
  r16 at lr 5 reaches 0.047 and diverges at 59; r32 at lr 5 reaches 0.041 and diverges at 60. For comparison, B2's
  no-DP T = 50 validation reference is 0.0535. **Inheriting the B3 lr would therefore make the B4 T = 50 low-rank
  reference severely undertrained.** This is reported for protocol review; nothing has been changed.
- **Integrity.**
  - The split fingerprint matches the frozen value.
  - `u.data` SHA-256 `06416e59…a490` matches `src/data.py` and `data_summary.json`.
  - Every archive manifest verifies.
  - The crashed-attempt manifest mistakenly listed its own file, because `find` ran while the redirect was creating it.
    That single self-entry was removed; the other 37 entries are unchanged and verify.
  - The live `b2_accounting.csv` is a sorted superset of the freeze snapshot: all 9 snapshot rows are identical, and the
    3 added rows are the T = 50 σ for ε 8/2/1 solved in B2's final sweep (the values reported in the B2 summary).

## 2026-10-02 — B4 PROTOCOL AMENDMENT (before any B4 private or data run)

**Status of the original B4 tuning protocol: SUPERSEDED, retained as history.** The original protocol (see "B3/B4
protocol (low-rank shared item matrix)" above) used each rank's frozen B3 local lr, then a C grid of
p25/p50/p90 of that rank's B3 update norms over rounds 1–50, then η_s ∈ {0.5, 1, 2}. It is kept verbatim above and is
not used. No B4 private, search or norm-statistics run had been executed under it; `results/` has no b4_* data outputs.

**Reason (validation only).** The round-50 summary of the completed B3 search histories (validation only, seed 42, no
new runs) shows that every B3-selected lr is near initialisation at round 50:

| Rank | Selected lr | Val NDCG@10 at round 50 |
|---|---|---|
| 4 | 1.25 | 0.0040 |
| 8 | 1.25 | 0.0055 |
| 16 | 1.25 | 0.0028 |
| 32 | 0.625 | 0.0025 |

B2's no-DP T = 50 validation reference is 0.0535, while lr 5 reaches 0.03–0.05 by round 50. Inheriting B3's
long-horizon lr would therefore make the T = 50 low-rank reference near-random: a horizon/optimisation confound. A C grid
derived from that undertrained rate's tiny updates would carry the same confound.

**Amended B4 protocol (finite and complete; no new numerical candidates, no expansions).**

1. **Joint lr × C check, per rank.** Settings: ε ≈ 4 (B2's T = 50 σ, reused), η_s = 1.0, seed 42, T = 50, validation
   only.
   - lr ∈ the existing B3 grid {1.25, 2.5, 5, 10, 20}. Add the already-tested edge value 0.625 **only** for a rank
     whose frozen B3 selection is 0.625; at present that is rank 32, pending the B3 freeze.
   - C ∈ the existing B2 grid {1.0, 1.5, 2.4}.
   - This gives 15 cells per rank (18 for a rank with the 0.625 edge).
   - **Selection:** argmax of final-round (T = 50) validation NDCG@10 over cells with status ok. Ties go to the smaller
     C, then the lr nearer 5 in |log(lr/5)|, then (deterministic residual) the smaller lr.
   - Failed and diverged cells are recorded and reported, and never selected.
2. **η_s check, per rank.** Only η_s ∈ {0.5, 1, 2} at the selected (lr, C), same seed and ε. Argmax of final-round
   validation NDCG@10; ties go to η_s nearer 1, then the smaller.
3. **Everything else unchanged:**
   - q 0.1, T 50, δ 1e-5, PRV primary / RDP cross-check, targets 8/4/2/1, and B2's σ reused only after the exact-B2
     compatibility check;
   - one joint clip over [ΔA, ΔB], noise on all D_r coordinates, fixed qN, balancing;
   - the five-seed validation-only main grid with retained checkpoints, and five-seed validation rank selection per ε
     before any B4 test exposure;
   - every existing freeze, inventory and checkpoint-validation scoring gate;
   - the pre-declared fixed-settings ε ≈ 4 diagnostic (lr 5, C 1, η_s 1, T 50, all five seeds), with identical main
     cells reused rather than retrained.
4. **No** extra private horizons, learning rates, C values or η_s values. The B3 first-50-round update-norm distributions
   may be computed as **diagnostics only**; they do not define any grid.
5. **Framing.** Frozen B2 is unchanged. B4 is reported as a *validated low-rank model vs the frozen full-rank B2 at
   matched ε*. Low-rank gets a greater optimisation budget (a joint lr × C check that B2 did not have for lr), which will
   be disclosed. No isolated-dimensionality causal claim, and no claim that dimension reduces ε.

**B3 status.** Not frozen and not test-scored; it waits for the supervisor's table review. The joint lr × C check reads
the frozen B3 selection (for the 0.625 rule) and is refused until B3 is frozen.

### 2026-10-02 — Thread enforcement and manifest procedure corrections

- **Threads.** `os.environ.setdefault` does not override inherited BLAS/OpenMP variables. Measured under
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=8`, it gave numpy OpenBLAS 8 threads and torch 8. The B3/B4 runners now call
  `src.threads.force_env()`, which overrides the variables before numpy/torch are imported, and `enforce_and_report()`
  sets and reports the live pools. Measured under the same hostile environment: 1/1 in the parent and in each pool
  worker. Future B3/B4 result rows record the per-worker thread counts.
- **Completed runs.** The launch shell had no thread variables set, and reproducing that condition gives OpenBLAS 1 and
  torch 1. The completed B3 search cells therefore ran single-threaded and are scientifically unchanged. They are not
  rerun.
- **Manifests.** The B3 attempt-1 manifest had included its own entry. That line was removed in place, so **the
  superseded manifest version was not retained**: its self-hash line cannot be reconstructed, though its 37 data
  entries are unchanged and verify. From now on, manifests are written outside the target directory, exclude all
  `MANIFEST*` files, and are then moved in. Superseded versions are kept as `MANIFEST.sha256.superseded_<timestamp>`,
  and the correction is recorded in this log.

## 2026-10-02 18:33:55 +05:30 (13:03:55Z) — Freeze B3 (supervisor-approved selection; BEFORE any B3 test exposure)

- **Frozen** in `configs/b3.yaml` (`protocol_frozen: true`): per-rank local lr {4: 1.25, 8: 1.25, 16: 1.25, 32: 0.625}.
  All other settings are B1's convergence protocol (eval_every 10, patience 100 evaluations, max_rounds 8000), with
  seeds [42, 123, 2026, 7, 99].
- **Caveat retained:** rank 32's lr 0.625 is the expanded edge. It beat 1.25 by +0.0005 validation NDCG@10 on a single
  seed (42). There will be no further expansion.
- **Retained:**
  - all 16 divergence outcomes (`results/b3_lr_search.csv`);
  - the crashed attempt-1 archive (`results/raw/b3_search_attempt1_crashed/`).
- **Snapshot:** `results/raw/b3_freeze_snapshot_20261002T130355Z/`. It holds:
  - the configs (b3, b1, dataset);
  - the low-rank code and its loops;
  - the frozen evaluator, data and federated modules;
  - `run_b3.py`, the threads and manifest helpers;
  - both B3 selection tables.

  The manifest has 14 entries with relative paths. It was written outside the directory, excludes MANIFEST*, and
  verifies with `sha256sum -c`.
- **Next steps, as authorised:**
  1. `--final-train`: five seeds × four ranks, validation only, retaining best-validation checkpoints.
  2. Only if all 20 jobs pass the exact inventory/convergence gate: `--final-score`, once per checkpoint, then the frozen
     B1 supplemental seeds 7/99 on separate paths.
  3. Any cap hit or failure is reported before scoring.
- **CPU budget:** at most 20 one-thread workers in total (12 B3 + 6 B4 + 2 B1).

### 2026-10-02T18:37:25+05:30 — Amended B4 step 1 launched (validation only, B4 protocol_frozen: false)

- Pre-run checks:
  - the B4 runner tests pass, including the grid-exactness, tie-rule, B3-freeze gate and η_s = 1 reuse tests;
  - the amended-protocol entry above was written before this run;
  - before this run, `results/` had no b4_* files.
- The grid resolved from frozen B3: ranks 4, 8 and 16 get 15 cells each, and rank 32 gets 18 (its frozen selection is
  0.625). That is **63 cells** at ε ≈ 4, η_s 1, seed 42, T 50.
- Command: `run_b4.py --lr-c-search --workers 6`, running concurrently with the B3 final training (12 workers), so 18 of
  the 20 one-thread worker slots are in use.
- The full-suite rerun is deferred until the production jobs are quiet. Its output-guard fixture snapshots the real
  `results/`, which these jobs are writing to.

### 2026-10-02T18:38:44+05:30 — Amended B4 step 1 result (validation, ε ≈ 4, η_s 1, seed 42, T 50); step 2 launched

- **Run:** exit 0, 63 cells, numpy OpenBLAS and torch at 1 thread in every worker.
- **Failures:** 17 cells diverged; all are recorded and none can be selected.
  - All lr 20 cells diverged.
  - lr 10 diverged at C 1.5/2.4 for ranks 4 and 8, and at C 2.4 for rank 32.
- **Selection** (predeclared rule):

  | Rank | lr | C | Val NDCG@10 |
  |---|---|---|---|
  | 4 | 10 | 1.0 | 0.0429 |
  | 8 | 10 | 1.0 | 0.0465 |
  | 16 | 5 | 1.0 | 0.0484 |
  | 32 | 5 | 1.0 | 0.0487 |

  B2's ε ≈ 4 validation is 0.0396.
- **Flags for review (no action; the protocol forbids expansion):**
  - Every rank selected C = 1.0, the lowest value of B2's existing grid.
  - Ranks 4 and 8 selected lr 10, which neighbours diverged cells, so it is a five-seed stability risk.
  - Rank 32's 0.625 cells stay near-initial at T 50 (≤ 0.0046).
- `per_rank` lr/C were written to `configs/b4.yaml` from `results/b4_lr_c_selection.csv`; the runner checks they match.
- **Step 2:** η_s ∈ {0.5, 2} at the selected (lr, C), 8 new cells. The η_s = 1 cell is the identical step-1 cell and is
  reused. B4 `protocol_frozen` stays false.

### 2026-10-02T18:47:06+05:30 — B3 final training (frozen, validation only): GATE FAILED → no B3 test scoring, no B1 supplemental

- **Run:** `--final-train --workers 12`, exit 0, 20 jobs. Every rank's seed-42 run reproduces its search cell exactly
  (best rounds 1430 / 3330 / 3920 / 5640).
- **Formal gate (`check_b3_inventory`): FAIL.**

  | Rank | Seed | Outcome |
  |---|---|---|
  | 4 | 2026 | Diverged at round 210 (best 150) |
  | 4 | 7 | Diverged at round 156 (best 140) |
  | 32 | 123 | Hit the 8000-round cap (best 7910, still improving; val 0.0939) |

- **Consequence:** as authorised, nothing was test-scored and the B1 supplemental was not run. No B3
  `per_user_test` or `test_result` file exists. The checkpoints and validation records are retained.
- **Interpretation caveat (not a gate item):** rank 32 early-stopped on seeds 2026 and 7 at best rounds 620 and 800, with
  val 0.058/0.059 against 0.086–0.094 on its other seeds. This is consistent with patience (1000 rounds) triggering on a
  long plateau at the slow lr 0.625.
- **Validation means over non-diverged runs** (seed counts differ by rank):

  | Rank | Mean val NDCG@10 | Runs counted |
  |---|---|---|
  | 4 | 0.0732 | 3 |
  | 8 | 0.0825 | 5 |
  | 16 | 0.0873 | 5 |
  | 32 | 0.0781 | 5, including the cap hit |

- **Communication through the selected checkpoint** (means over non-diverged runs) vs B1's 861,184 B/client/round and
  81.9 GB mean (84.1 / 82.8 / 78.8 GB):

  | Rank | Payload vs B1 | Volume to best round | Volume vs B1 |
  |---|---|---|---|
  | 4 | 0.065× | 9.8 GB | 0.12× |
  | 8 | 0.13× | 32.3 GB | 0.39× |
  | 16 | 0.26× | 77.4 GB | 0.95× |
  | 32 | 0.52× | 176.6 GB (range 26–334) | **2.16×** |

  **A smaller per-round payload does not imply a smaller total volume.**
- **Tests:** the full suite passes, 180 tests, run with the production jobs idle.
- **Awaiting the supervisor's decision** on the B3 gate failure. No retry, protocol change or scoring was done.

### 2026-10-02T18:51:04+05:30 — B3: initial freeze SUPERSEDED (before any test); bounded correction 1

- **Superseded:** the initial B3 freeze (snapshot `b3_freeze_snapshot_20261002T130355Z`). It failed the 20-job gate:
  rank 4 diverged on seeds 2026 and 7, and rank 32 on seed 123 hit the cap. No test exposure occurred.
- **One bounded correction, supervisor-directed:**
  - rank 4 → the already-tested lr 0.625;
  - rank 32 → the already-tested lr 1.25;
  - ranks 8 and 16 unchanged, reusing their verified checkpoints.

  All five seeds are run. There are no new candidates, no budget or plateau change, and no further tuning: if either
  alternative fails convergence, the bottleneck is reported.
- **Artifacts:** new tags (`r4_lr0.625_*`, `r32_lr1.25_*`). The original runs and checkpoints are kept, and the
  initial `b3_final_validation.csv` is preserved in the snapshot.
- **Snapshot:** `results/raw/b3_correction1_snapshot_20261002T132103Z` (configs, code, tables; manifest verified).
- **Code:** the resume/score cache now compares only per-job scientific fields: dataset, dim/init/rank, q, optimizer, lr,
  E, L2, server lr, aggregation, stopping and evaluation. Another rank's `selected_lr` no longer affects reuse. This is
  covered by a focused test.
- **B4:** the 63-cell lr × C grid is pinned to the INITIAL B3 snapshot (`b3_reference_config`, recomputed as 63 cells),
  so this correction does not alter it retroactively.
- **Test scoring stays blocked** until all 20 jobs pass the gate.
- **B4 stability screen launched alongside:** per_rank lr {4: 10, 8: 10, 16: 5, 32: 5}, C 1, η_s {4: 0.5, 8: 0.5,
  16: 1, 32: 1}. Five seeds × {noDP, ε 8/4/2/1}, validation only, kind `stability`, with T 50 checkpoints retained.
  Not frozen and not test-scored. Workers: 10 for B3 and 10 for B4.

### 2026-10-02T18:56:16+05:30 — B3 correction 1 and B4 stability screen results (validation only; nothing test-scored)

- **B3 correction 1:** B3 correction1 GATE: FAIL not converged (report before scoring): [[32, 123, 'diverged']]. Rank 32 at lr 1.25 diverged on seed
  123 (round 226). Rank 4 at lr 0.625 converged on all seeds, and ranks 8 and 16 were reused. Per the instruction there is
  no further tuning; this is reported as a bottleneck. Test scoring stays blocked.
- **B4 stability screen:** 100 jobs, `results/b4_stability_screen.csv`. The failures by rank are (counts out of 5
  seeds):

  | Rank | Failed runs |
  |---|---|
  | 4 | no-DP 5, ε 8 1, ε 1 4 |
  | 8 | no-DP 5 |
  | 16 | no-DP 2, ε 1 1 |
  | 32 | no-DP 2 |

  Every no-DP control is unstable on at least 2 of 5 seeds: the settings were selected under clipping at ε ≈ 4 and are
  run without clipping. Not frozen. Failed jobs get no utility value, and no partial-seed averages are taken.

### 2026-10-02T19:01:49+05:30 — Pre-run declarations: B3 revision 2 and B4 bounded fallback (supervisor decisions)

**B3 revision 2** (before running):
- **Final lr:** {4: 0.625, 8: 1.25, 16: 1.25, 32: 0.625}. Rank 32 at lr 1.25 is shown to be unstable (seed 123 diverged),
  while the original lr 0.625 was stable apart from seed 123 hitting the cap (best 7910 of 8000).
- **ONE final convergence diagnostic:** rank 32, lr 0.625, seed 123, max_rounds **16000**, with patience 100 and
  eval_every 10 unchanged. It uses new tagged paths (`r32_lr0.625_seed123_max16000`), so the capped original is not
  overwritten.
- **Reused** (verified scientific match, early-stopped below 8000): rank 32 seeds 42/2026/7/99 at 0.625 (original), rank 4
  at 0.625 on all 5 seeds (correction 1), ranks 8 and 16 at 1.25 (original). Raising only this job's ceiling cannot
  affect those paths. Each record carries its `max_rounds_ceiling`.
- **If the diagnostic still caps or diverges, B3 stops at that bottleneck.** Test scoring stays blocked until the
  20-job gate passes.

**B4:** the current winners are rejected on multi-seed stability; nothing was tested. **ONE bounded fallback**, with no
expansions and no second fallback:
- **Grid:** C fixed at 1.0; local_lr ∈ {2.5, 5}; server_lr ∈ {0.5, 1}. That is 4 combinations per rank.
- **Runs:** each combination on 5 seeds × {noDP, ε 8/4/2/1}, validation only, T 50 checkpoints retained, kind
  `fallback`. Identical completed stability jobs (ranks 16/32 at lr 5, η 1) are reused.
- **Eligibility:** all 25 jobs are finite and reach exactly T 50.
- **Selection:** the eligible combination with the greatest five-seed ε 4 validation mean per rank. Ties go to the
  server lr nearer 1, then the lr nearer 5. A rank with no eligible combination is reported as a bottleneck.
- **Retained:** the exact-settings diagnostic and the larger-tuning-budget disclosure.
- **Workers:** B3 2 + B4 18.
- 2026-10-02T19:06:01+05:30 B4 fallback attempt 1 crashed at 311/350 new jobs: reuse refused a diverged stability job (result, no checkpoint by design). Fix: recorded non-ok result without checkpoint = verified failure. Relaunched; completed jobs reused. Log archived as b4_fallback_log_attempt1_crashed.txt.
- 2026-10-02T19:07:14+05:30 B4 fallback attempt 2 refused a reused checkpoint on a last-ulp epsilon difference (3.963885356647058 vs ...0567). Fix: float fields compared with rel_tol 1e-12 (other fields exact; malformed-checkpoint tests pass, incl. epsilon x0.5). Relaunched.

### 2026-10-02T19:08:31+05:30 — B4 bounded fallback result (validation only; not frozen; no test)

- 400 jobs (50 reused), exit 0, 19 failures. Eligible (all 25 finite/exact T50) & selected by 5-seed eps4 val mean: r4 lr2.5/slr1 0.0340; r8 lr5/slr0.5 0.0420; r16 lr5/slr0.5 0.0361; r32 lr5/slr0.5 0.0421. Every rank has >=1 eligible combo. lr5/slr1 ineligible at every rank (noDP divergences). Full table: results/b4_fallback_screen.csv; codex_stability.

## 2026-10-02T19:11:33+05:30 (20261002T134133Z) — B3 FINAL FREEZE (revision 2); gate PASSED; test scoring authorised

- **Gate:** 20/20 early stopping. The rank 32 seed 123 diagnostic (lr 0.625, ceiling 16000) reached best round 8350 of
  9350, val 0.0958. All other records were reused under verified scientific matches with the 8000 ceiling.
- **Frozen lr:** {4: 0.625, 8: 1.25, 16: 1.25, 32: 0.625}, with `max_rounds_override` {32_123: 16000}.
- **Caveats retained:**
  - rank 32 is the expanded edge, selected on a single seed;
  - rank 32 seed 123 needed the larger ceiling;
  - rank 32 early-stopped on seeds 2026 and 7 at best rounds 620 and 800 (val 0.058/0.059), a plausible plateau stop;
  - rank 4 was corrected from 1.25 to 0.625;
  - ranks are compared at different learning rates and stopping budgets.
- **Snapshot:** `results/raw/b3_rev2_freeze_snapshot_20261002T134133Z` (manifest verified).
- **Next:** `--final-score`, once per retained checkpoint, then the frozen B1 supplemental seeds 7/99 on separate paths.

## 2026-10-02T19:14:35+05:30 (20261002T134435Z) — B4 FREEZE (bounded fallback winners; BEFORE any B4 test)

- **Frozen settings:** C 1, T 50, q 0.1, δ 1e-5, B2's σ.

  | Rank | local lr | η_s |
  |---|---|---|
  | 4 | 2.5 | 1 |
  | 8 | 5 | 0.5 |
  | 16 | 5 | 0.5 |
  | 32 | 5 | 0.5 |

  All rejected stages are preserved: the superseded original inheritance protocol, the lr × C and η_s winners rejected on
  multi-seed stability, and the stability screen.
- **Inventory:** `results/b4_final_validation.csv` is built from the RETAINED fallback/stability checkpoints, with no
  retraining. It has 100 main rows (4 ranks × 5 levels × 5 seeds, all ok) and 20 fixed-settings diagnostic rows (lr 5,
  C 1, η_s 1, ε 4, all ok), marked with `role`. Checks passed:
  - every checkpoint's SHA-256 matches its record;
  - `validate_checkpoint` passes on all 120 (shapes, dtype/finiteness, accounting, scientific config);
  - `check_main_inventory` passes.
- **Five-seed validation means:**

  | Rank | noDP | ε 8 | ε 4 | ε 2 | ε 1 |
  |---|---|---|---|---|---|
  | 4 | 0.0358 | 0.0344 | 0.0340 | 0.0254 | 0.0142 |
  | 8 | 0.0438 | 0.0433 | 0.0420 | 0.0313 | 0.0191 |
  | 16 | 0.0434 | 0.0416 | 0.0361 | 0.0353 | 0.0212 |
  | 32 | 0.0468 | 0.0428 | 0.0421 | 0.0330 | 0.0163 |

- **Validation-selected rank** (`results/b4_rank_selection.csv`, written before test): noDP 32, ε 8 rank 8, ε 4 rank 32,
  ε 2 rank 16, ε 1 rank 16.
- **Disclosure:** low-rank received a larger tuning budget than frozen B2, and prior test exposure exists. This is a
  validated low-rank vs frozen full-rank comparison, with no isolated-dimensionality claim.
- **Snapshot:** `results/raw/b4_freeze_snapshot_20261002T134435Z` (manifest verified).
- 2026-10-02T19:16:36+05:30 codex_final_gate loaded; full suite 181 passed; launching B4 --score-test (8 workers, once per retained checkpoint) and B2 supplemental 7/99 (results/raw/b2_supplemental_orchestration.py, 12 workers, separate roots).

## 2026-10-02 — FINAL B3/B4 test analysis (predeclared; no tuning on test)

- **Analysis:** five seeds; each user's NDCG@10 is averaged over seeds; paired user bootstrap with 100,000 common
  resamples, seed 2026, in batches. The script is `experiments/analyse_b3b4.py`, and it writes
  `results/b34_*.csv`.
- **B3 − B1** (98.75% simultaneous CIs): r4 −0.0295, r8 −0.0180, r16 −0.0127, r32 −0.0209. All are below B1.
- **B4 − B2** (99.6875% simultaneous CIs):
  - **ε 8:** all ranks are below B2.
  - **ε 4:** no rank differs.
  - **ε 2:** r8, r16 and r32 are above B2 (+0.009 to +0.011); r4 includes 0.
  - **ε 1:** only r8 excludes 0 (+0.0067).
- **Secondary results:** see `results/b34_contrasts.csv` and docs/B4_RESULTS.md.
  - The no-DP capacity controls are all below B2's no-DP control.
  - The exact-settings ε 4 diagnostic: every rank's 95% CI against B2 includes 0.
- **Final audit:**
  - The 23 raw files match the 2026-09-28 session-start snapshot.
  - Of the 877 post-B2 non-doc files, 876 are identical. The one change is the append-only
    `b2_final_command_log.txt`, last written at 17:34 during the B2 doc phase.
  - All 100 B4 main and 20 diagnostic checkpoints match the frozen per_rank and fixed-diagnostic hyperparameters.
  - Inventories are complete: B1 5/5, B2 25/25.
  - Freeze before first test: B3 19:11:33 → 19:11:35; B4 19:14:35 → 19:16:38 (+05:30).
- **Docs:** docs/B3_RESULTS.md, docs/B4_RESULTS.md, a README section, and a CODEX_HANDOFF update (the previous copy is
  archived).
- **Supervisor review of the results:**
  - **Primary adjusted positives:** only r8/r16/r32 at ε 2, and r8 at ε 1.
  - **Negatives:** every B3 rank is worse than B1; every B4 rank is worse than B2 at ε 8; there is no reliable gain at
    ε 4.
  - **Validation-selected r16 at ε 1:** positive at 95%, but the adjusted CI includes 0. It is not a primary discovery,
    and there is no test-based reselection.
  - **H1:** limited support, with no monotonic trend.
  - **Caveats added to the docs:**
    - raw factor noise is not effective-Q noise (r4 at ε 1 has a larger final Q norm than B2 at ε 1, despite less
      factor noise);
    - there are confounds (optimiser, lr, η_s, balancing, tuning);
    - B4 volume is compared with the matched B2 (0.065/0.130/0.260/0.519×), while B3 r32 needs 1.92× B1's volume.
- 2026-10-02T19:25:18+05:30 **Final review fixes (no training, no evaluation):**
  - **B3 doc:** rank 32 seeds 2026/7 have BEST rounds 620/800 and stopped at 1620/1800. The communication table now
    shows payload bytes, GB to best and GB for the whole run.
  - **B4 doc:**
    - ε 4 reads "no reliable difference detected", and all low-rank seed SDs are shown;
    - the DP scope is stated: frozen hyperparameters and assumed secure aggregation; selection, evaluation and the
      research process are outside the guarantee;
    - balancing preserves AB;
    - the effective noise in Q includes Z_A B + A Z_B + Z_A Z_B, and the final Q norm is not a direct measure of it;
    - a popularity reference was added (0.0443; every model at ε ≤ 2 is below it).
  - **Analysis:** `analyse_b3b4.py` input checks pass on 170 files: 942 unique user ids, identical sorted ids across
    all models and seeds, all values finite. HR@10 and MRR@10 five-seed means and SDs were added. The NDCG contrasts,
    perturbation and communication CSVs are byte-identical to v1, which is archived in
    `results/raw/archive_b34_analysis_v1_20261002T135329Z`.
  - **Figures:** `results/plots/b4_utility_vs_epsilon.{png,svg}` and `results/plots/b3_utility_vs_total_comm.{png,svg}`,
    generated by `experiments/plot_b3b4.py`.
  - **Handoff:** the TL;DR was replaced with the final state, and the stale in-progress instructions and resolved open
    items were removed. The audit wording now reads 876 of 877 identical plus one append-only log, and the test wording
    was corrected.
- 2026-10-02T19:26:47+05:30 **COMPLETION:** the popularity wording was corrected to a point reference: every five-seed DP mean at ε ≤ 2 is below it, B4 r32 at ε 4 is numerically close (0.0445 vs 0.0443), and no significance claim is made. The doc figure links were verified. **B3 and B4 are finalized, and no further runs will be made.** Nothing is committed or pushed.

## 2026-10-02 — Published to GitHub

- **Commit:** `f9c0a2b257f0ba1669719fce5dd969d5a50d1c2b`, "Finalize B2 protocol and five-seed B3/B4 research results",
  https://github.com/Shuyanokoji7/CS-670/commit/f9c0a2b257f0ba1669719fce5dd969d5a50d1c2b. It was a fast-forward push to
  `main` with no force.
- **Scope:** `fedrec-dp/` only, 2946 files (97.9 MB), respecting `.gitignore`. Every staged blob equals its working
  file, and nothing outside the project was touched.
- **Audit references:** `results/raw/audit_final_2026-10-02/` holds byte-identical copies, a README and a manifest.
- **Validation:** 181 tests passed.
- **Not published:** `checkpoints/`, `data/raw/` and `data/processed/` are ignored and retained locally only.

## 2026-10-02 — New-agent handoff and execution-mode update

- The user requested a continuation handoff and explicitly changed execution mode:
  the next agent works independently; Claude is no longer the executor.
- Added `handoff.md` with final frozen B0–B4 state, five-seed results, audit history,
  publishing state, stale-documentation notes, and proposed paper directions.
- Proposed follow-up studies remain proposals, not a declared experiment protocol.
- Documentation-only work: rechecked all 23 raw-file hashes, the frozen split
  fingerprint/counts, the initial clean git state at `81680fd`, and collection of
  181 tests. No training, model test scoring or frozen-result changes were made.
- This handoff was created locally after the published commits; it was not pushed.

## 2026-10-02 — E1 effective-noise direction authorized; bounded protocol declared before implementation

- The user selected direction 2 and authorized independent implementation,
  experiments and documentation without further permission gates. This supersedes
  older instructions to wait for a supervisor or to use Claude. No sub-agents used.
- Read the handoffs, complete research log, current configs and B2–B4 reports/results.
  Verified 181 tests (26 warnings), all 23 raw hashes and the frozen split fingerprint.
  Initial git state has only the existing handoff and its append-only log update.
- Recorded SHA-256 hashes for 3932 existing code/config/test/result/checkpoint/data
  artifacts in `results/extensions/effective_noise_v1/audit/protected_initial.json`.
  Existing artifacts are protected; no B0–B4 reruns or evaluator changes.
- Declared `docs/extensions/EFFECTIVE_NOISE_PROTOCOL.md` and
  `configs/extensions/effective_noise_v1.yaml` before implementation/experiments.
  E1-Full, E1-Two and E1-FixedB use a bounded equal four-candidate budget per model
  unit, three selection seeds, all six privacy/control levels for eligibility,
  five final seeds, T50, q.1, C1, exact B2 sigma checks, and no grid expansion.
- Added the predeclared one-release user-level DP popularity baseline with a
  fixed sqrt(20) L2 contribution bound and analytic Gaussian calibration.
- Test scoring is blocked until five-seed inventory, rank selection and a
  dated immutable snapshot freeze. Known ML-100K test exposure remains disclosed.
- Verified primary prior-art pages for FFA-LoRA, FedASK, public item features,
  personalized joint DP and the recent linear parameterization paper. Fixed-factor
  learning is a baseline; proposed novelty is the recommendation-specific analysis.

## 2026-10-02 — E1 implementation and verification before real-data search

- Added separate E1 simulator, staged runner, analytic-Gaussian DP popularity, conditional moment derivations, synthetic gauge/Monte Carlo script and analysis script. Frozen source modules are unchanged.
- Full suite: 203 passed, 26 existing warnings. New mechanism tests use synthetic fixtures only. Tests verify gradients, rank-r score equivalence, decomposition/moments, user sensitivity, empty rounds, local-state locality, full-rank arithmetic identity, cache integrity and freeze gating.
- Noise diagnostics compare the noisy step with its own same-state/noiseless-signal counterfactual; public random margin probes use no held-out labels. Float32 arithmetic/balancing residual is reported separately.
- Commands executed: `.venv/bin/python experiments/run_effective_noise.py accounting`, `.venv/bin/python experiments/effective_noise_geometry.py`, and `.venv/bin/python experiments/run_effective_noise.py profile` (prefixed with PYTHONDONTWRITEBYTECODE=1). All are validation/synthetic-only. Search/final/test remain pending.

### E1 real-data profile and search launch (validation only)

- Profile at lr5/slr.5/eps4/seed42: Full 3.67 s, Two r16 2.01 s,
  FixedB r16 1.85 s; all finite. Validation numbers are retained as three declared
  search cells, not a separate selection exercise. Twelve one-thread workers used.
- Exact stored B2 sigmas independently recomputed to PRV/RDP guarantees.
  One-release popularity sigma at epsilon 8/4/2/1 is
  .600229/1.081162/1.993812/3.730632 (analytic Gaussian; separate q1/T1 record).
- Synthetic Monte Carlo uses 2000 draws per method/noise scale; empirical/theory
  matrix and score energy ratios range about .997–1.003, supporting implementation.
- Launched the declared 648-job search, validation only, no expansions:
  `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/run_effective_noise.py search --workers 12`.
  Full output is retained in the E1 `search_command.log`. Failures remain in the table.
- Before extension test scoring, declared an additional **secondary** historical
  comparison of FixedB against frozen B4 from saved per-user CSVs, with initialization
  and historical tuning confounds disclosed. No baseline is rescored or retrained.
- Recommendation geometry note: fixing B saves only 3.67% of coordinates versus
  Two at equal rank because A is much larger than B. The main payload reduction
  versus full rank is shared by both low-rank methods.

### E1 search outcome and five-seed confirmation launch (validation only)

- Search completed all 648 declared jobs; 11 failed trajectories retained. Each of nine model units has an eligible candidate.
- Full and FixedB (every rank) select local lr5/server lr1. Two ranks4/8/16 select lr5/slr.5, rank32 selects lr5/slr1. All use C1, T50. No grid expansion; every FixedB rate is at the upper declared edge, a limitation to report.
- FixedB three-seed eps4 validation is lower than Two at every rank. This is not grounds to expand its grid or change the selection rule.
- Launched `.venv/bin/python experiments/run_effective_noise.py final-train --workers 12` with PYTHONDONTWRITEBYTECODE=1; selected three-seed jobs reused only after hash/config/code/split checks. Test remains blocked pending complete five-seed stability.

## 20261002T153741Z — E1 FINAL FREEZE before extension test scoring

- Complete finite five-seed inventory: 270 training checkpoints and 21 popularity score arrays.
- Equal-budget selections and validation rank selection retained in the E1 output directory.
- No extension test scoring has occurred; known historical ML-100K test exposure remains.
- Immutable snapshot: `results/extensions/effective_noise_v1/snapshots/freeze_20261002T153741Z`; manifest verified.
- Existing 3932 protected artifacts verified unchanged. Test scoring now authorized by the declared protocol.

### E1 final scoring and diagnostic correction (no retraining or rescore)

- Five-seed inventory passed: 270/270 training states finite at exactly T50; all 21 popularity score arrays retained. Protocol/config/checkpoint freeze at 20261002T153741Z preceded first extension test output at 2026-10-02T15:38:44.686919Z. Additional analysis/theory/tests snapshot manifest created at 15:38:42.296203Z, before first test output.
- Scored each of 291 retained models once, with the frozen evaluator. No historical checkpoint was rescored. All 3932 protected artifacts remain unchanged.
- Numerical stability caveat: finiteness is only the declared eligibility criterion, not proof of well-conditioned training. Two r32/eps1 contains very large finite local P on several seeds, and Two r4/eps1 has score-shock outliers. They remain in all results; no post-test exclusions.
- Found an E1 **diagnostic-only** bug: the frozen runner computes mean user norm in float32, so it reports infinity on finite large P for r32/eps1/seed123. The conditional expected-energy diagnostics already use float64 and are unaffected; checkpoints/ranking scores are finite. Raw logs and frozen code are preserved. The live analysis now recomputes final mean/median/max user norms in float64 from retained checkpoints and reports the overflow count. No training/settings/test output changed. The original analysis remains in the pre-test snapshot.
- Selected-setting eps1 expected bilinear energy shares are about .0021–.0030. This is not evidence that the bilinear term dominates the observed failures; linear factor amplification and local-state dynamics are central candidates.

### E1 final findings and documentation

- Primary family: zero positive, eight negative, eight zero-containing Bonferroni intervals (16 contrasts). Every rank is worse at eps2; r8/r16 lose at eps4, r4/r8 lose at eps1. No test-based reselection.
- User-level DP popularity eps1 test NDCG .048389 (seed SD .001125), bounded no-noise .049885. This exceeds every collaborative eps1 mean; every collaborative mean at eps<=4 is below the DP-popularity point estimate. Secondary intervals are explicitly exploratory.
- Effective noise: eps1 expected bilinear shares .207%/.232%/.251%/.301% across Two ranks; much larger effects arise from linear factor scales and private local P. FixedB score-equivalence to rank-r BPR and only 3.67% incremental payload savings limit the novelty narrative.
- Added results report, theory, contribution audit, artifact index, plots and E1 continuation handoff. No second dataset, metadata model or communication-budget experiment is claimed as completed.
- Final-test harness correction: live pytest stdout was initially redirected inside the guarded results tree, causing four artifact-guard teardown errors although all 203 assertions passed. That log is retained. Rerun captures stdout in /tmp and copies it only after completion; no mechanism or experiment changed.

## 2026-10-02 — E1 COMPLETE: final audit passed

- Final full suite: 203 passed, 26 warnings (exit 0). The first redirected-output guard errors are archived; the successful log is `results/extensions/effective_noise_v1/final_pytest.log`.
- All 3932 protected baseline files and all 23 raw hashes verify; frozen split fingerprint unchanged. Reloaded/checksummed all 270 final training states, with no test rescoring.
- All 291 saved per-user test files have the same 942 unique users; every NDCG/HR/MRR value and summary recomputed directly from saved ranks agrees. No model was evaluated again.
- Every retained training record reports one numpy OpenBLAS and one torch thread. All frozen snapshot manifests and local documentation links verify.
- Final machine-readable audit: `results/extensions/effective_noise_v1/audit/final_audit.json`. E1 root manifest has 2148 entries, excluding MANIFEST files themselves, and verifies.
- E1 is finalized locally. Original B0–B4 remain frozen; no commit or push made. Follow-up priorities are documented in `docs/extensions/E1_HANDOFF.md`, with independent replication and controlled update-scale/communication studies still pending.

## 2026-10-03 — flexible research continuation, E2/E2b and curated UGP materials

- User objective: pursue a defensible paper contribution independently; the
  research direction is explicitly not fixed. Create an organized sibling
  `CS-670/` report base with selected evidence, not the entire iteration archive.
- Expanded primary-literature review. PRISM, FLoRG, FedRot-LoRA, FedGSA,
  NoiseCurve, GeoDP, DiSK and established private recommendation/personalization
  constrain novelty. Record source links and actual reading depth in
  `../CS-670/references/PRIOR_ART_AUDIT.md`; 25 core citations and issued search
  families are saved. No general fixed-factor, alignment or noise-dynamics
  novelty is claimed. Exact overlap for the narrower diagnostic is still open.
- Declared `NOISE_MEMORY_PROTOCOL.md` before outputs. Added the separate
  `experiments/probe_noise_memory.py`, reusing frozen E1 round updates and
  thirty hash-checked final checkpoints. Full/FixedB-r8/Two-r8, epsilon 1/2,
  five checkpoint seeds, two pulse draws and amplitudes .25/1 yield 120 probes.
  All completed, no exclusions, 1680 observations. No validation/test targets
  scored, no historical model retrained or rescored. Additional rounds are
  diagnostic counterfactuals, outside the starting model's DP release claim.
- E2's large delayed Two-factor distances motivated a disclosed adaptive
  follow-up. Declared `NOISE_MEMORY_ALIGNMENT_PROTOCOL.md` before E2b outputs;
  `probe_noise_memory_aligned.py` repeated all forty Two-r8 probes with
  orthogonal Procrustes alignment to each reference pre-round state. All
  completed, 560 observations. This changes the cross-branch noise coupling,
  not the marginal private mechanism in exact arithmetic. Maximum real-probe
  relative product-preservation error is 3.913264641762453e-8.
- At alpha1, lag10 after shared reset, raw Two-r8 mean top-10 disagreement is
  17.8546% / 9.3960% at starting epsilon1 / epsilon2, versus .4204% / .1083%
  under aligned coupling. Full/FixedB-r8 at epsilon1 retain .5212% / .3747%.
  Each summary averages two draws within checkpoint seed before five-seed
  aggregation. Seed dispersion, finite score-scale outliers, all amplitudes
  and branches remain retained. These are label-free ranking differences,
  not NDCG loss, independent samples, or a causal percentage decomposition.
- Six E2 and two E2b synthetic check cases passed: zero-pulse equality,
  both-reset equality through ten rounds, RNG matching, exact score-decomposition
  closure and product-preserving alignment. Real probes additionally check
  reset equality, RNG streams, input hashes, finiteness and decomposition.
- Added a separate post-pilot synthetic null example in the report bundle:
  equivalent (A,B) and (-A,-B) have positive raw-common-noise distance, zero
  consistently transformed distance. Ten thousand float64 draws give expected
  / observed matrix squared distance 6.345313 / 6.341142 (MC SE .012443).
  This verifies elementary algebra; it is not a new theorem or training method.
- Created `CS-670/`: integrated UGP draft, 45 explicitly whitelisted source
  copies, six figures in PNG/PDF/SVG, methods, material iteration history,
  results guide, pilot theory/results, bibliography, prior-art/search audit,
  research decisions and provenance. Bulk checkpoints, raw data, superseded
  sweeps and process logs remain in the archive. No unfavorable result removed.
- Saved-evidence verification passed: all 3932 protected historical files,
  E1's 2148-entry root manifest and both snapshots, all thirty input checkpoint
  hashes, E2/E2b source freezes, all 2240 original observation rows aggregated
  back to their seed summaries, copied-file hashes, primary contrast counts,
  validation-selected plotting table and local document links. Audit script:
  `../CS-670/reproducibility/verify_bundle.py --archive --write-audit`.
  This is separate from E1's historical 203-test suite result; no new full
  baseline suite or held-out model scoring was run in this continuation.
- New findings support a careful measurement audit and modest local-state
  persistence, not an improved recommender or confirmed paper novelty.
  Independent data, alternative couplings/optimizers, actionable prediction,
  published private ALS and equal-total-communication comparisons remain
  outstanding. Direction can change if the candidate fails these gates.
- Updated `handoff.md` to point to the latest continuation handoff and report
  folder. No commit, push, external message or publication made.
- Final new artifact manifests verify: E2 126 entries, E2b 46 entries,
  curated CS-670 bundle 84 entries (about 2.63 MB before its manifest).
  The final bundle audit verifies 47 local document links and 18 figure exports.

### 2026-10-03 — E3 replication, E4 tail screen and baseline audit

- Continued under the user's authorization to seek a defensible paper
  contribution independently, with no fixed method direction. No breakthrough
  or paper-level novelty is claimed. The user's permission question concerned
  the official ML-1M download: the network sandbox blocked DNS, escalation was
  approved, and the public archive downloaded successfully. No further research
  instruction was needed.
- Declared `docs/extensions/NOISE_REPLICATION_PROTOCOL.md` before E3 outputs.
  Verified the official archive's MD5 and ZIP CRC; stored raw/split hashes.
  A first checksum-read attempt expected GNU instead of BSD MD5 formatting,
  failed before writing a dataset, and was corrected before preparation.
  The unchanged split functions yield 6035 users, 3706 items and 563206
  training positives, with 6035 validation/test targets each. Fingerprint:
  `e68ef765cc661f5db1864f9f8b2843bee70c7ad261c116421d2c7102ec71bd54`.
  Licensed inputs stay outside CS-670 and are ignored under data/extensions.
- E3 code/protocol frozen at 17:36:48 UTC. Twelve synthetic cases passed.
  Thirty Full/FixedB-r8/Two-r8 runs at epsilon1/2 and five seeds completed,
  transferring E1 settings without tuning. Two hundred probes then completed
  with 2800 observations, comparing raw/aligned coupling and SVD/no balancing
  during continuation. No failures or replacement seeds. Maximum alignment
  product error 3.9126854174e-8. Frozen code:
  `experiments/run_noise_replication.py`; results: `noise_replication_v1`.
- E3 Two-r8 shared-reset lag10 disagreement with SVD, raw/aligned:
  ML-100K epsilon1 18.6641%/.4219%, epsilon2 5.2031%/.0859%; ML-1M epsilon1
  5.2344%/.0625%, epsilon2 3.3438%/.0078%. Without balancing, raw/aligned
  churn matches for every seed; means .3438%/.0781% on ML-100K and
  .0703%/.0156% on ML-1M. These are 128-user observer diagnostics, not
  percent users affected or accuracy improvements. All users still train.
  Turning balancing off only during continuation is not training a new
  optimizer from scratch. The broad intrinsic-memory-instability explanation
  is rejected; a narrower coupling diagnostic survives. Both datasets are
  MovieLens, with population overlap unaudited.
- ML-1M full-population validation means at epsilon1/2: Full
  .019533/.024624; FixedB .013661/.016482; Two .015254/.015440; DP popularity
  .026124/.026161. No superiority test or retuning. Validation is exposed;
  ML-1M test targets have NOT been scored. E3 continuations/diagnostics do
  not inherit the initial epsilon. See `NOISE_REPLICATION_RESULTS.md`.
- Declared `RANKING_TAIL_PROTOCOL.md` and independently checked the known
  product-normal/noncentral-chi-square law. Its mathematics is prior art
  (Gaunt); generic noisy-score ranking stability also overlaps Urmian et al.
  Six synthetic cases with 200000 draws each passed moment, sign-probability,
  invariance and closed-form checks. Frozen at 17:49:38 UTC, E4 inspected
  50 existing states and 3200 predeclared item pairs, using no held-out labels.
  No flagged pairs. Maximum absolute matched-Gaussian error 7.66859e-6;
  linearized error 7.26162e-5. Zero of 1600 primary pairs exceeds .01 in
  either dataset, so the advancement gate failed. No post-hoc expansion.
  Code `experiments/probe_ranking_tails.py`; results `ranking_tails_v1`;
  interpretation `RANKING_TAIL_RESULTS.md`.
- Saved-evidence follow-up audit independently recomputes E3 summaries from
  raw probes, 30 validation means from 181050 saved ranks, E4 seed summaries,
  Gaussian formulas and source/checkpoint hashes. It passed. Archive manifests
  contain 275 E3 and 59 E4 files, without changing older manifests.
- CS-670 now selects 65 source files; three new figures bring the total to
  nine, each in PNG/PDF/SVG and visually inspected. The report, results guide,
  iteration history, claim assessment and prior-art audit include negative
  results. Historical raw sweeps, new per-probe/pair records and checkpoints
  remain in the archive. Added a concise `NOVELTY_STATUS.md`.
- Read the main DPALS algorithm/practical modifications/privacy section and
  supplementary privacy proof; DP-CMF algorithm/privacy/evaluation; and
  Krichene et al. 2023 task-weighted privacy theorem/proof. Added the latter
  citation, bringing the bibliography to 28. The implementation audit records
  why public metadata, local reconstruction and item-frequency weighting are
  established controls. Published ALS and metadata-only personalization are
  still unimplemented. Noise calibration must cover every sufficient statistic
  and match the exact adjacency/sampling; no imported guarantee was claimed.
- Updated both handoffs and report research decisions. The original CS-670
  manifest/audit were preserved in `results/extensions/report_bundle_snapshots/`
  before replacing the current draft's audit. No external messages, commit,
  push or publication. No sub-agent was used.
- Final verification passed after documentation updates: all 3932 protected
  historical files; E1 manifests and snapshots; E2/E2b/E3/E4 manifests;
  frozen source/checkpoint hashes; original E3 endpoint and E4 pair-to-seed
  reconstruction; 65 copied-source hashes; 65 local links; 28 unique references;
  27 figure exports. The refreshed CS-670 manifest verifies 116 files
  (3418001 bytes before the manifest). No full baseline test suite was rerun.
