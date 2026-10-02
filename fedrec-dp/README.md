# Dimensionality of Shared Representations in DP Federated Recommendation

## Research Question

**Can reducing the dimensionality of privacy-sensitive shared representations improve the
privacy–utility–communication trade-off in differentially private federated recommender systems?**

## Hypothesis

**H1:** Under the same formal differential privacy guarantee, a lower-dimensional shared
representation may outperform a full-dimensional shared representation in sufficiently
noisy privacy regimes.

This is a hypothesis to be tested, not an assumption. The evaluation framework is
designed so that it can be **disproved**. For example, full dimensionality may win
at every privacy level, or the effect may not exceed seed-to-seed variance.

Expected trade-off:

```text
Too small                       Too large
representation                  representation

low capacity                    high capacity
lower communication             high communication
fewer perturbed dimensions      many perturbed dimensions
       \                              /
        \                            /
         ---- possible sweet spot ---
```

**Lower dimensionality does not automatically provide stronger DP.** The formal privacy
guarantee will depend on the DP mechanism, clipping, sampling, number of rounds, noise
multiplier and privacy accountant. Model dimensionality does not directly change
epsilon under the standard Gaussian mechanism. The research question is whether
dimensionality changes **utility and communication under the same privacy guarantee**.

## Status

Implemented and tested: dataset pipeline, chronological split with leakage checks,
full-ranking evaluation, ranking metrics, random and popularity baselines (Phase 0), and
**B0, centralised BPR matrix factorisation**, **B1, federated BPR without DP**, and **B2,
user-level DP federated BPR** (see below).

**Not implemented yet:** real (cryptographic) secure aggregation and low-rank representations (B3, B4).
The only DP claim in this repository is the narrow B2 claim stated in the B2 section.

- Threat model: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md). User-level DP,
  honest-but-curious server, secure aggregation **assumed/simulated, not
  cryptographically implemented**.
- Decisions and limitations: [`RESEARCH_LOG.md`](RESEARCH_LOG.md)
- Dataset statistics: [`results/DATASET_REPORT.md`](results/DATASET_REPORT.md)

## Setup

Dataset: **MovieLens-100K** (official GroupLens release). Place the extracted `ml-100k/`
folder in `data/raw/`. If it is missing, `python -m src.data` downloads it from
`https://files.grouplens.org/datasets/movielens/ml-100k.zip`. On every load the SHA-256 of
`u.data` is checked against the pinned official value, so modified raw data is rejected.

```bash
cd fedrec-dp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduction commands

All commands are run from `fedrec-dp/` with the venv active. They have been run as written.

```bash
# 1. Preprocess: implicit conversion, filtering, split, leakage validation,
#    writes data/processed/ml-100k/, results/data_summary.json, results/DATASET_REPORT.md
python -m src.data

# 2. Random-ranking baseline  -> results/random_baseline.csv
python -m src.baselines random

# 3. Popularity baseline      -> results/popularity_baseline.csv
python -m src.baselines popularity

# (or both at once)
python -m src.baselines all

# 4. Unit tests: metrics, evaluation protocol, split leakage and determinism
python -m pytest
```

## B0 — Centralised BPR-MF (non-private reference)

B0 is standard BPR matrix factorisation (d = 64, `score = p_u · q_i`, no biases) trained centrally
on the frozen training split. It shows that a plain collaborative-filtering model learns
personalised rankings under this exact pipeline, and it is the non-federated, non-private upper
reference for B1–B4. It uses the same frozen split and the same evaluator (`src/evaluate.py`) as
every other model.

Frozen configuration (`configs/b0.yaml`, revised stopping protocol of 2026-10-02):
- Adam, lr 1e-3, L2 1e-2 (per-example penalty on the batch embeddings), batch 1024;
- one uniform negative per positive (any item that is not a train positive of the user);
- early stopping on validation NDCG@10 with patience 100 epochs, max 1000.

The configuration is the validation-NDCG@10 argmax of a 14-run search. The test set is evaluated
once on the best-validation checkpoint. Earlier B0 configurations and results are archived (see `RESEARCH_LOG.md`).

### B0-UW — user-weighted centralised control (diagnostic)

B0 weights every training *interaction* equally. Federated B1 weights every *user* equally, because
each client averages its own loss and the server averages clients. B0-UW is B0 with exactly one change,
`example_sampling: user_uniform`: each training example is drawn as a uniform user and then a uniform positive of
that user, so the expected loss is the equal-user objective. It keeps B0's hyperparameters, so it is a strict
control and not a tuned model. It separates the weighting effect (B0 → B0-UW) from the federation effect (B0-UW → B1).

## B1 — Federated BPR-MF (no differential privacy)

B1 is the B0 model trained as a simulated cross-device federated system. Each of the 942 users is a client.
- **Local, never sent:** the user vector p_u, which persists across rounds.
- **Global:** the item matrix Q.
- **Each round:**
  - clients join independently with probability q (Poisson sampling; empty rounds are skipped);
  - each selected client downloads Q, runs local full-batch SGD on B0's BPR loss over its own training
    positives, and uploads the dense ΔQ_u = Q_local − Q;
  - the server applies equal-user FedAvg: Q ← Q + mean_u ΔQ_u.
- **Same as B0:** split, candidates, evaluator, d = 64 and scoring function.

Frozen configuration (`configs/b1.yaml`): q = 0.1, local SGD lr 5.0, 2 local epochs, L2 1e-5,
server lr 1, validation every 10 rounds, early stopping after 100 evaluations (1000 rounds)
without improvement (max 8000 rounds). It was chosen on validation only.

### Results (test, seeds 42/123/2026, mean ± std)

| Model | Test NDCG@10 | Test HR@10 (= Recall@10) | Test MRR@10 |
|---|---|---|---|
| Random | 0.0038 | 0.0085 | 0.0024 |
| Popularity | 0.0443 | 0.0870 | 0.0314 |
| B0 centralised, interaction-weighted | 0.0860 ± 0.0005 | 0.1720 ± 0.0021 | 0.0601 ± 0.0008 |
| B0-UW centralised, user-weighted (control) | 0.0917 ± 0.0019 | 0.1819 ± 0.0058 | 0.0645 ± 0.0011 |
| B1 federated, user-weighted | 0.0875 ± 0.0021 | 0.1642 ± 0.0054 | 0.0645 ± 0.0012 |

Paired per-user test NDCG@10 differences, with 95% bootstrap CIs:

| Comparison | Effect measured | Difference [95% CI] |
|---|---|---|
| B0-UW − B0 | weighting | +0.0058 [−0.0002, +0.0119] |
| B1 − B0-UW | federation | −0.0042 [−0.0109, +0.0023] |
| B1 − B0 | total | +0.0016 [−0.0054, +0.0087] |

Against the same-weighting centralised control, B1 retains 95% of NDCG@10, and the gap is not
statistically detectable. Simulated communication (float32 tensors, not network traffic) is 0.86 MB per selected
client per round, with about 94 clients per round. Seed 42 used 84 GB up to its best round. A non-private sparse upload
would be about 10% of the dense size. The dense format is kept because DP noise in B2 must cover every coordinate.

Reproduce (from `fedrec-dp/`, venv active, after `python -m src.data`). All commands were run as written:

```bash
# B0: validation-only search (parallel workers), then the frozen config on three seeds
python experiments/run_b0.py --config configs/b0.yaml --search --workers 4
python experiments/run_b0.py --config configs/b0.yaml --seeds 42 123 2026

# B0-UW control (same runner, different config): seeds, plus the validation-only lr sensitivity diagnostic
python experiments/run_b0.py --config configs/b0_uw.yaml --seeds 42 123 2026
python experiments/run_b0.py --config configs/b0_uw.yaml --search --workers 3

# validation-only stopping-rule check for one seed (never touches test)
python experiments/patience_check.py --config configs/b0_uw.yaml --seed 42 --patience 100

# B1: validation-only staged search, convergence check, seeds (+ B0 comparison, personalisation check)
python experiments/run_b1.py --config configs/b1.yaml --search --workers 4
python experiments/run_b1.py --config configs/b1.yaml --check 0.1 5.0 2 --max-rounds 3000
python experiments/run_b1.py --config configs/b1.yaml --seeds 42 123 2026 --workers 3
python experiments/run_b1.py --config configs/b1.yaml --aggregation-diagnostic

# three-way weighting/federation decomposition (needs the seed runs above)
python experiments/compare_weighting.py

python -m pytest
```

Training is deterministic within an environment: repeated runs reproduce to the reported precision and are
normally byte-identical. See `RESEARCH_LOG.md` for all search trials, protocol revisions and problems.

## B2 — User-level DP federated BPR

B2 is the frozen B1 system plus four additions:
- **Whole-client clipping:** each client's entire shared update ΔQ_u is clipped to Frobenius norm C = 1.5. p_u is local, never
  clipped and never uploaded.
- **Gaussian noise** N(0, σ²C² I) on the **sum** of clipped updates. Secure aggregation is **assumed/simulated**.
- **A fixed denominator** qN = 94.2: Q ← Q + η_s (sum + noise) / qN, with η_s = 2.
- **Accounting:** exactly T = 1000 private rounds with no early stopping, accounted with Opacus PRV (primary) and RDP
  (cross-check). The privacy unit is one user (add/remove-one-user adjacency), δ = 1e-5, q = 0.1. Opacus is used only
  for accounting.

**Claim scope.** The final B2 training mechanism, conditional on fixed hyperparameters and under the stated
secure-aggregation assumption, is accounted as user-level (ε, δ)-DP. Several things are **not** covered:
- the research process (baselines, the non-private clipping-grid statistics, and C/η_s selection on validation);
- validation monitoring;
- the noise RNG, which is a reproducible PCG64 stream and not cryptographically secure.

Lower dimensionality does not change ε.

| Level (test, seeds 42/123/2026) | ε (PRV) | σ | NDCG@10 | HR@10 = Recall@10 | MRR@10 | Retention |
|---|---|---|---|---|---|---|
| matched no-DP (B2 protocol without DP) | ∞ | 0 | 0.0884 ± 0.0046 | 0.1674 ± 0.0060 | 0.0648 ± 0.0040 | 1.00 |
| ε ≈ 8 | 7.94 | 2.06 | 0.0222 ± 0.0050 | 0.0456 ± 0.0074 | 0.0153 ± 0.0043 | 0.25 |
| ε ≈ 4 | 3.97 | 3.56 | 0.0193 ± 0.0058 | 0.0368 ± 0.0086 | 0.0141 ± 0.0050 | 0.22 |
| ε ≈ 2 | 1.99 | 6.43 | 0.0104 ± 0.0024 | 0.0202 ± 0.0055 | 0.0074 ± 0.0014 | 0.12 |
| ε ≈ 1 | 1.00 | 11.98 | 0.0054 ± 0.0022 | 0.0124 ± 0.0043 | 0.0034 ± 0.0017 | 0.06 |

With about 94 clients per round, the noise norm σC√D (D = 107,648) is 70–370× the clipped aggregate, and Q becomes
noise-dominated. **Every private level falls below popularity (0.0443).** The protocol-only control (B1-DPReady, 0.0882) and the
η_s change are both negligible relative to B1. Communication is unchanged from B1 (0.86 MB per selected client per round).
B2 is the full-rank DP reference for B4.

Reproduce (from `fedrec-dp/`, venv active, after `python -m src.data`). All commands were run as written:

```bash
pip install --no-deps opacus==1.6.0 && pip install scipy==1.15.3 opt_einsum==3.4.0   # accountants only

python experiments/run_b2.py --accounting                   # solve sigma per target eps (PRV) + RDP cross-check
python experiments/run_b2.py --norm-stats                   # B1 client-update norm quantiles (no test, no DP)
python experiments/run_b2.py --sanity --clip 1.5            # mechanism checks at eps≈4 (validation only)
python experiments/run_b2.py --clip-search --workers 3      # C grid at eps≈4 (validation only)
python experiments/run_b2.py --server-lr-check --workers 3  # eta_s check (validation only; run because the failure rule fired)
python experiments/run_b2.py --run --epsilon 4 --seed 42    # one B2 run (round-T model, test evaluated once)
python experiments/run_b2.py --sweep --seeds 42 123 2026    # no-DP controls + eps 8/4/2/1, then analysis + plots
python experiments/run_b2.py --analyse --seeds 42 123 2026  # re-run analysis/plots from saved results only
python -m pytest
```

## Protocol summary

| Item | Setting |
|---|---|
| Task | Implicit Top-K recommendation (not rating prediction) |
| Positive | rating >= 4 (training positives and held-out targets). Lower ratings are neither positives nor explicit negatives, but they count as *observed* for candidate filtering |
| Filter | users with < 3 positives removed (1 user: raw id 685, 0 positives) |
| Split | per-user chronological leave-two-out: last positive = test, second-last = validation, rest = train |
| Timestamp ties | equal timestamps contain **no reliable internal chronology** (MovieLens records rating entry time, often in bulk), so ties are ordered by a fixed pseudo-random key SHA-256(user_id, item_id, `split_seed`=2026). Not item id |
| Seeds | `split.split_seed` (2026) is the only seed preprocessing reads. The model-training `seed` (42) can never change the split |
| Candidates | all 1682 catalog items − every item the user rated (**any** rating) strictly before the held-out event. The target is kept. Test therefore also excludes the validation item and low ratings between validation and test. Later ratings never filter earlier candidate sets |
| Ranking | full ranking, no sampled negatives. Score ties ranked above the target (pessimistic) |
| Primary metric | NDCG@10 |
| Secondary | HR@10, MRR@10. Recall@10 == HR@10 here (one held-out item per user), so it is not independent evidence |
| Config | `configs/dataset.yaml` |
| Frozen split | used unchanged by B0–B4. Content fingerprint `6faed6fc3d47…6989` is pinned in `tests/test_data_split.py` |

## Layout

```text
configs/dataset.yaml      central dataset/evaluation config
src/data.py               loading (ML-100K; ML-1M registered, unused), split, validation, report
src/metrics.py            NDCG / HR / Recall / MRR (rank-based and generic list-based)
src/evaluate.py           shared full-ranking evaluation for every model
src/baselines.py          random and popularity baselines
src/utils.py              config, hashing, JSON helpers
src/bpr.py                B0 BPR-MF model, loss, negative sampler
src/train_bpr.py          B0 training loop with validation early stopping
experiments/run_b0.py     B0 search / training / multi-seed summary
configs/b0.yaml           frozen B0 configuration
src/federated.py          B1 federated simulator (local p_u, global Q, sampling, FedAvg, comm accounting)
src/train_federated.py    B1 round loop with validation early stopping
experiments/run_b1.py     B1 search / check / training / multi-seed analysis
experiments/compare_weighting.py  B0 vs B0-UW vs B1 paired decomposition
experiments/patience_check.py     validation-only stopping-rule check
configs/b0_uw.yaml        B0-UW user-weighted control
configs/b1.yaml           frozen B1 configuration
src/privacy.py            B2 clipping, Gaussian aggregate noise, accounting, DP simulator
src/train_dp_federated.py B2 fixed-horizon training loop
experiments/run_b2.py     B2 accounting / norm stats / searches / sweep / analysis
configs/b2.yaml           frozen B2 configuration
checkpoints/              B0, B0-UW, B1, B2 checkpoints (config, seeds, split fingerprint; B2: privacy metadata)
tests/                    test_metrics.py, test_evaluate.py, test_data_split.py, test_bpr.py, test_federated.py, test_privacy.py (92 tests)
docs/THREAT_MODEL.md
results/                  data_summary.json, DATASET_REPORT.md, *_baseline.csv, b0_*.csv, plots/, raw/
data/raw/                 read-only raw data
data/processed/           generated split, full rating history (for candidate filtering), id maps
```

Adding MovieLens-1M later requires only `dataset: movielens_1m` in a config (loader
already registered in `src/data.py`; pin its SHA-256 after the first download).
