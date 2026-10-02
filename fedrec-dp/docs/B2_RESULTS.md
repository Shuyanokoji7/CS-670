# B2 Results Note — User-Level DP Federated BPR

2 October 2026. This is a short results note, not a full report. The decision record is `RESEARCH_LOG.md` (entries from
"Phase B2" through "FINAL B2 results").

## What B2 is

B2 is frozen B1 with four additions:
- **Whole-client clipping** of each shared update ΔQ_u to Frobenius norm C.
- **Gaussian noise** N(0, σ²C² I) added to the **sum** of clipped updates, under assumed/simulated secure aggregation.
- **A fixed denominator** qN = 94.2.
- **Exactly T private rounds**, with no early stopping. Accounting uses the Opacus PRV accountant (primary) and the RDP
  accountant (cross-check).

The privacy unit is one user (add/remove-one-user adjacency), with δ = 1e-5 and Poisson sampling at q = 0.1. p_u stays
local and is never clipped or uploaded.

**Claim scope.** The final B2 training mechanism, conditional on fixed hyperparameters and under the stated
secure-aggregation assumption, is accounted as user-level (ε, δ)-DP. The guarantee does **not** cover the selection of
T, C or η_s on validation, the non-private B1 statistics used to build the clipping grid, the simulated secure
aggregation, or the reproducible (non-cryptographic) noise RNG. Neither C nor the dimension D changes ε.

## Initial vs final protocol

| | Initial (superseded) | Final (frozen 2026-10-02) |
|---|---|---|
| T | 1000 (inherited from non-private B1) | **50** (selected under fixed ε on validation) |
| C | 1.5 | **1.0** |
| η_s | 2.0 | **1.0** |
| ε≈4 test NDCG@10 | 0.0193 | **0.0456** |

The revised protocol changed T, C **and** η_s together and materially improves the private point estimates (ε≈4:
0.019306 → 0.045590). That change is **not attributable to T alone**. The statement that every private level was below
popularity applies only to the superseded initial protocol, which is archived in `results/raw/superseded_b2_T1000/`.

**Final selection** (validation only, final-round NDCG@10):
- **T:** the grid {100, 250, 500, 1000} at ε≈4, with σ re-solved for each T, made T = 100 best. That is the lower edge, so a
  one-time expansion to T = 50 followed. The top two were within 0.003 on seed 42, so three-seed means decided:
  T 50 = 0.0291, T 100 = 0.0241.
- **C:** 1.0 (0.0332), against 0.0230 for C = 1.5 and 0.0105 for C = 2.4.
- **η_s:** 1.0 (0.0344), against 0.0213 for 0.5 and 0.0332 for 2.0.

Tables: `results/b2_t_sweep.csv`, `results/b2_t_selection.csv`, `results/b2_clip_search_T50.csv` and
`results/b2_server_lr_check_T50.csv`.

## Final results (test; mean ± std over seeds 42/123/2026)

| Level | ε PRV / RDP | σ | NDCG@10 | HR@10 = Recall@10* | MRR@10 | Retention |
|---|---|---|---|---|---|---|
| matched no-DP (T 50, qN, η_s 1) | ∞ | 0 | 0.0582 ± 0.0021 | 0.1090 ± 0.0034 | 0.0429 ± 0.0021 | 1.000 |
| ε≈8 | 7.97 / 9.12 | 0.8051 | 0.0544 ± 0.0007 | 0.1030 ± 0.0021 | 0.0397 ± 0.0005 | 0.935 |
| ε≈4 | 3.96 / 4.50 | 1.1519 | 0.0456 ± 0.0010 | 0.0839 ± 0.0046 | 0.0340 ± 0.0020 | 0.783 |
| ε≈2 | 1.99 / 2.22 | 1.7604 | 0.0274 ± 0.0012 | 0.0502 ± 0.0067 | 0.0205 ± 0.0003 | 0.470 |
| ε≈1 | 0.99 / 1.09 | 2.9775 | 0.0115 ± 0.0008 | 0.0226 ± 0.0012 | 0.0082 ± 0.0011 | 0.198 |

\*With one held-out item per user, HR@10 and Recall@10 are identical by definition, so they are not two separate pieces of
evidence.

**Reference points:**
- frozen B1: 0.0875 ± 0.0021;
- popularity: 0.044292, compared on point estimates only. Popularity per-user outputs are not saved and were not re-scored,
  so no paired or significance claim is made. ε≈8 is above it (0.054411, +0.010119), ε≈4 is near it (0.045590, +0.001298),
  and ε≈2 (0.027378) and ε≈1 (0.011526) are below it.

**Paired per-user test NDCG@10** (seed-averaged; 95% bootstrap CI):

| Comparison | Difference | 95% CI |
|---|---|---|
| matched no-DP − B1: **horizon/protocol effect**, not DP | −0.0293 | [−0.0443, −0.0143] |
| ε≈8 − matched no-DP | −0.0038 | [−0.0094, +0.0017], includes 0: no reliable DP loss detected |
| ε≈4 − matched no-DP | −0.0126 | [−0.0201, −0.0056] |
| ε≈2 − matched no-DP | −0.0308 | [−0.0407, −0.0216] |
| ε≈1 − matched no-DP | −0.0467 | [−0.0582, −0.0360] |

## Diagnostics (qualified)

- **Clipping:** 32% of selected clients at ε≈8, rising to 54% at ε≈1.
- **Noise:** the realised noise norm closely matches σC√D (D = 107,648); per-round means differ slightly, as expected. The signal-to-noise ratio
  ‖clipped sum‖ / ‖noise‖ falls from 0.034 at ε≈8 to 0.011 at ε≈1.
- **Item norms:** the final item-vector norms (0.50 / 0.70 / 1.07 / 1.79 for ε≈8 → 1, against 0.11 without DP) stay close
  to the pure-noise random-walk prediction η_s σC√T/(qN)·√d. That is consistent with accumulated Gaussian perturbation
  dominating the *magnitude* of Q. Utility nevertheless survives at ε 8 and 4, so the norm diagnostic alone does not
  determine ranking quality.
- **Overall:** the diagnostics are consistent with accumulated high-dimensional Gaussian perturbation being a major cause of
  the DP loss. B4 is designed to test this directly; it is not established here.

**Accounting at T = 50:** RDP is looser than PRV by different amounts at each level. ε PRV / ε RDP: 7.97 / 9.12 (+14.5%),
3.96 / 4.50 (+13.5%), 1.99 / 2.22 (+11.4%), 0.99 / 1.09 (+9.8%).

**Implication for B4.** Final B2 retains 93.5% of the matched control at ε≈8 and is near popularity at ε≈4. A fair B4
comparison, at the same ε, δ, q, selection rules and accountant, is therefore more demanding than the superseded T = 1000
result suggested.

**Communication:** 861,184 B per selected client per round (dense, float32), about 4.0 GB per 50-round run, compared with
about 81 GB at T = 1000. B2 is the full-rank DP communication reference for B4.

## Limitations

- **Grid edges:** T = 50 and C = 1.0 lie at the edges of their explored grids, and there was no further expansion, by rule.
- **Seeds:** C and η_s were selected on seed 42 only. T was confirmed on three seeds by the seed rule. Each final
  configuration has three seeds.
- **The short horizon has a large non-private cost of its own** (−0.0293 vs B1). Final B2 utility therefore reflects both
  DP noise and the short horizon that the fixed ε budget favours.
- **The popularity comparison uses means only**, not a paired test.
- **Scope:** a single dataset (MovieLens-100K), with a small client population (942 users, about 94 per round).
- **Disclosed exception:** before the freeze, a regression test made 9 accidental real-data test evaluations of 2-round
  models. They were never inspected or used. See `results/raw/audit_pre_freeze_test_regression/` and the audit-correction
  entry in `RESEARCH_LOG.md`.
