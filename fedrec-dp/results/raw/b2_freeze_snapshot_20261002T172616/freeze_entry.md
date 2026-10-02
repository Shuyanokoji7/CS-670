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
