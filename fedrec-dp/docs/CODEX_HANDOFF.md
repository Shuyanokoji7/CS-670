# Codex Handoff — DP Federated Recommender Research (`fedrec-dp`)

Written 2026-10-02. It is for an agent continuing this project mid-phase. Read it fully before running anything.
`RESEARCH_LOG.md` is the authoritative decision record. This file summarises it and gives the exact current state.

---

## 0. TL;DR — current state (updated 2026-10-02 after FINAL B3/B4)

> Earlier versions of this handoff are archived in `docs/archive/` (pre-B2-final and pre-B3B4-final). Sections 4.1–4.2
> are historical. **The current state is §0 plus the "B3/B4 handoff update" at the end.**

- **Research question:** under the *same* user-level (ε, δ) guarantee, can a lower-dimensional shared item representation
  improve the privacy–utility–communication trade-off of DP federated BPR? This was treated as a hypothesis to test.
- **Phases:** Phase 0, B0, B0-UW, B1, B2 (final), B3 (revision 2) and B4 are all **frozen and test-scored**. Nothing is
  in progress, and no further experiment is authorised.
- **Answer:** limited support for H1. Low-rank is reliably better than frozen B2 only at ε 2 (ranks 8/16/32) and at ε 1
  (rank 8). It is worse at ε 8, and no reliable difference is detected at ε 4. Without DP, every rank is worse than B1.
  See `docs/B3_RESULTS.md` and `docs/B4_RESULTS.md`.
- **Popularity point reference** (train-only, test NDCG@10 0.0443): every five-seed DP mean at ε ≤ 2 is below it. B4 r32
  at ε 4 is numerically close (0.0445 vs 0.0443). No paired significance claim is made.
- **Tests:** 181 passing (`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider`).

---

## 1. Environment

```bash
cd "/home/ag/Amazon ML/fedrec-dp"
source .venv/bin/activate          # Python 3.10, CPU-only torch 2.14.0+cpu
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider   # expect 113 passed (final B2)
```

- `requirements.txt` pins numpy 2.2.6, pandas 2.3.3, PyYAML 6.0.3, pytest 9.1.1, torch 2.14.0+cpu (CPU index),
  matplotlib 3.10.9, scipy 1.15.3, opt_einsum 3.4.0, opacus 1.6.0.
- **Install opacus with `pip install --no-deps opacus==1.6.0`**, otherwise it may replace the pinned torch. Opacus is
  used **only for accounting** (PRVAccountant / RDPAccountant). Never use PrivacyEngine: the privacy unit is a whole client
  update, not an example.
- 20 CPU cores, 2 GPUs (unused; CPU is chosen for determinism).
- Raw data: `data/raw/ml-100k/` (the user's upload, copied). Never write there. The `u.data` SHA-256 is pinned in
  `src/data.py` and checked on every load.

## 2. Ground rules from the user (non-negotiable; they have enforced all of these)

1. **Never change the frozen split** (fingerprint `6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989`),
   the evaluator (`src/evaluate.py`), candidate rules or metrics.
2. **Select on validation only.** The test set is evaluated **once per frozen configuration**, after the configuration is
   frozen. Never tune, select T/C/lr, or pick checkpoints using test.
3. **Declare protocol changes in `RESEARCH_LOG.md` BEFORE running them**, and before any test evaluation. The log is
   **append-only**: add dated entries; don't rewrite history. Mark superseded entries with a short note at the top.
4. **Archive, never delete** superseded outputs (results, per-user files, checkpoints, plots, configs) under
   `results/raw/superseded_*`.
5. **Separate confounds** with explicit controls (e.g. B0-UW for weighting, B1-DPReady / matched no-DP for protocol vs
   DP). Report each effect separately with paired per-user bootstrap CIs.
6. **Report honestly.** No fabricated or rounded-up numbers, no "significant" when the CI includes 0, and no claims that
   go beyond the evidence (e.g. "consistent with", not "caused by"). When a number in a doc turns out wrong, fix it and
   say so.
7. **Privacy claim wording** (exact scope): *"The final B2 training mechanism, conditional on fixed hyperparameters and
   under the stated secure-aggregation assumption, is accounted as user-level (ε, δ)-DP."* Secure aggregation is
   **simulated/assumed**. The noise RNG is reproducible PCG64, **not** cryptographic. **Dimension D and clip norm C do
   NOT change ε.**
8. **README commands must have actually been run** exactly as written.
9. **Do only the phase asked.** The user gates every phase explicitly (B3/B4 not until asked).
10. Pre-declared selection rules are followed even when the result is unappealing (e.g. B2's η_s = 2 was frozen by
    rule although the differences were noise-level). Don't override a rule after seeing numbers; report the tension.

## 3. Phase summary (all test numbers: mean ± std over seeds 42/123/2026)

| Phase | What | Frozen config | Test NDCG@10 |
|---|---|---|---|
| Phase 0 | ML-100K, rating ≥ 4 positive, 942 users, chronological leave-two-out, hash tie-break (split_seed 2026), full ranking excluding all previously rated items | `configs/dataset.yaml` | random 0.0038, popularity **0.0443** |
| B0 | centralised BPR-MF, d = 64, Adam | lr 1e-3, L2 1e-2, batch 1024, patience 100 epochs, max 1000 | 0.0860 ± 0.0005 |
| B0-UW | B0 with user-uniform example sampling (equal-user objective); strict control, not tuned | `configs/b0_uw.yaml` | 0.0917 ± 0.0019 |
| B1 | federated BPR, local p_u, global Q, Poisson q = 0.1, local full-batch SGD lr 5, E = 2, L2 1e-5, equal-user FedAvg (÷ realised \|S_t\|), dense ΔQ | `configs/b1.yaml`, patience 100 evals (1000 rounds) | 0.0875 ± 0.0021 |
| **B2 FINAL (frozen 2026-10-02)** | B1 + whole-client clip C = 1.0 + N(0, σ²C²) on the clipped sum + ÷ qN = 94.2, η_s = 1, T = 50, δ 1e-5, PRV primary / RDP cross-check | `configs/b2.yaml` (protocol_frozen true); `docs/B2_RESULTS.md` | matched no-DP 0.0582; ε≈8 0.0544, ε≈4 0.0456, ε≈2 0.0274, ε≈1 0.0115 |
| B2 (initial, T = 1000, **superseded**) | B1 + whole-client clip C = 1.5 + N(0, σ²C²) on the clipped sum + ÷ qN = 94.2, η_s = 2, δ 1e-5, PRV primary / RDP cross-check | archived in `results/raw/superseded_b2_T1000/` | no-DP matched 0.0884; ε≈8 0.0222, ε≈4 0.0193, ε≈2 0.0104, ε≈1 0.0054 |

Key findings to preserve:

- **Weighting decomposition:** B0-UW − B0 = +0.0058 [−0.0002, +0.0119]; B1 − B0-UW = −0.0042 [−0.0109, +0.0023];
  B1 − B0 = +0.0016 [−0.0054, +0.0087]. Point estimates suggest a weighting contribution to B1's parity with B0, but all
  overall CIs include zero, so no reliable difference is detected.
- **Final B2 (T 50, C 1.0, η_s 1):** ε≈8 0.054411 (93.5% of matched no-DP 0.058196; DP effect CI includes 0), ε≈4 0.045590
  (near popularity 0.044292; point estimates only), ε≈2/1 below popularity. B1 → matched horizon/protocol effect −0.029344
  [−0.044349, −0.014269]. The revised T/C/η_s together, not T alone, improve on the superseded T = 1000 result. A fair B4
  comparison is therefore more demanding.
- **Initial B2 (superseded, T = 1000):** every ε was below popularity. The item embeddings at round T match a pure-noise random walk
  (predicted ‖item row‖ = η_s σC√T/(qN)·√d, e.g. 28.7 vs 28.4 measured at ε≈4). Per-round ‖clipped sum‖/‖noise‖ is 0.003–0.014.
  The protocol control (B1-DPReady, η_s 1) and the η_s change are both negligible relative to B1.
- **Communication:** dense protocol, 861,184 B per selected client per round (430,592 each way, float32). A sparse
  touched-row upload would be about 9.8% of that. Dense transmission is THIS project's dense reference protocol for DP noise
  on the shared update. It is not a universal requirement. Frame B3/B4 as "reducing the
  dense shared representation required by the DP-compatible protocol", not as "solving FL communication".

## 4. B2 robustness check (privacy-aware training horizon) — HISTORICAL record; current state is §4.3

> §4 intro, §4.1 and §4.2 are retained as written **before** final B2. Their values ("92 tests", "NOT yet exercised",
> next steps) are historical. Current values: 113 tests, B2 frozen and complete (§4.3).

The user's full spec is the last message in the session. Summary: at a **fixed ε ≈ 4** (δ 1e-5, q 0.1), only T changes
and σ is re-solved per T. The question is whether a shorter horizon materially fixes the collapse.
**The protocol is already declared** in `RESEARCH_LOG.md` → entry "2026-10-02 — B2 robustness check: privacy-aware
training horizon". Read it; follow it literally.

### 4.1 Done so far

- [x] Verified: 92 tests pass, raw hash unchanged, fingerprint unchanged, B1 output MD5s unchanged (pinned in
      `tests/test_privacy.py::B1_FROZEN_MD5`), no B3/B4 code.
- [x] B2 checkpoint reproducibility check found a **naming-collision bug** (B1-DPReady overwrote the matched no-DP
      checkpoints `b2_nodp_seed*.pt`). The reported numbers are unaffected. It is logged in RESEARCH_LOG and in the archive README. Fixed:
      all B2 outputs are now named `{level}_T{T}_C{C}_slr{eta_s}_seed{seed}`.
- [x] Archived the initial B2 outputs → `results/raw/superseded_b2_T1000/` (139 files plus a README with the known issue).
      **Copies** of these old files are still in `results/b2_*.csv`, `results/plots/b2_*.png`,
      `checkpoints/b2_{level}_seed*.pt`. They will be overwritten by the final sweep's analysis. That is fine because they are archived.
- [x] Protocol declared in RESEARCH_LOG (T grid, boundary rule, seed rule, C recheck, η_s condition, freeze gate).
- [x] `configs/b2.yaml`: `T: 1000 # UNDER REVISION`, `protocol_frozen: false`,
      `t_grid: [100, 250, 500, 1000]`, `t_boundary: {low: 50, high: 1500}`, `t_seed_rule_margin: 0.003`,
      `t_seed_rule_seeds: [123, 2026]`.
- [x] `experiments/run_b2.py` refactored. It is **NOT yet exercised end-to-end after the refactor**, so expect small bugs:
  - `sigma_for(cfg, eps, T)` solves or looks up σ per (T, target ε) and caches it in `results/b2_accounting.csv`, which has a T column.
  - `run_one(job(...))` takes a job dict: epsilon, seed, clip, server_lr, evaluate_test, T.
  - `require_frozen(cfg)` blocks every test evaluation unless `protocol_frozen: true`. It is called in `run_one` when
    `evaluate_test`, and in `sweep`.
  - New per-run diagnostics: `final_item_norm_mean`, `pure_noise_item_norm_prediction`, `final_user_norm_mean`.
  - `--t-sweep` implements the grid **plus the boundary and seed rules automatically**, and writes `results/b2_t_sweep.csv`.
  - `--clip-search` and `--server-lr-check` now run at the configured T and write `results/b2_clip_search_T{T}.csv` and
    `results/b2_server_lr_check_T{T}.csv`.
  - `analyse()` expects T-tagged file names.

### 4.2 Next steps (in order)

1. **Add the tests** required by the spec (§18) before running, e.g. in `tests/test_privacy.py` or a new
   `tests/test_b2_protocol.py`. Use the RDP accountant in tests for speed. They must check:
   - σ is re-solved when T changes, and a larger T needs a larger σ for the same target ε (`solve_sigma`, `sigma_for`);
   - the accountant composes exactly T steps: `run_one` asserts `sim.round == T == rec["T"]`, and also test
     `train_dp_federated(T=3)` → 3 rows, `privacy_record(..., T)`;
   - the validation-only path never evaluates test: monkeypatch `experiments.run_b2.evaluate` and
     `src.train_federated.evaluate` to raise on `target == "test"`, then call `run_one(job(4, 42, T=2))` (needs the σ cache
     entry for T = 2; or monkeypatch `sigma_for`);
   - the final test runner refuses unless frozen: `require_frozen`, and `run_one(job(..., evaluate_test=True))` with an
     unfrozen config raises;
   - B1 outputs are unchanged (this test already exists).

   Do not weaken the existing 92 tests.
2. `python experiments/run_b2.py --t-sweep` (validation only, seed 42, about 1–3 min per run with 15 workers). Report the
   table: T, target ε, σ, PRV ε, RDP ε, val NDCG@10, HR@10, MRR@10, frac clipped, median pre-clip norm, clipped-sum norm,
   noise norm, SNR, final item norm.
3. Set `privacy.T` in `configs/b2.yaml` to the selected T, then run `--clip-search` (existing grid {1.0, 1.5, 2.4}, argmax
   final validation NDCG@10; the smaller-C tie preference is NOT applied, see the log). Set `clip_norm`.
4. η_s recheck (`--server-lr-check`, {0.5, 1, 2}) **only if** the selected T ≠ 1000 **and** its seed-42 val NDCG@10 beats
   T = 1000's by > 0.003. Otherwise keep η_s = 2.
5. **Write the freeze entry to RESEARCH_LOG** (selected T, C, η_s, q, δ, local lr 5, E 2, PRV/RDP, target ε levels, exact
   selection logic, all validation numbers). It must state: **"No test-set result has yet been evaluated under this revised
   B2 protocol."** Then set `protocol_frozen: true` and update the `T:` comment.
6. `python experiments/run_b2.py --accounting` (solves σ for ε ∈ {8, 4, 2, 1} at the final T), then
   `python experiments/run_b2.py --sweep --seeds 42 123 2026 --workers 15`. This runs the matched no-DP control (same T, qN,
   η_s), B1-DPReady (η_s 1) if η_s ≠ 1, and the four ε levels; it evaluates test once and runs `analyse()` (summary, paired,
   per-user, diagnostics, plots).
   - **Note:** if the final T < 1000, the no-DP control at T rounds will be under-trained relative to B1. Report that
     B1 → control "protocol/horizon effect" separately and **don't attribute it to DP**. Retention is DP / matched no-DP.
     Also give popularity (0.0443) as the external reference.
7. Final diagnostics per level (already in `results/b2_summary.csv` after `analyse`): clip fraction, median pre-clip norm,
   shrink factor, clipped-sum norm, realised noise norm vs σC√D, SNR, final item norm vs pure-noise prediction.
8. Interpretation rule: if a shorter T substantially improves B2, the T = 1000 result stays visible as the **"superseded
   fixed-1000-round B2 result"** (horizon confound). If utility stays poor: *"Full-rank user-level DP remains highly destructive
   even after privacy-aware training-horizon selection."* Never say "high dimensionality is definitely the cause"; say *"The
   diagnostics are consistent with accumulated high-dimensional Gaussian perturbation being a major cause."*
9. Docs: README B2 section (distinguish **Initial B2** with T = 1000 inherited from B1, and **Final B2** with privacy-aware T),
   a RESEARCH_LOG results entry, a short B2 results note (no `docs/B2_REPORT.md` exists yet; ask whether the user wants a full
   report). Re-verify: tests, raw hash, fingerprint, B1 MD5s. Run every README command you list.
10. Final response format: the user's spec §22 (20 numbered sections). Then **stop; do not start B3/B4.**

### 4.3 CURRENT STATE (2026-10-02): B2 complete and frozen

- Robustness check done, with all rules declared in `RESEARCH_LOG.md` before running:
  - T sweep → T 50 (boundary rule, then the seed rule);
  - C argmax → 1.0;
  - η_s check (its condition was met) → 1.0.
- Freeze entry written before any experimental test evaluation. Evidence snapshot:
  `results/raw/b2_freeze_snapshot_20261002T172616/`.
- Accounting at T 50 (`results/b2_accounting.csv`, full-key cache): σ = 0.8051 / 1.1519 / 1.7604 / 2.9775 for ε 8 / 4 / 2 / 1.
- One final sweep (15 runs, exit 0). Outputs: `results/b2_summary.csv`, `b2_paired.csv`, `b2_privacy_sweep.csv`,
  `b2_per_user_test.csv`, `b2_diagnostics.csv`, plots, and `checkpoints/b2_*_T50_*`.
- **Disclosed audit exception:** 9 pre-freeze real-data test evaluations of 2-round models were made by an earlier regression
  test. They were never used. Evidence is in `results/raw/audit_pre_freeze_test_regression/`. All runner tests now use a
  synthetic split.
- Archives: `results/raw/superseded_b2_T1000/` (intact, 139 files), `archive_before_C_lr_recheck_2026-10-02/`,
  `archive_before_final_sweep_2026-10-02/`.
- **Next:** wait for an explicit B3/B4 authorization. When it comes, B4 must be compared with final B2 *and* popularity at a
  matched ε, T and q, using the same accountant. Note that the final B2 horizon (T 50) has its own large non-private cost.
- Still open with the user:
  - the "MKL" wording in the older docs;
  - the length of the B1 report;
  - a paired popularity comparison, which would need popularity per-user test ranks to be re-scored and is not done.

## 5. Code map

| File | Role |
|---|---|
| `src/data.py` | loading (raw hash check), hash tie-break split, `previously_rated`, `split_fingerprint`, `load_processed` (re-validates) |
| `src/evaluate.py` | frozen full-ranking evaluator, pessimistic ties. **Do not change.** |
| `src/metrics.py` | NDCG/HR/Recall/MRR |
| `src/bpr.py`, `src/train_bpr.py` | B0/B0-UW model and training (`example_sampling: interaction \| user_uniform`) |
| `src/federated.py` | B1 simulator (`FederatedBPR`, `local_update` with hand-derived gradients, Poisson `sample_clients`, `aggregate`, comm accounting) |
| `src/train_federated.py` | B1 loop with validation early stopping |
| `src/privacy.py` | B2: `clip_update` (1e-6 margin plus assertion), `gaussian_noise`, `private_update`, `compute_epsilon`/`solve_sigma` (Opacus), `privacy_record`, `DPFederatedBPR` (subclass of B1; bit-identical to B1 when C = ∞, σ = 0 and the denominator is realised) |
| `src/train_dp_federated.py` | B2 fixed-horizon loop (exactly T rounds, validation diagnostics only) |
| `experiments/run_b0.py` | B0 and B0-UW (via `--config configs/b0_uw.yaml`; output prefix from the config) |
| `experiments/run_b1.py` | B1 search/check/seeds/analysis |
| `experiments/run_b2.py` | B2 (see §4) |
| `experiments/compare_weighting.py`, `experiments/patience_check.py` | B0 vs B0-UW vs B1 decomposition; stopping-rule check |
| `tests/` | 113 tests: data split (incl. pinned fingerprint), metrics, evaluate, bpr, federated, privacy (incl. pinned B1 MD5s), b2_protocol (synthetic-split runner tests, full-key σ cache, T-selection rules) |
| `docs/` | `THREAT_MODEL.md` (B2 simulator-vs-protocol note added), `PHASE0_REPORT.md`, `B0_REPORT.md` (superseded notice at top), `B1_REPORT.md` (revised with B0-UW), `B2_RESULTS.md` (final B2 results note), this file, `archive/` (previous handoff) |

## 6. Pitfalls already hit (don't repeat)

- **Output-name collisions:** any two runs that differ in a hyperparameter must have different file and checkpoint names.
- **Float32 clipping:** a clipped norm can exceed C by about 1e-8 relative. Keep the 1e-6 margin and assertion.
- **σ precision:** store σ with `%.12g`; recompute ε from the stored σ when checking.
- **Determinism:** keep `OMP/OPENBLAS/MKL_NUM_THREADS=1` in the B1/B2 runners and torch `num_threads=4` for B0. Changing
  thread counts can change float sums.
- **Parallel accounting:** solve σ sequentially *before* starting parallel workers, because the cache file is shared
  (`sweep` and `t_sweep` already do this).
- **Shell wait loops:** `pgrep -f "<pattern>"` can match its own command line and hang. Use `wait`, or background tasks.
- **Early-stopping patience:** both patience 10/30 (B0) and 50 (B1) were too short on noisy validation curves; the common
  protocol is now 100 validation checks. B2 uses no early stopping at all (fixed T).
- **Analysis-only reruns:** if analysis or plotting fails after training, fix it and run `--analyse`. Never retrain just
  to regenerate tables (that would also re-evaluate test).

## 7. Open items awaiting the user (do not resolve unilaterally)

- `docs/B0_REPORT.md`, `README.md` and `RESEARCH_LOG.md` still say a one-off B0 seed-42 last-bit difference was "most likely
  MKL". Later probes showed it is **not** MKL, alignment or venv related (B0 training has no GEMM; other runs are identical). The user
  called it minor and has not yet asked for a rewording.
- `docs/B1_REPORT.md` is about 6,800 words; an appendix move was offered. No decision yet.
- ~~No B2 report exists~~ — `docs/B2_RESULTS.md` exists. Concise B3/B4 results docs also exist.
- ~~Before B3/B4: judge B4 against popularity~~ — done: see §0 and docs/B4_RESULTS.md. Every five-seed DP mean at ε ≤ 2 is
  below the popularity point reference (0.0443).

## 8. Verification snippet

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider   # 113 passed
python -c "from src.data import load_processed, split_fingerprint; print(split_fingerprint(load_processed()))"
# -> 6faed6fc3d47b5b6fa638adfeea83cd7409d50c39fa01f85c10379d0daef6989
md5sum results/b1_test_results.csv   # 8d3cfcee6f29c7c9dba6338497988cbd (others pinned in tests/test_privacy.py)
```

---

## B3/B4 handoff update (2026-10-02, after final test scoring)

- **B3**, frozen in revision 2 (lr {4: 0.625, 8: 1.25, 16: 1.25, 32: 0.625}; the rank 32 seed 123 run used a 16000
  ceiling instead of the original 8000, and every other run stopped early below 8000). See `docs/B3_RESULTS.md`.
  **Superseded stages:** the initial freeze and correction 1. Both failed the gate, and neither was test-scored.
- **B4**, frozen with the bounded fallback (C 1; r4 lr 2.5 / η_s 1; r8/16/32 lr 5 / η_s 0.5). See `docs/B4_RESULTS.md`.
  **Superseded or rejected:** the original B3-lr-inheritance protocol, and the lr × C plus η_s winners, rejected on
  five-seed no-DP stability. **Low-rank had a larger tuning budget than frozen B2.**
- **Analysis:** `experiments/analyse_b3b4.py` writes `results/b34_{test_means,contrasts,perturbation,communication}.csv`.
- **Supplemental seeds 7/99:** B1 in `results/raw/supplemental_b1/`, B2 in `results/raw/supplemental_b2/`. Both are exact
  frozen replications, and the core B1/B2 outputs are unchanged (876 of 877 post-B2 non-doc files identical; the one change is the
  append-only `results/raw/b2_final_command_log.txt`, explained in RESEARCH_LOG).
- **Limits:** DP holds for the training procedure under the threat model, and ε is equal by construction (no claim that
  dimension reduces ε). There was prior test exposure, so there is no pristine holdout. Comparisons are conditional on
  these training runs. No isolated-dimensionality causal claim is made. A smaller per-round payload does not imply
  smaller total traffic: B3 r32 needs 1.92× B1's volume to reach its best round.
- **Tests:** 181 passing. The new B3/B4 experimental tests use synthetic fixtures; the existing suite also includes
  pinned-fingerprint and integrity checks. Nothing is committed or pushed.
- **Supervisor-reviewed reading:**
  - Only four of the 16 adjusted cells are reliably positive: r8/r16/r32 at ε 2, and r8 at ε 1.
  - Every rank is worse than B2 at ε 8, and none shows a reliable gain at ε 4.
  - The validation-selected r16 at ε 1 is positive only at 95%, not after adjustment. It is not replaced by the
    test-best r8.
  - This is limited support for H1.
  - Raw factor noise is not effective-Q noise.
  - B4 volume is 0.065/0.130/0.260/0.519× the matched B2's.

## Completion status (2026-10-02)

B3 and B4 are **finalized**. No further runs are authorised or pending, all frozen and core outputs are unchanged, and
nothing is committed or pushed.
