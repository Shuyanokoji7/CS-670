# Audit evidence: pre-freeze real-data test-path regression jobs (DO NOT USE FOR MODEL CHOICE)

The B2 protocol regression test `test_frozen_test_jobs_write_distinct_T_C_slr_outputs` (tests/test_b2_protocol.py, written
2026-10-02) redirected outputs and config to a temporary directory, but it loaded the REAL frozen MovieLens split. Its three
`evaluate_test=True` jobs (eps≈4 / no-DP eta_s 2 / no-DP eta_s 1; T = 2 rounds; seed 42) therefore evaluated the real held-out
test labels, before the final B2 freeze.

This happened in 3 pytest sessions, i.e. 9 test evaluations of 2-round models:
- the first file-only run (pytest-19; its temporary directory was auto-deleted by pytest's retention policy, so there is no
  evidence);
- the file rerun (pytest-21, copied here);
- the full suite (pytest-22, copied here).

The scores were never printed, inspected or used, for T, C or η_s selection or for anything else. They are preserved here,
unopened, only as audit evidence. `MANIFEST.sha256` lists every file. The regression tests now use a synthetic split for all
test-path checks.
