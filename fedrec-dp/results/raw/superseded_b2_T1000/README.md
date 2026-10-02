# Superseded B2 (fixed T = 1000 inherited from B1) — archived 2026-10-02

Initial B2 protocol: T = 1000, C = 1.5, eta_s = 2, q = 0.1, delta = 1e-5, PRV primary. All result CSVs, per-user files,
diagnostics, plots and the config are as produced. Superseded by the privacy-aware horizon study (RESEARCH_LOG.md).

KNOWN ISSUE: checkpoints/b2_nodp_seed{42,123,2026}.pt were overwritten by the B1-DPReady control (eta_s = 1), because both
used the name b2_nodp_seed*.pt. They therefore contain B1-DPReady weights (reload test NDCG@10 0.0910/0.0855/0.0882 =
B1-DPReady's saved results). The matched no-DP (eta_s = 2) checkpoints were lost. All result CSVs/per-user files are
correct (their file names include eta_s). Fixed in the runner by naming files with T, C and eta_s.
