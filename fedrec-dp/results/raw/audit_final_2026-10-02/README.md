# Final audit references (2026-10-02)

These files are byte-identical copies of the reference lists used in the final B3/B4 audit.

| File | Contents |
|---|---|
| `raw_hashes_session_start_2026-09-28.txt` | SHA-256 of the 23 MovieLens-100K raw files (`data/raw/ml-100k/*`), recorded at session start on 2026-09-28. |
| `nondoc_hashes_after_verification.txt` | SHA-256 of 877 non-doc files, recorded 2026-10-02 17:30 +05:30, right after the final B2 results. |
| `final_audit.txt` | Summary of the final audit. |

Results of the final audit (summarised in `final_audit.txt`):
- the 23 raw files match;
- 876 of the 877 non-doc files are identical. The one change is the append-only
  `results/raw/b2_final_command_log.txt`, last written at 17:34 during B2 documentation, before any B3 work;
- all B4 hyperparameters match the frozen config;
- the freeze for each phase came before its first test scoring.

**Paths.** All paths are relative to the `fedrec-dp/` project root, so they are portable. To check them, run from that
root, for example:

    sha256sum -c results/raw/audit_final_2026-10-02/raw_hashes_session_start_2026-09-28.txt

**Not bundled in git.** `data/raw/`, `data/processed/` and `checkpoints/` (the `.pt` files) are ignored by
`.gitignore`. They are retained only on the original machine. Every entry for those paths needs the files supplied
locally: download MovieLens-100K into `data/raw/ml-100k/`, regenerate `data/processed/` with `src/data.py` (the split
fingerprint is pinned in the tests), or retrain to regenerate checkpoints. Retrained checkpoints are only expected to
match if every determinism condition is reproduced. Entries for checkpoints, and for later-updated docs and logs, will
not verify from a fresh clone.
