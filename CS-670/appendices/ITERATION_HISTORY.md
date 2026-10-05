# Material experiment revisions

This is the report-level history needed to interpret final evidence. The complete
chronology, commands, superseded outputs and snapshots remain in
`../../fedrec-dp/RESEARCH_LOG.md` and `../../fedrec-dp/results/raw/`.

| Stage | Revision / issue | Consequence for the report |
|---|---|---|
| Phase 0 | Fixed positive threshold, chronological tie breaking, full-ranking candidate filtering and raw hashes. | Use one canonical evaluation protocol across final baselines. |
| B0/B1 | Longer stopping protocol and added B0-UW. | Original apparent federation effect mixed interaction/user weighting. Overall paired intervals for that decomposition include zero. |
| B2 | Initial T1000/C1.5/server2 superseded by T50/C1/server1 after validation studies. | Three settings changed; do not attribute the final gain solely to shorter training. |
| B3 | Some initial rates diverged; rank32 needed a declared convergence-budget correction. | Use final selected rates and completed five-seed outcomes. Retain the convergence exception. |
| B4 | Inherited long-horizon rates undertrained at T50; expanded bounded tuning and stability fallback. | B4 received more tuning than B2; final table is not an isolated dimension experiment. |
| Historical test exposure | Nine accidental two-round B2 real-data test evaluations in earlier regression tests. | Not inspected or used for selection, but ML-100K cannot be described as pristine. Original audit remains archived. |
| E1 | Separately named, equal four-candidate search with common Two/FixedB effective initialization. | It is a controlled extension, not a rerun or replacement of B4. Eleven failures are retained. |
| E1 diagnostics | Float32 mean local-user norm overflowed for a very large but finite trajectory. | Corrected analysis reads P in float64; original raw logs preserved, no altered training or test ranks. |
| E1 verification | Writing live pytest stdout inside guarded results caused fixture errors. | Rerun to /tmp passed 203 tests; both logs are retained. This was a harness/output issue. |
| E2 | Pulse/reset diagnostic declared before its outputs; 120 checkpoint probes. | Exploratory, no held-out scoring or new DP claim. |
| E2b | Large Two-factor probe distances motivated a separate alignment-control protocol. | Disclosed post-E2 adaptation; not preplanned confirmatory evidence. |
| E3 | Transferred frozen settings to ML-1M; added raw/aligned coupling with/without continuation balancing. | 30 training runs and 200 probes completed. The large coupling gap depends on balancing; ML-1M validation is exposed but its test remains unscored. |
| E4 | Screened known product-normal ranking tails after checking overlapping literature. | 50 states and 3200 predeclared pairs show negligible practical Gaussian-risk error. Candidate rejected without enlarging the sample after inspection. |

Failed/unstable results are summarized because they affect conclusions. Bulk
intermediate sweeps are excluded from the writing folder to keep the active
evidence unambiguous, not to hide unfavorable outcomes. E1's raw outputs,
manifests and historical B0-B4 outputs remain the source of record.
