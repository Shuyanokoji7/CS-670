# CS-670 UGP report workspace

The formal submission is now a complete [LaTeX report](submission/main.tex),
with a separate [numerical supplement](submission/supplement.tex).
Compiled copies: [report PDF](submission/CS670_Report.pdf),
[supplement PDF](submission/CS670_Supplement.pdf), and
[Overleaf-ready source ZIP](submission/CS670_LaTeX_Source.zip).
See the [submission guide](submission/README.md) for compilation, source layout,
and the optional author/institution fields. The earlier Markdown report remains
as a working draft; the LaTeX version is the submission document.

This folder is the curated writing base for the whole federated recommendation
project. The experimental archive and implementation remain in `../fedrec-dp`.
Materials here are selected for explaining the methods, results and limitations;
superseded sweeps, model checkpoints and bulk logs belong in the archive.

The current research objective is flexible: identify a defensible contribution
in private/federated recommendation, using controlled evidence and checked prior
work. No paper-level novelty is presumed.

## Reading order

1. [Final LaTeX report](submission/main.tex) and
   [supplement](submission/supplement.tex): comprehensive formal account.
   The [earlier draft](report/UGP_REPORT_DRAFT.md) is retained for context.
2. [Results guide](results/README.md): final tables, statistical units and column meanings.
3. [Figures and captions](figures/CAPTIONS.md): nine figures, each in PNG/PDF/SVG.
4. [Methods appendix](appendices/METHODS_AND_LIMITATIONS.md) and
   [material iteration history](appendices/ITERATION_HISTORY.md).
5. [Prior-art audit](references/PRIOR_ART_AUDIT.md) and
   [28-entry bibliography](references/references.bib).
6. [E2/E2b results](research/E2_RESULTS.md),
   [mathematical interpretation](research/NOISE_ATTRIBUTION_NOTE.md) and
   [research decisions](research/RESEARCH_DECISIONS.md).
   Follow with [E3 replication](research/E3_RESULTS.md),
   [E4 rejected candidate](research/E4_RESULTS.md) and
   [current novelty assessment](research/NOVELTY_STATUS.md).
7. [Curation and reproduction](reproducibility/CURATION_POLICY.md),
   [copied-source provenance](reproducibility/provenance.csv) and
   [verification record](reproducibility/audit.json).

## Status as of 2026-10-03

- B0–B4 and E1 remain frozen historical evidence. No historical model was
  retrained or rescored for this report.
- E2/E2b completed 120+40 diagnostic probes, without held-out target scoring.
  The large apparent Two-factor persistence depends strongly on noise coupling;
  smaller local-state effects remain. This is not an improved-recommender claim.
- E3 completed 30 ML-1M training runs and 200 probes. Its balancing control
  narrows the large coupling effect to a representation-dependent diagnostic.
  ML-1M validation is exposed; its test targets have not been scored.
- E4 completed a 3200-pair ranking-tail screen; none of the primary pairs
  exceeded its practical-error threshold. This candidate was rejected.
- There are 65 explicitly selected source files, plus newly authored narrative,
  figures, scripts and one synthetic null example. All originals and superseded
  runs remain in the sibling archive. Bulk checkpoints and logs are excluded.
- The integrity audit verifies all 3932 protected historical artifacts, the
  2148-entry E1 root manifest and its snapshots, frozen probe sources, checkpoint
  hashes, copied tables and regenerated summaries. No new full training-suite
  run is implied; E1's historical 203-test result and E2's synthetic checks
  are documented separately.

A figure or table is included for relevance, including failed hypotheses and
unfavorable findings. The LaTeX submission added on 5 October includes formal
layout, appendices, mathematical derivations and generated numerical tables.
Author and institution details remain editable because they were not supplied.
The diagnostic has a second-dataset replication; paper novelty and practical
value remain open. No new training or test scoring was performed for the report.
