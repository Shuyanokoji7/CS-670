# Curation and reproduction policy

## Inclusion rule

Include a file when it directly supports a report claim, a figure, a method
definition, a limitation, a citation or reproduction of those materials.
The evidence whitelist is executable in [curate_sources.py](curate_sources.py).
[provenance.csv](provenance.csv) records original relative paths, exact SHA-256,
size and the reason for inclusion. Tables are copied without numeric alteration.
Derived selected-rank tables are produced by the figure script and identified
as derived in the results guide.

## Deliberate exclusions

- Superseded parameter sweeps and duplicate freeze snapshots: preserve in the
  research archive; describe material revisions in the iteration appendix.
- Model checkpoints and per-user private local state: large, unnecessary for
  report writing, and outside the DP release guarantee.
- Bulk per-user/round logs, command output and process logs: keep archived for
  auditing; include aggregate and seed-level evidence sufficient for claims.
- Raw MovieLens files, environment packages, bytecode and temporary caches.
- Stale top-level historical README/handoffs as report narrative. Their obsolete
  status text could misrepresent which models are complete.
- Downloaded paper PDFs and third-party summaries. Store checked citations,
  primary links, original comparison notes and reading-depth limits instead.

## Reproduce the report materials

From the parent workspace, using the existing research environment:

```bash
PYTHONDONTWRITEBYTECODE=1 fedrec-dp/.venv/bin/python CS-670/reproducibility/curate_sources.py
MPLCONFIGDIR=/tmp/cs670-matplotlib PYTHONDONTWRITEBYTECODE=1 fedrec-dp/.venv/bin/python CS-670/reproducibility/make_figures.py
MPLCONFIGDIR=/tmp/cs670-matplotlib PYTHONDONTWRITEBYTECODE=1 fedrec-dp/.venv/bin/python CS-670/reproducibility/make_followup_figures.py
PYTHONDONTWRITEBYTECODE=1 fedrec-dp/.venv/bin/python CS-670/reproducibility/coupling_null_example.py
```

The first command refreshes the explicit table/configuration copies from the
sibling archive. The second uses only curated tables and writes PNG, SVG and
PDF figures. The follow-up figure script renders E3/E4; the final command
regenerates a small synthetic algebra check. None
trains models or evaluates held-out targets. Figure error bars are seed SD.
This bundle can travel independently for writing and
plotting; refreshing sources and reproducing training require the archive.

The original B0-B4/E1 reproduction runners live under `fedrec-dp/experiments/`.
Do not execute their test-scoring stages just to regenerate a report figure.
E2 runner: `experiments/probe_noise_memory.py`; E2b:
`experiments/probe_noise_memory_aligned.py`. Both refuse overwriting a started
probe collection. Protocol and source hashes were frozen before real outputs.

To verify local copies, derived summaries, document links and numerical claims:

```bash
PYTHONDONTWRITEBYTECODE=1 fedrec-dp/.venv/bin/python CS-670/reproducibility/verify_bundle.py
```

Add `--archive` to verify the 3932 historical protected files, E1 manifests,
E2/E2b source freezes and original checkpoint/probe records in the sibling
archive. Add `--write-audit` to replace the local verification record after
successful checks. Neither option trains models or scores targets. Updating
files intentionally invalidates the previous bundle manifest; create a new
dated manifest after revisions rather than interpreting that as corruption.

`MANIFEST.sha256` inventories all finalized bundle files other than manifests
themselves. It can be checked from the bundle directory with
`sha256sum --check MANIFEST.sha256`. New E2/E2b archive directories also have
separate manifests; historical E1 manifests are unchanged.

E3 runner: `experiments/run_noise_replication.py`; E4:
`experiments/probe_ranking_tails.py`. Their frozen execution stages refuse
overwriting existing starts. E3 loads validation but never scores test targets;
E4 uses only training interactions and saved factors. The separate
`experiments/summarize_noise_followup.py` recomputes aggregates from saved
evidence and verifies source/checkpoint hashes without training or model
scoring. Its generated completion audit is curated. The E3/E4 report copies
keep compact endpoints and screen summaries; raw per-probe/per-pair files,
local checkpoints and licensed MovieLens-1M inputs remain in the archive.

## Configuration interpretation

Copied configs are immutable source evidence and retain historical comments.
In B3 use `selected_lr`, not the `local_lr: null` template or its stale PENDING
comment. In B4 use `per_rank`. B2 runs `privacy.T=50`, not its unused
`federated.max_rounds=1000`. Noise multipliers are taken from accounting rows,
not the per-run template placeholder zero. B0-UW's file is `b0_uw.yaml`.

The bundle is a content base, not a university-specific thesis template.
Author details, supervisor, department, prescribed chapter order and submission
format must be added when known. No fabricated front matter is included.
