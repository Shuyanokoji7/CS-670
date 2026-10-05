# CS-670 final LaTeX report

Completed outputs: [49-page report](CS670_Report.pdf),
[25-page numerical supplement](CS670_Supplement.pdf), and the portable archive
`CS670_LaTeX_Source.zip`.

Open **main.tex** for the complete report and **supplement.tex** for the detailed
numerical supplement. Both are self-contained LaTeX projects within this folder:
the bibliography, vector figures, generated table fragments, and source CSVs
are included. No research dataset or model checkpoint is required to compile.

The report covers Phase 0, B0/B0-UW/B1–B4, E1, E2/E2b, E3, and E4. It includes
the research questions, related work, methods, user-level DP scope, evaluation,
all nine curated figures, results, rejected hypotheses, limitations, future
work, derivations, algorithm details, iteration history, and reproducibility.
The companion supplement retains the full utility and contrast tables,
geometry, activity/cold-target analysis, personalization ablations, diagnostic
endpoints, privacy calibration, rank choices, and communication.

## Files and editing

- `main.tex`, `supplement.tex`: compilation entry points.
- `metadata.tex`: optional author, student ID, supervisor, department and institution.
- `preamble.tex`: layout, packages, mathematical notation and figure macros.
- `sections/`, `appendices/`: editable narrative and mathematical source.
- `references.bib`: 27 formal references. The research search log retains an
  additional screened title with unresolved date metadata; it is not cited here.
- `figures/`: nine existing vector PDF figures, copied without alteration.
- `data/`: selected frozen CSVs, copied without numerical changes.
- `tables/`: generated LaTeX table fragments. Edit the generator, not numeric cells.
- `scripts/generate_tables.py`: reproducible table formatting using pandas.
- `scripts/build_documents.py`: compiler wrapper for both PDFs.
- `scripts/check_sources.py`: dependency, citation, source-identity and build checks.
- `quality/`: source hashes and submission validation records.
- `build/`: local compilation logs and intermediate files, when available.

Author and institutional details were not supplied. Their fields are blank and
omitted from the cover; enter them in `metadata.tex` before handing in. No
supervisor certification, signature or institutional declaration is invented.

## Compile

Upload the source folder/ZIP to Overleaf and choose `main.tex` as the main file;
choose `supplement.tex` to compile the supplement. Use pdfLaTeX. Locally:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

Alternatively, Tectonic handles LaTeX and BibTeX reruns:

```sh
tectonic --keep-logs main.tex
tectonic --keep-logs supplement.tex
```

Table fragments are already generated. Regenerate them only when intentionally
reformatting the retained data, with `python3 scripts/generate_tables.py` in an
environment containing pandas. Compiling does not train or rescore models.

## Scientific scope

This is a complete report of the work performed, including unfavorable results.
It does not claim confirmed paper-level novelty, a new state-of-the-art method,
or an end-to-end DP guarantee for the research process. ML-1M validation has
been scored; its test set remains unscored. The main report states the narrower
guarantees and evidence supporting each conclusion.

## Verification

Both documents were compiled with Tectonic 0.17.0 and its standard bundle.
The final logs contain no unresolved references or citations, duplicate labels,
or overfull boxes. All 21 copied CSVs and nine figure PDFs match the curated
originals. A PDF text-boundary check covers all 74 pages, and 28 rendered pages
were visually inspected (eight front-matter/initial pages and twenty later
samples). The historical evidence audit also passed with all 3,932 protected
files unchanged. These are report and saved-evidence checks, not new training
or a newly rerun full model test suite.
