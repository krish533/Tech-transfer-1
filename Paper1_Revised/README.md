# PCSI manuscript — edited Overleaf-ready version

**Based on:** attached `tt1 (14).pdf` (36 pages, dated September 24, 2026). The source `.tex` was not attached and was not present in the public GitHub replication repository, so this is a **standalone LaTeX reconstruction**, not a patch of the original Overleaf project.

## Overleaf

1. Download this GitHub branch as a ZIP and import the entire repository into Overleaf. Keep `P1_replication_package/paper_outputs/figures/` in the project so the manuscript can find its two figures. Alternatively, use the separate standalone Overleaf ZIP from the ChatGPT revision conversation, which includes cropped images.
2. Set `Paper1_Revised/main.tex` as the Main document.
3. Compile using **pdfLaTeX** (no BibTeX or Biber step is required; `references_manual.tex` is already included).
4. `references.bib` is also provided for eventual use in the original project; the standalone draft does not require it.

## Implemented

- Abstract shortened to fewer than 100 words.
- First paragraph and overall introduction refocused on substantive rules versus communicative framing.
- Results reorganized: distribution, trends, between/within patterns, institutional correlates, linguistic diagnostics.
- Table 2 institution names and dates removed from displayed examples while preserving the literal sentence text.
- R1/R2, land-grant, distributional tests, and multiple-testing notes clarified.
- Table 5 **rerun with a common 144-university sample** in all three columns. New robustness checks include source-year-count weighting, year fixed effects, source-only observations, and exploratory keyword-based topic shares. The coefficients are descriptive and do not identify causal effects.
- Conclusion shortened, with testable future research directions.
- All main tables and appendix tables reconstructed from the supplied PDF; repeated old captions and rogue Section 6.4 text omitted.
- This GitHub branch points to the existing replication-package PDF figures (`fig1_pci_histogram.pdf` and `fig3_temporal_trend_pcsi.pdf`). The separate standalone Overleaf ZIP instead includes image crops from the supplied PDF. Confirm that the GitHub figures match the manuscript's updated statistics and regenerate them from current replication code before submission.

## Before journal submission

- Verify the exact Carnegie classification vintage, timing of institutional characteristics, and original source-document identifiers for Table 2.
- Common-sample Table 5 and the bounded sensitivity checks have been rerun. Exact reproducibility script: `P1_replication_package/paper_outputs/code/rerun_table5.py`; audit report and checked numeric output: `P1_replication_package/paper_outputs/tables/table5_rerun/`. Run the script against the tracked canonical datasets in your local Python environment.
- The keyword-based topic-composition sensitivity is exploratory and **not** the definitive within-topic PCSI analysis. For the latter, join sentence-level classifier scores to expert-validated, substantively comparable legal-policy topics.
- Inspect plotted figures against replication originals (the supplied PDF had overlapping/obsolete caption text).
- Check all bibliography metadata against the original reference library; this reconstruction preserves the cited works but has abbreviated some bibliography titles.

This deliverable combines editorial revisions with re-estimated institutional coefficients and bounded sensitivity checks from the existing canonical replication files. No classifier retraining or new causal analysis was performed.
## GitHub branch notes

This directory was added on the `paper1-manuscript-revision` branch without changing `main`. The manuscript source and both reference formats are here; plots are referenced from the existing replication package to avoid committing reconstructed raster crops. No new estimates have been run.
