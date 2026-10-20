# Tech-transfer-1

Replication materials are in `P1_replication_package/`.

Main files:
- `P1_replication_package/data/raw/policy_sentences_cleaned_combined.csv`
- `P1_replication_package/data/derived/sentence_scores_canonical.csv`
- `P1_replication_package/data/derived/policy_level_indices_institution_year.csv`
- `P1_replication_package/pipeline/run_all.py`
- `P1_replication_package/paper_outputs/code/generate_paper_outputs.py`

## Manuscript

The paper is in `manuscript/`:
- `main.tex`: the manuscript (compile with pdfLaTeX + BibTeX; `main.pdf` is the compiled version)
- `references.bib`: bibliography
- `tables/`: every table, generated from the replication outputs
- `figures/`: figures copied from `P1_replication_package/paper_outputs/figures/`
- `build_tables.py`: regenerates `tables/` and `figures/`
- `verification/`: citation register, source-verification status, and Carnegie provenance note

To rebuild everything from the data:

```bash
cd P1_replication_package
python pipeline/04_build_primary_analysis_panel.py
python paper_outputs/code/generate_strengthened_results.py --panel-file data/derived/policy_level_indices_primary_1944_2025.csv --sentences-file data/derived/sentence_scores_canonical.csv
python paper_outputs/code/generate_final_manuscript_figures.py --panel-file data/derived/policy_level_indices_primary_1944_2025.csv
python paper_outputs/code/generate_revision_analyses.py
cd ..
python manuscript/build_tables.py
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

For Overleaf, link this repository through Overleaf's GitHub integration and set `manuscript/main.tex` as the main document.

