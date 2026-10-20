# Replication Package: Communication as Governance

This package reproduces the measurement pipeline and manuscript analyses for **Communication as Governance: Measuring Communicative Stance in University Intellectual Property Policies**.

The manuscript's primary analytical window is **1944--2025**. The underlying archive retains an isolated one-sentence Caltech record from 1925 for provenance and robustness checks, but the main paper does not use its 1925--1943 carry-forward observations.

## Package structure

### Core data chain
- `data/raw/policy_sentences_cleaned_combined.csv`
- `data/derived/sentence_scores_canonical.csv`
- `data/derived/policy_level_indices_institution_year.csv` (full canonical archive panel)
- `data/derived/policy_level_indices_primary_1944_2025.csv` (generated primary manuscript panel)

### Model
- `model/srn_cls_model/`

### Replication pipeline
- `pipeline/01_build_input_corpus.py`
- `pipeline/02_score_sentences.py`
- `pipeline/03_build_policy_level_indices.py`
- `pipeline/04_build_primary_analysis_panel.py`
- `pipeline/run_all.py`

### Manuscript outputs
- core output code: `paper_outputs/code/generate_paper_outputs.py`
- strengthened results/robustness code: `paper_outputs/code/generate_strengthened_results.py`
- final-manuscript figure code: `paper_outputs/code/generate_final_manuscript_figures.py`
- figures: `paper_outputs/figures/`
- core tables: `paper_outputs/tables/`
- strengthened tables/results: `paper_outputs/tables/strengthened_results/`
- appendix support files: `paper_outputs/appendix/`

## Current corpus counts

### Full archive
- raw sentence rows: `87,160`
- canonical scored sentence rows: `87,160`
- institutions: `150`
- observed source institution-years: `481`
- source policy documents: `519`
- policy-in-force institution-year rows through 2025: `4,296`

### Primary manuscript sample, 1944--2025
- sentence rows: `87,159`
- institutions: `150`
- directly observed institution-year policy records: `480`
- source policy documents: `518`
- policy-in-force institution-year observations: `4,277`

Unless explicitly labeled as a full-archive robustness exercise, manuscript results use the 1944--2025 sample.

## Panel construction

The canonical institution-year file is constructed in two stages:

1. When an institution has multiple source policy documents in the same year, all scored sentences from those documents are pooled into one directly observed institution-year policy record.
2. Each observed policy record is treated as the policy in force until the next observed revision for that institution. The final panel retains both `Source_Year` and `Is_Carried_Forward`, so directly observed and inherited annual values can always be distinguished.

The full canonical panel is stored as `data/derived/policy_level_indices_institution_year.csv`. The script `pipeline/04_build_primary_analysis_panel.py` then restricts that file to 1944--2025 and asserts the manuscript sample counts before writing `data/derived/policy_level_indices_primary_1944_2025.csv`.

The 4,277 primary panel rows are therefore **policy-in-force institution-years, not 4,277 distinct policy documents**.

## How to run

From the package root:

```bash
python pipeline/run_all.py
```

The complete run:

1. rebuilds the sentence input corpus;
2. scores sentences with the preserved BERT classifier and temperature calibration;
3. reconstructs the full observed-policy and annual policy-in-force panel;
4. builds and validates the 1944--2025 primary manuscript panel;
5. regenerates the core manuscript tables from that primary panel;
6. regenerates the strengthened robustness outputs used in the final Results and Appendix; and
7. regenerates the two figure filenames referenced directly by the final manuscript.

Temporary run files are written to `data/intermediate/`.

If the canonical scored sentences and full institution-year panel already exist and only the final-paper robustness outputs are needed, run:

```bash
python paper_outputs/code/generate_strengthened_results.py \
  --panel-file data/derived/policy_level_indices_primary_1944_2025.csv \
  --sentences-file data/derived/sentence_scores_canonical.csv
```

By default, strengthened outputs are written to:

`paper_outputs/tables/strengthened_results/`

To regenerate only the two figures referenced by the final LaTeX manuscript, run:

```bash
python paper_outputs/code/generate_final_manuscript_figures.py \
  --panel-file data/derived/policy_level_indices_primary_1944_2025.csv
```

This writes:
- `paper_outputs/figures/fig1_pci_histogram.pdf`
- `paper_outputs/figures/fig1_pci_histogram.png`
- `paper_outputs/figures/fig3_temporal_trend_pcsi.pdf`
- `paper_outputs/figures/fig3_temporal_trend_pcsi.png`

## Core derived files

### Sentence-level scores
File: `data/derived/sentence_scores_canonical.csv`

Important columns:
- `P_restrictive_calib`
- `P_neutral_calib`
- `P_supportive_calib`
- `Pred_Label`
- `SRN_score = P_supportive_calib - P_restrictive_calib`
- `ToneScore_0_1 = P_supportive_calib + 0.5 * P_neutral_calib`

`ToneScore_0_1` is the sentence-level quantity aggregated into the Policy Communication Stance Index (PCSI).

### Full canonical institution-year panel
File: `data/derived/policy_level_indices_institution_year.csv`

This file preserves the full archive, including the isolated 1925 Caltech provenance record and its carry-forward years.

### Primary manuscript panel
Generated file: `data/derived/policy_level_indices_primary_1944_2025.csv`

This is the 1944--2025 panel routed into the manuscript-output code. Its construction is checked against the final-paper sample counts before analysis proceeds.

Panel structure:
- `Institution`
- `Year`
- `Source_Year`
- `Is_Carried_Forward`

Main index variables:
- `Mean_Tone_Score`: baseline PCSI
- `Median_Tone_Score`: median sentence-score aggregation
- `Tone_Index`
- `Clarity_Index`
- `Legal_Load_Index`

Additional fields include document length, lexical component measures, class shares, and institutional metadata used in descriptive analyses.

## Strengthened final-paper analyses

`generate_strengthened_results.py` reproduces the additional exercises used to strengthen the final manuscript. The script deliberately starts from the canonical sentence scores and final panel so that the robustness checks remain downstream of the same measurement pipeline.

It produces the following output families.

### Distribution and sample checks
- `distribution_samples.csv`

Reports the baseline 4,277 policy-in-force observations and the 480 directly observed policy records separately.

### Alternative aggregation checks
- `pcsi_aggregation_pearson_4277.csv`
- `pcsi_aggregation_spearman_4277.csv`
- `pcsi_aggregation_pearson_direct480.csv`
- `pcsi_aggregation_spearman_direct480.csv`
- `pcsi_aggregation_summary_4277.csv`
- `pcsi_aggregation_agreement_4277.csv`

These reconstruct median, 5% trimmed-mean, and sentence-length-weighted PCSI versions directly from sentence-level calibrated scores and then apply the canonical carry-forward structure.

### Persistence and revision dynamics
- `variance_decomposition.csv`
- `revision_changes.csv`
- `revision_change_summary.csv`

The variance decomposition reports between- and within-university shares for the baseline and alternative aggregations, both in the policy-in-force panel and among directly observed records. Revision files compare consecutive observed policies within universities. The compact summary is stored in the repository; the row-level `revision_changes.csv` is regenerated by the script on each run.

### Institutional robustness
- `institutional_aggregation_robustness.csv`
- `institutional_observed_only.csv`

These verify that the principal descriptive institutional differences are not artifacts of the baseline sentence aggregation or long-lasting policy versions.

### Linguistic interpretation and length sensitivity
- `transparent_index_correlations.csv`
- `linguistic_length_sensitivity.csv`

These report relationships among Tone, Clarity, and Legal Load and reproduce the manuscript regressions adding log policy word count or log sentence count with university-clustered standard errors.

### Temporal robustness
- `temporal_common_cohorts.csv`
- `temporal_observed_only_trends.csv`
- `observed_policy_decade_means.csv`
- `temporal_aggregation_robustness.csv`
- `year2025_freshness.csv`

These hold university composition fixed across multiple cohort starts, estimate trends using directly observed policies only, test alternative PCSI aggregations within the fixed 2000 cohort, and verify that the 2025 cross-section is not driven by unusually old carried-forward policies.

## Revision analyses (October 2026)

`paper_outputs/code/generate_revision_analyses.py` adds the exercises used in the October 2026 manuscript. It runs last in `pipeline/run_all.py` and can be run alone from the package root:

```bash
python paper_outputs/code/generate_revision_analyses.py
```

Outputs in `paper_outputs/tables/revision_analyses/`:
- `table5_institutional_robustness.csv`: institutional correlates under the university-mean specifications, Column (1) on the common 144-university sample, WLS (weights = source policy years; = sentences), era-adjusted university means (panel demeaned by calendar year), and source-text university-years with source-year fixed effects and university-clustered SEs.
- `archive_vintage_by_type.csv`: mean source-policy year by institutional type.
- `table6_shapley_r2.csv`: Shapley (dominance-analysis) shares of R^2 for Tone, Clarity and Legal Load, which do not depend on the order of entry.
- `topic_coverage.csv`, `topic_within_trends.csv`, `topic_shift_share.csv`: keyword-based clause topics, within-topic university-FE trends on source-text years from 1980, and a within-topic vs. composition shift-share for the 63 universities with source text in both 1980--1999 and 2010--2025 (universities weighted equally).
- `deontic_coverage.csv`, `deontic_summary.csv`, `deontic_form_regression.txt`, `deontic_shift_share.csv`: sentences coded by deontic operator (obligation, prohibition, permission, none) and sanction language following the Institutional Grammar (Crawford and Ostrom 1995); stance by rule form; PCSI net of rule form; shift-share over rule forms.
- `alternative_constructions.csv`, `classifier_confidence_by_decade.csv`: headline results under hard-label PCSI, high-confidence-only PCSI, and framing net of rule form; model confidence by decade.

Figure: `paper_outputs/figures/fig_topic_trends.pdf`.

Key benchmarks:
- Private R1 vs. public R1, source-text years with year FE, full controls and region FE: about `0.040` (clustered SE about `0.012`); land-grant: about `-0.005` (not significant).
- Shapley share of explained R^2 attributable to Tone: about `0.89` (panel) and `0.78` (source-text years).
- Within-university trend, ownership/assignment sentences: about `-0.0003` per year (not significant); all sentences: about `-0.00117`.
- Exact shift-share, 1980--1999 to 2010--2025 (63 universities, decomposed university by university then averaged): observed change `-0.0288`; within topics `-0.0222` (77%), topic composition `-0.0066` (23%); within rule forms `-0.0271` (94%), rule-form composition `-0.0017` (6%). Per-university contributions are in `topic_shift_share_by_university.csv` and `deontic_shift_share_by_university.csv`.
- Rule form (deontic x sanction cells) explains about `4.8%` of sentence-level variance in stance; PCSI net of rule form correlates about `0.94` with baseline PCSI.

### Independent audit of manuscript numbers

`paper_outputs/code/audit_manuscript_numbers.py` recomputes 39 numbers reported in the manuscript (sample counts, distribution, variance shares, Shapley shares, rule-form statistics, within-topic trends, both shift-share decompositions, the within-year private-R1 coefficient, and the alternative constructions) with code written separately from the analysis scripts, and compares each with the reported value. It writes `paper_outputs/tables/revision_analyses/audit_report.csv` and exits with an error if any check fails.

### Verifying classifier inference

The model weights are stored with Git LFS (`git lfs pull` after cloning). `paper_outputs/code/verify_inference_reproducibility.py --n 2000` rescores a random sample of sentences with the preserved model and temperature and compares the calibrated probabilities and labels with `sentence_scores_canonical.csv`; results are written to `paper_outputs/tables/revision_analyses/inference_reproducibility.csv`. A full rescore of all 87,160 sentences (`pipeline/run_all.py`) takes several hours on a CPU.

### Inter-rater reliability

`paper_outputs/code/compute_interrater_reliability.py` draws a fresh random sample of corpus sentences for double-coding and scores the completed sheet:

```bash
python paper_outputs/code/compute_interrater_reliability.py draw --n 400 --seed 20261019   # writes data/annotation/irr_coding_sheet.csv
# two coders fill coder1 and coder2 independently; adjudicate disagreements in "adjudicated"
python paper_outputs/code/compute_interrater_reliability.py score
```

Scoring writes `paper_outputs/tables/revision_analyses/interrater_reliability.csv` (raw agreement, Cohen's kappa, class-specific agreement, and classifier agreement and macro-F1 against the adjudicated labels) and `manuscript/irr_numbers.tex`, which the manuscript reads automatically in place of its red placeholders.

## Selected manuscript benchmarks reproduced by the strengthened script

The code contains explicit sample assertions for the final manuscript sample:
- 4,277 policy-in-force institution-year observations;
- 480 directly observed institution-year records; and
- 150 universities.

Key benchmark results include:
- primary mean PCSI: about `0.437`;
- directly observed mean PCSI: about `0.429`;
- policy-in-force between-university variance share: about `0.610`;
- directly observed between-university variance share: about `0.487`;
- 330 observed policy-to-policy revision transitions;
- median absolute revision change: about `0.021`;
- Tone--Legal Load correlation: about `0.029`;
- fixed-2000-cohort PCSI trend: about `-0.00112` per year; and
- directly observed-policy university-FE trend from 1980 onward: about `-0.00117` per year.

These benchmarks are reproducibility checks, not hard-coded inputs to the analysis.

## Model and inference provenance

The released classifier is a fine-tuned `bert-base-uncased` three-class sentence classifier for restrictive, neutral, and supportive communicative stance.

Preserved model statistics used in the paper:
- training sentences: `2,458`
- validation sentences: `274`
- total preserved labeled sample: `2,732`
- validation accuracy: about `0.901`
- validation macro-F1: about `0.894`
- temperature scaling parameter: `0.8462`

The package reproduces inference from the saved classifier, calibration, aggregation, panel construction, and manuscript-facing analyses. It does **not** currently reproduce the original model fine-tuning from coder-level pre-adjudication records. For this reason, the manuscript does not report historical inter-rater-reliability statistics that cannot be reconstructed from the preserved files.

## Interpretation

PCSI measures the communicative stance encoded in formal university IP policy text. It is not a measure of:
- the substantive generosity of IP rules;
- legal enforceability;
- university quality;
- faculty perceptions; or
- commercialization performance.

Commercialization outcomes are intentionally kept outside the construction of PCSI so they can be studied subsequently as outcomes rather than embedded into the measure itself.

## Environment

See `requirements.txt` for the Python dependencies. The main pipeline uses pandas, NumPy, SciPy, statsmodels, matplotlib/seaborn for existing paper-output generation, and Transformers/PyTorch for classifier inference.

## Large model files

The fine-tuned model weights are large (`model.safetensors` is approximately 438 MB). If inference-ready weights are distributed through GitHub, Git LFS is required; otherwise the weights should be distributed through a separate archival location.
