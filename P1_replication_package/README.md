# Replication Package: Communication as Governance

This package reproduces the measurement pipeline and manuscript analyses for **Communication as Governance: Measuring Communicative Stance in University Intellectual Property Policies**.

The manuscript's primary analytical window is **1944--2025**. The underlying archive retains an isolated one-sentence Caltech record from 1925 for provenance and robustness checks, but the main paper does not use its 1925--1943 carry-forward observations.

## Package structure

### Core data chain
- `data/raw/policy_sentences_cleaned_combined.csv`
- `data/derived/sentence_scores_canonical.csv`
- `data/derived/policy_level_indices_institution_year.csv`

### Model
- `model/srn_cls_model/`

### Replication pipeline
- `pipeline/01_build_input_corpus.py`
- `pipeline/02_score_sentences.py`
- `pipeline/03_build_policy_level_indices.py`
- `pipeline/run_all.py`

### Manuscript outputs
- core output code: `paper_outputs/code/generate_paper_outputs.py`
- strengthened results/robustness code: `paper_outputs/code/generate_strengthened_results.py`
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

The final institution-year file is constructed in two stages:

1. When an institution has multiple source policy documents in the same year, all scored sentences from those documents are pooled into one directly observed institution-year policy record.
2. Each observed policy record is treated as the policy in force until the next observed revision for that institution. The final panel retains both `Source_Year` and `Is_Carried_Forward`, so directly observed and inherited annual values can always be distinguished.

The 4,277 primary panel rows are therefore **policy-in-force institution-years, not 4,277 distinct policy documents**.

## How to run

From the package root:

```bash
python pipeline/run_all.py
```

The complete run:

1. rebuilds the sentence input corpus;
2. scores sentences with the preserved BERT classifier and temperature calibration;
3. reconstructs observed policy-level indices and the annual policy-in-force panel;
4. regenerates the core manuscript tables and figures; and
5. regenerates the strengthened robustness outputs used in the final Results and Appendix.

Temporary run files are written to `data/intermediate/`.

If the canonical scored sentences and institution-year panel already exist and only the final-paper robustness outputs are needed, run:

```bash
python paper_outputs/code/generate_strengthened_results.py
```

By default, those outputs are written to:

`paper_outputs/tables/strengthened_results/`

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

### Final institution-year panel
File: `data/derived/policy_level_indices_institution_year.csv`

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

The variance decomposition reports between- and within-university shares for the baseline and alternative aggregations, both in the policy-in-force panel and among directly observed records. Revision files compare consecutive observed policies within universities.

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

## Selected manuscript benchmarks reproduced by the strengthened script

The script contains explicit sample assertions for the final manuscript sample:
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

These benchmarks are intended as reproducibility checks, not additional hard-coded inputs to the analysis.

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
