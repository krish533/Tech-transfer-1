# Paper 1 Replication Package

This folder is the replication package for Paper 1. It is organized around the three core data layers used in the paper:

1. `data/raw/policy_sentences_cleaned_combined.csv`
2. `data/derived/sentence_scores_canonical.csv`
3. `data/derived/policy_level_indices_institution_year.csv`

The package also includes:

4. `data/derived/institution_metadata.csv`
5. manuscript-facing figures and tables under `outputs/`

The sentence-level NLP model is already trained. The annotation/training information is incorporated into the released model weights. To reproduce sentence-level inference exactly, this package uses the released fine-tuned model together with the fixed temperature-scaling value stored in:

- `model/srn_cls_model/temperature_scaling.json`

This package reproduces the inference-and-aggregation pipeline from cleaned policy sentences to final institution-year indices. It does not retrain the model from scratch.

## What is included

### Raw data
- `data/raw/policy_sentences_cleaned_combined.csv`

This is the merged cleaned sentence corpus created from the `120` and `32` institution files. It is already the metadata-QC corrected raw sentence file used for replication.

### Model
- `model/srn_cls_model/`

This contains the fine-tuned BERT classifier used for sentence scoring. The model is already trained; the released weights are the trained model.

### Scripts
- `scripts/01_build_input_corpus.py`
  - copies the released raw sentence file into the rerun workspace
- `scripts/02_score_sentences.py`
  - scores sentences with the released model using a fixed temperature-scaling value
- `scripts/03_build_policy_level_indices.py`
  - removes exact duplicate rows and builds the final institution-year policy indices
- `scripts/run_all.py`
  - runs the full replication pipeline end to end

### Final outputs
- `data/derived/sentence_scores_canonical.csv`
- `data/derived/policy_level_indices_institution_year.csv`
- `data/derived/institution_metadata.csv`

### Paper outputs
- `outputs/figures/`
- `outputs/tables/`
- `outputs/appendix/`

These files are generated directly from the released institution-year indices and the
matched institution metadata companion file. They cover the manuscript figures and
tables that are derived from the policy-level indices and sub-indices.

## Pipeline overview

### Step 1. Start from the cleaned sentence corpus

The package begins from:

- `data/raw/policy_sentences_cleaned_combined.csv`

This file is already the metadata-QC corrected raw sentence corpus used in the paper.

Each row is one cleaned policy sentence with:

- `Institution`
- `Year`
- `File`
- `Sentence_Number`
- `Sentence_Cleaned`

### Step 2. Use the corrected raw sentence corpus

Before release, we corrected known institution/year metadata errors against the verified policy sources. This includes cases such as:

- `2027 -> 2021` for `Virginia Commonwealth University`
- institution-name standardization for merged or previously misnamed records

The corrected raw corpus has:

- `87,326` raw sentence rows
- `149` verified institutions
- `480` institution-year observations

### Step 3. Score every sentence with the fine-tuned BERT model

Each sentence receives calibrated probabilities for:

- restrictive
- neutral
- supportive

The released model outputs:

- `P_restrictive_calib`
- `P_neutral_calib`
- `P_supportive_calib`
- `SRN_score = P_supportive_calib - P_restrictive_calib`
- `ToneScore_0_1 = P_supportive_calib + 0.5 * P_neutral_calib`
- `Pred_Label`

### Step 4. Remove exact duplicate sentence rows

We remove exact duplicates on:

- `Institution`
- `Year`
- `File`
- `Sentence_Number`
- `Sentence_Cleaned`

Current result:

- `87,326` raw rows
- `86,874` canonical sentence rows
- `452` exact duplicate rows removed

### Step 5. Aggregate to institution-year

Paper 1 uses the **institution-year** as the final policy object.

If an institution has multiple policy documents in the same year, we aggregate all of those documents together before constructing the final institution-year scores and sub-indices.

The final institution-year file is:

- `data/derived/policy_level_indices_institution_year.csv`

Current count:

- `480` institution-year observations

### Step 6. Construct the sub-indices

The final institution-year file includes:

- `Mean_Tone_Score`
- `Median_Tone_Score`
- `Tone_Index`
- `Clarity_Index`
- `Legal_Load_Index`

Interpretation:

- `Mean_Tone_Score` and `Median_Tone_Score` are based on the BERT sentence scores
- `Tone_Index`, `Clarity_Index`, and `Legal_Load_Index` are lexicon- and feature-based companion indices computed from the same scored sentence corpus

### Step 7. Add institution metadata for cross-sectional outputs

Some manuscript figures and tables require institution descriptors such as state,
public/private status, Carnegie R1 status, medical-school indicator, urbanicity, and
land-grant status. Those fields are stored in:

- `data/derived/institution_metadata.csv`

This metadata companion file is aligned to the 149 institutions in the released
institution-year index.

### Step 8. Generate manuscript-facing outputs

The package includes:

- `scripts/04_generate_paper_outputs.py`

This script regenerates the figures and tables that are directly derived from the
institution-year indices and sub-indices, including:

- descriptive PCI summary outputs
- cross-sectional heterogeneity outputs
- decomposition outputs
- quintile profile outputs
- temporal trend outputs
- legal-load over-time outputs
- ranking/reference outputs

## Validation

The released inference pipeline was checked against the existing internal scored reference.

Result:

- predicted label match rate = `1.0`
- changed labels = `0`
- any probability differences are only floating-point noise

So the released sentence-scoring pipeline reproduces the current scored outputs for practical purposes.

See:

- `VALIDATION_REPORT.md`
- `MANUSCRIPT_DATA_AUDIT.md`

## How to rerun

From the package root:

```powershell
python scripts/run_all.py
```

This rebuilds:

- `data/derived/sentence_scores_canonical.csv`
- `data/derived/policy_level_indices_institution_year.csv`
- `outputs/figures/*`
- `outputs/tables/*`
- `outputs/appendix/*`

Intermediate rerun files are written under:

- `data/intermediate/`

and are not part of the final replication outputs.
