# Validation Report

## 1. Sentence-score consistency

The released inference pipeline was checked against the existing internal scored reference file. The comparison key was:

- `Institution`
- `Year`
- `File`
- `Sentence_Number`
- `Sentence_Cleaned`

Result:

- predicted label match rate: `1.0`
- changed labels: `0`

Interpretation:

- the reviewer-facing rerun is effectively identical to the current internal scored sentence file
- any probability differences are only at floating-point noise scale

## 2. Model status and fixed temperature scaling

The released NLP model is already fine-tuned. The annotation/training information is incorporated into the model weights. This package therefore reproduces exact sentence-level inference using:

- the released fine-tuned BERT model
- a fixed temperature-scaling value stored in `model/srn_cls_model/temperature_scaling.json`

Current value:

- `temperature = 0.8461815715`

This preserves exact sentence-level scoring without requiring release of the labeled training/calibration file.

## 3. Canonical counts

After metadata QC:

- raw sentence rows: `87,326`
- institutions: `149`
- institution-years: `480`

The released `data/raw/policy_sentences_cleaned_combined.csv` is already this metadata-QC corrected raw corpus.

After exact-row deduplication:

- canonical sentence rows: `86,874`
- duplicate rows removed: `452`

Observed source documents:

- policy documents keyed by `Institution + Year + File`: `518`

Final Paper 1 analysis object:

- institution-year records: `480`

## 4. Practical implication

This reviewer/public package is sufficient to reproduce:

- sentence-level NLP scores
- institution-year policy tone scores
- institution-year `Tone_Index`
- institution-year `Clarity_Index`
- institution-year `Legal_Load_Index`

It is not designed to reproduce the original model training workflow. That is an intentional packaging choice, not a gap in inference replication.
