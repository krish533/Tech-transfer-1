# Fine-Tuned BERT Policy Tone Pipeline

This folder contains a cleaned, GitHub-ready version of the sentence scoring and policy-index construction pipeline used for the university IP policy project.

## What this pipeline does

1. Score each cleaned policy sentence with a fine-tuned three-class BERT classifier:
   - `restrictive`
   - `neutral`
   - `supportive`
2. Apply temperature scaling to obtain calibrated class probabilities.
3. Convert calibrated probabilities into a continuous sentence-level tone score.
4. Aggregate sentence scores to the institution-year policy level.
5. Construct additional lexicon-based subindices:
   - `Tone_Index`
   - `Clarity_Index`
   - `Legal_Load_Index`

## Core construction

Let a sentence-level calibrated probability vector be:

`(P_restrictive, P_neutral, P_supportive)`

The main 0-1 tone score is:

`ToneScore_0_1 = 0 * P_restrictive + 0.5 * P_neutral + 1 * P_supportive`

Because probabilities sum to one, the alternative signed score used in some intermediate files is:

`SRN_score = P_supportive - P_restrictive = 2 * ToneScore_0_1 - 1`

These two sentence scores are affine transformations of each other. They preserve ranking, but `ToneScore_0_1` is easier to interpret:

- `0` = fully restrictive
- `0.5` = neutral midpoint
- `1` = fully supportive

At the policy level, the main index is the mean across all sentences in an institution-year policy:

`Mean_Tone_Score = mean(ToneScore_0_1)`

The pipeline also outputs `Median_Tone_Score`.

## Files

- `scripts/score_sentences.py`
  Sentence-level scoring with calibrated probabilities and GPU/CPU inference.
- `scripts/build_policy_indices.py`
  Policy-level aggregation and construction of tone, clarity, and legal-load subindices.
- `scripts/forward_fill_merge.py`
  Forward-fill merge of policy indices into a university-year panel.
- `METHODS_ALIGNMENT.md`
  Notes on what the current model artifacts support, and where draft paper text should be revised.
- `requirements.txt`
  Minimal Python dependencies.
- `.gitignore`
  Excludes temporary outputs and oversized model artifacts by default.

## Expected inputs

Sentence scoring expects a table with at least:

- `Institution`
- `Year`
- `Sentence_Cleaned`

Calibration expects a labeled table with:

- a sentence text column
- a label column containing `restrictive`, `neutral`, or `supportive`

## Example workflow

### 1. Score cleaned sentences

```bash
python scripts/score_sentences.py \
  --model-dir ../Fine-tuned\ BERT/srn_cls_model \
  --calib-file ../../path/to/labeled_sentences.csv \
  --calib-text-col Sentence \
  --calib-label-col Label_Text \
  --score-file ../151_Institutions.csv \
  --score-text-col Sentence_Cleaned \
  --out-file ../151_Institutions_srn_scored.csv \
  --device cuda
```

### 2. Build policy-level indices

```bash
python scripts/build_policy_indices.py \
  --scored-file ../151_Institutions_srn_scored.csv \
  --out-file ../policy_level_subindices_rebuilt.csv
```

### 3. Forward-fill merge into a panel

```bash
python scripts/forward_fill_merge.py \
  --policy-file ../policy_level_subindices_rebuilt.csv \
  --panel-file ../../UTT/merged_autm_tone_data_cleaned_submission_ready.csv \
  --out-file ../../UTT/merged_autm_tone_data_with_policy_indices.csv \
  --institution-col institution \
  --year-col year
```

## Method notes for the paper

- GPU is an implementation detail, not a measurement choice. It changes runtime, not the construction of the index.
- The paper should describe the classifier as a sentence-level, three-class BERT model with post-hoc temperature scaling and probability aggregation.
- The paper should distinguish clearly between:
  - the main BERT-based `Mean_Tone_Score`
  - the lexicon-based `Tone_Index`
  - `Clarity_Index`
  - `Legal_Load_Index`

## Literature support

- Devlin et al. (2019), BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Guo et al. (2017), On Calibration of Modern Neural Networks.
- Chalkidis et al. (2020), LEGAL-BERT: The Muppets straight out of Law School.

## GitHub upload note

The current fine-tuned model weight file is too large for a normal GitHub commit:

- `srn_cls_model/model.safetensors` is about 438 MB.

If you want the repository to include inference-ready weights, use Git LFS. Otherwise, exclude the large weight files from GitHub and provide a download link or archive separately.
