# Methods Alignment Notes

This note compares the draft paper text against the currently preserved model artifacts and scripts in `UTT/Fine-tuned BERT`.

## What is clearly supported by the current artifacts

- Model family: `bert-base-uncased` sentence classifier with three labels.
- Labels:
  - `restrictive = 0`
  - `neutral = 1`
  - `supportive = 2`
- Maximum sequence length: `256`
- Batch size: `8`
- Epochs: `3`
- Learning rate: `2e-5`
- Weight decay: `0.05`
- Label smoothing factor: `0.10`
- Temperature scaling is used after model fitting.
- Sentence-level probabilities are converted into a continuous tone score and then aggregated to policy level.

These are supported by:

- `UTT/Fine-tuned BERT/srn_cls_model/config.json`
- `UTT/Fine-tuned BERT/srn_cls_model/label_map.json`
- `UTT/Fine-tuned BERT/srn_cls_model/metrics.json`
- `UTT/Fine-tuned BERT/srn_cls_model/ckpts/checkpoint-924/trainer_state.json`
- `UTT/Fine-tuned BERT/calibrate_and_score_srn.py`

## What should be revised in the draft paper

### 1. Draft sample size does not match current saved model metadata

Draft text:

- annotation sample size = `2,847`

Current saved model metadata:

- `n_train = 2458`
- `n_val = 274`
- total implied sample = `2732`

This suggests the current saved model is not the same artifact as the draft-paper numbers, or the draft numbers were not updated.

### 2. Draft validation metrics do not match current saved model metadata

Draft text:

- accuracy = `0.803`
- macro F1 = `0.788`

Current `metrics.json`:

- accuracy = `0.9014598540145985`
- macro F1 = `0.8937165432834986`

Again, the draft and the preserved model artifact appear to reflect different versions.

### 3. Draft says the training can be reproduced with `03_BERT_training.py`

That script is not currently present in `UTT/Fine-tuned BERT`.

What is present:

- `calibrate_and_score_srn.py`
- `P1.py`
- `P1 Regression.py`
- trained model artifacts in `srn_cls_model/`

If the paper says the full training is reproducible from the repository, that claim is too strong unless the original training script and labeled training data are also included.

### 4. Calibration description should be made precise

The current scoring script fits a scalar temperature `T` on a labeled calibration file, but the script default is:

- `--calib-max-samples 500`

So the safest wording is not "we always calibrate on the full validation set" unless that exact run setting is documented. A safer phrasing is:

"We apply post-hoc temperature scaling using a labeled calibration set."

If you want the paper to state "held-out validation set", then the public replication package should use the full held-out validation set explicitly and document it.

### 5. Distinguish the main BERT index from the lexicon subindices

The draft should avoid collapsing these into one construct.

- `Mean_Tone_Score` is the main BERT-based policy communication index.
- `Tone_Index` is a lexicon-based stylistic subindex.
- `Clarity_Index` is a lexicon/readability-style subindex.
- `Legal_Load_Index` is a legal-density subindex.

These are related but conceptually distinct.

## Recommended paper wording

Use wording close to this:

"We fine-tune a three-class BERT sentence classifier to predict whether each policy sentence is restrictive, neutral, or supportive toward faculty inventors. We then apply temperature scaling to calibrate the model's predicted probabilities. For each sentence, we compute a continuous tone score equal to 0 for restrictive language, 0.5 for neutral language, and 1 for supportive language, weighted by the calibrated class probabilities. We aggregate these sentence-level scores to the institution-year policy level using the mean and median across all sentences in the policy."

That description is aligned with the preserved scoring and construction logic.
