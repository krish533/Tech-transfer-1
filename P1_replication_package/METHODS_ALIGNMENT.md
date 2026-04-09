# Methods Alignment Note

This note clarifies how the Paper 1 replication package maps to the manuscript.

## 1. What the sentence-level NLP model does

The released NLP model is a **sentence-level, three-class BERT classifier**. Each sentence is classified into:

- restrictive
- neutral
- supportive

The model outputs calibrated class probabilities and then maps them into a continuous sentence-level tone measure:

`ToneScore_0_1 = P_supportive + 0.5 * P_neutral`

This is the basis of the main BERT-based policy tone measure used in the package.

## 2. Temperature scaling

The classifier uses **post-hoc temperature scaling** to improve probability calibration. This is an implementation step in the inference pipeline, not a change in the substantive meaning of the policy index.

The released package stores the fitted temperature value in:

- `model/srn_cls_model/temperature_scaling.json`

## 3. GPU note

GPU use is an **implementation detail**, not a measurement choice.

- GPU changes runtime
- GPU does **not** change the construction of the index
- the sentence scores and final institution-year indices are defined by the trained model weights and the fixed temperature value

## 4. Distinguish the BERT score from the lexicon-based indices

The paper should distinguish clearly between:

### Main BERT-based measure
- `Mean_Tone_Score`

This is the primary sentence-score-based policy tone measure. It is produced by:

1. scoring each sentence with the trained BERT classifier
2. converting calibrated probabilities into `ToneScore_0_1`
3. aggregating sentence-level scores to the institution-year level

### Lexicon- and feature-based companion indices
- `Tone_Index`
- `Clarity_Index`
- `Legal_Load_Index`

These are **not** direct BERT class outputs. They are constructed from linguistic features extracted from the same scored sentence corpus.

## 5. Recommended manuscript wording

The manuscript should describe the classifier as:

- a sentence-level, three-class BERT model
- with post-hoc temperature scaling
- followed by probability aggregation to the institution-year level

The manuscript should also avoid presenting GPU use as part of the measurement design.

## 6. GitHub note

The current fine-tuned model file is large:

- `srn_cls_model/model.safetensors` is about `438 MB`

If the repository is intended to include inference-ready weights, Git LFS should be used. Otherwise, the weight file should be excluded from normal GitHub commits and provided separately.
