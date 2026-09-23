# Replication Package — Communication as Governance

This package reproduces the policy-index construction pipeline from raw policy sentences to sentence-level scores to the final institution-year file used in the manuscript.

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

### Paper outputs
- code: `paper_outputs/code/generate_paper_outputs.py`
- figures: `paper_outputs/figures/`
- tables: `paper_outputs/tables/`
- appendix files: `paper_outputs/appendix/`

`generate_paper_outputs.py` produces the figures, tables, and appendix files used in the manuscript from the final institution-year file.

## Current corpus counts and primary analysis window

The replication package preserves the full archive, including the isolated one-sentence
Caltech fragment from 1925. The manuscript's primary analytical sample begins in 1944
because carrying that fragment forward through 1943 would give a single sentence
disproportionate weight in the early panel.

### Full archive
- raw sentence rows: `87,160`
- canonical scored sentence rows: `87,160`
- institutions: `150`
- observed source institution-years: `481`
- source policy documents: `519`
- policy-in-force institution-year rows through 2025: `4,296`

### Primary manuscript sample (1944--2025)
- sentence rows: `87,159`
- institutions: `150`
- directly observed institution-year policy records: `480`
- source policy documents: `518`
- policy-in-force institution-year observations: `4,277`

Unless otherwise noted, the manuscript's main tables, figures, and reported summary
statistics use the 1944--2025 primary analytical sample. The full-archive version is retained
for transparency and robustness checks.

The manuscript's **primary analysis window is 1944–2025**. The only source record before 1944 is a one-sentence Caltech fragment from 1925. Because the policy-in-force procedure would otherwise carry that single sentence forward through 1943, the manuscript begins in 1944, the first year with a substantive multi-sentence policy record. The raw and derived files retain the 1925 observation for auditability and robustness checks.

Within the primary window there are `87,159` sentence rows, `518` source policy documents, `480` directly observed institution-years, `150` institutions, and `4,277` policy-in-force institution-year observations.

The final file is constructed in two stages:

1. If an institution has more than one source document in the same year, all scored sentences from those documents are pooled to form a single observed institution-year policy record.
2. Starting from each institution's first observed policy year, that record is treated as the active policy for each following year until a newer policy document appears. When a revision or update is observed, the score changes from that year onward.

## How to run

From the package root:

```powershell
python pipeline/run_all.py
```

This rebuilds:

- `data/derived/sentence_scores_canonical.csv`
- `data/derived/policy_level_indices_institution_year.csv`
- all manuscript-facing figures and tables under `paper_outputs/`

Temporary run files are written to `data/intermediate/` during execution.

## What each file contains

### 1. Raw sentences
File:
- `data/raw/policy_sentences_cleaned_combined.csv`

Each row is one cleaned sentence from a source IP policy document.

Key columns:
- `Institution`: institution name used in the replication package
- `Year`: source policy year used for the document
- `File`: source text filename
- `Sentence_Number`: sentence order within the source file
- `Sentence_Cleaned`: cleaned sentence text used for inference
- `Cleaned_Char_Count`: character count of the cleaned sentence
- `STATE`, `Private`, `Carnegie R1`, `MEDSCHOOL`, `Urbanicity (cat)`, `Land-Grant Institution`, `Stem program`: institution descriptors carried through to the final aggregation
- `Type`: derived institutional type (`Private R1`, `Public R1`, `Private R2`, `Public R2`)
- `Med`: indicator shown as `Y/N` for medical school status
- `LG`: indicator shown as `Y/N` for land-grant status

### 2. Sentence-level scores
File:
- `data/derived/sentence_scores_canonical.csv`

This is the sentence-level inference output from the trained BERT classifier.

Key score columns:
- `P_restrictive_calib`: calibrated probability that the sentence is restrictive
- `P_neutral_calib`: calibrated probability that the sentence is neutral
- `P_supportive_calib`: calibrated probability that the sentence is supportive
- `Pred_Label`: highest-probability predicted class
- `SRN_score`: continuous stance score defined as `P_supportive_calib - P_restrictive_calib`
- `ToneScore_0_1`: probability-aggregation score defined as `P_supportive_calib + 0.5 * P_neutral_calib`

Interpretation:
- `ToneScore_0_1 = 0` corresponds to fully restrictive language
- `ToneScore_0_1 = 0.5` corresponds to neutral language
- `ToneScore_0_1 = 1` corresponds to fully supportive language

### 3. Final institution-year indices
File:
- `data/derived/policy_level_indices_institution_year.csv`

This is the final analysis file used for the manuscript. It extends each observed policy record forward year by year until the next observed revision.

Panel structure columns:
- `Institution_Year_Key`: unique institution-year key
- `Year`: panel year
- `Source_Year`: year of the observed policy document from which the current row is inherited
- `Is_Carried_Forward`: indicator equal to `1` if the row is carried forward from the most recent observed policy document and `0` if the row corresponds to an observed policy year

Core index columns:
- `Mean_Tone_Score`: mean of `ToneScore_0_1` across all sentences in the institution-year
- `Median_Tone_Score`: median of `ToneScore_0_1` across all sentences in the institution-year
- `Tone_Index`: lexicon-based tone index built from supportive, restrictive, sanction, pronoun, and modal language features
- `Clarity_Index`: lexicon/structure-based clarity index built from sentence length, long-sentence share, and procedural cues
- `Legal_Load_Index`: lexicon-based legal density index built from legalese and IP technical terminology

Supporting aggregation columns:
- `n_sentences`: total scored sentences in the institution-year
- `n_words`: total words across scored sentences
- `supportive_share`, `neutral_share`, `restrictive_share`: share of sentences assigned to each predicted class
- `supportive_per_1000w`, `restrictive_per_1000w`, `sanction_per_1000w`: normalized lexical rates
- `second_person_per_1000w`, `inclusive_we_per_1000w`: normalized interpersonal-language rates
- `obligation_modal_share`: obligation-modality share relative to permission plus obligation modals
- `mean_sentence_length`, `long_sentence_share`, `procedural_share`: sentence-structure features used in the clarity index
- `legalese_per_1000w`, `iptech_per_1000w`: normalized legal and technical language rates
- `madey2002_per_1000w`, `roche2011_per_1000w`: normalized case-reference term rates

Institution descriptors in the final file:
- `STATE`
- `Private`
- `Carnegie R1`
- `MEDSCHOOL`
- `Urbanicity (cat)`
- `Land-Grant Institution`
- `Stem program`
- `Type`
- `Med`
- `LG`

## Model and inference

The released NLP model is already trained. The model is the scaling device used to construct the manuscript's Policy Communication Stance Index (PCSI). The current saved evaluation metadata report validation accuracy of `0.9015` and macro-F1 of `0.8937` (`n_train = 2,458`, `n_val = 274`). The public package reproduces inference and aggregation from the saved model, but it does not currently reproduce model training from scratch because the coder-level training file is not included.

- `model/srn_cls_model/model.safetensors`

Inference uses that trained classifier together with the fixed temperature-scaling value stored in:

- `model/srn_cls_model/temperature_scaling.json`

This package reproduces:

- sentence-level scoring
- observed institution-year aggregation
- year-by-year panel construction through 2025
- sub-index construction
- manuscript figures and tables

It does not retrain the model from scratch.

## Method notes for the paper

- GPU is an implementation detail, not a measurement choice. It changes runtime, not the construction of the index.
- The paper should describe the classifier as a sentence-level, three-class BERT model with post-hoc temperature scaling and probability aggregation.
- The paper should distinguish clearly between:
  - the main BERT-based `Mean_Tone_Score`
  - the lexicon-based `Tone_Index`
  - `Clarity_Index`
  - `Legal_Load_Index`

## Literature support

- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- Chalkidis et al. (2020), *LEGAL-BERT: The Muppets straight out of Law School*.

## GitHub note

The current fine-tuned model weights are too large for a normal GitHub commit:

- `model/srn_cls_model/model.safetensors` is about `438 MB`

If the repository needs inference-ready weights, use Git LFS. Otherwise, exclude the large weight files and provide them separately.
