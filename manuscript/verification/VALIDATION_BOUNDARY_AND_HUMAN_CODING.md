# PCSI classifier and independent human coding: validation boundary

### What passed independently

1. All 87,159 stored calibrated sentence scores recompute as `P_supportive_calib + 0.5 * P_neutral_calib` to floating-point tolerance.
2. Every saved university/source-year aggregation (480 observations) and annual policy-in-force row (4,277 rows, 3,797 carried forward) reconstructs.
3. The 2025 frequency and distribution checks, source-year university fixed-effects slope, full-controls university regression, Shapley R² shares, and exact topic and rule-form shift-share identities pass. See `replication/independent_run/`.
4. The original repository's saved model metadata reports 2,458 training and 274 validation sentences, 0.901459854 accuracy and 0.893716543 macro-F1. These are **historical saved artifacts**, not newly reproduced held-out accuracy estimates.

### What cannot be claimed independently

* The original human coder-level files and reconciliation records were not preserved in the released package. The original adjudicated 2,732-item training set is not independently reconstructible from the scored sentences.
* The GitHub random 400-sentence independent coding sheet `P1_replication_package/data/annotation/irr_coding_sheet.csv` contains blank `coder1`, `coder2`, and `adjudicated` columns in the released template. No new human Cohen's κ, inter-rater agreement, adjudicated classifier accuracy, or macro-F1 can be calculated without **actual independent coding**.
* A single model-generated set of labels, keyword labeling, or two AI-generated 'coders' would not substitute for genuine independent human reliability evidence; such labels must not be reported as measured human agreement.
* We have not repeated the original classifier fine-tuning or re-run held-out evaluation with its original labeled records. Git LFS model weights are a different artifact than the historical labeled evaluation set.

### Reproducible completion procedure when human coding becomes available

1. Download `irr_coding_sheet.csv` from the [PCSI GitHub annotation folder](https://github.com/krish533/Tech-transfer-1/tree/main/P1_replication_package/data/annotation).
2. Two coders independently apply the rubric to all 400 sentences without looking at predictions or each other's labels. Label restrictive, neutral, or supportive.
3. Keep separate, time-stamped original coder columns; adjudicate disagreements in `adjudicated`. Do **not** overwrite coder1 or coder2.
4. Run `python paper_outputs/code/compute_interrater_reliability.py score` from the replication package root. The script produces an observed-agreement, Cohen's κ, and classifier-alignment table and enables `irr_numbers.tex` only if completed.
5. Inspect class-specific counts, disagreement patterns, potentially unrelated text, and confidence intervals; then recompile the manuscript. If annotator disagreement is high or systematic, qualify the validity claims and revise the label rubric before generalizing.

The present manuscript intentionally renders the missing statistics as **not reported**, not as invented zeroes or synthetic numbers.
