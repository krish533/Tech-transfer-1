# Double-coding sheet for inter-rater reliability

`irr_coding_sheet.csv` holds 400 sentences drawn at random (seed 20261019) from the 1944–2025 PCSI corpus by
`paper_outputs/code/compute_interrater_reliability.py draw`. Classifier predictions are deliberately not included.

## Current status

- `coder2` is complete. It was filled by an AI coder (Claude, Anthropic) on 20 October 2026, using only the rubric below and the sentence text, without access to the classifier's predictions. The labels were committed before any comparison with model output.
- `coder1` must be completed by a trained human coder, working without looking at `coder2`.
- The manuscript reports the resulting statistics as agreement between a human coder and an AI coder, not between two human coders. For two-human reliability, add a second human coder's labels in a new column and score those two columns instead.

## Instructions for coders

The human coder works independently: hide or ignore the `coder2` column until `coder1` is complete.

1. Read the full sentence and record its **dominant** communicative stance in your column (`coder1` or `coder2`), using exactly one of:
   - `restrictive`: mandatory assignment, categorical obligations, unilateral institutional authority, prohibitions, sanctions, forfeiture, or compliance demands.
   - `supportive`: assistance, guidance, encouragement, collaboration, shared interests, or facilitation.
   - `neutral`: primarily definitional, descriptive, or procedural, with no clear supportive or restrictive orientation.
2. Code how the sentence is framed, not whether the underlying rule is generous to inventors. A mandatory rule stated as part of a guided, assisted process can be supportive; a simple sentence asserting institutional control is restrictive.
3. Sentences that are not about IP governance (headings, unrelated policy text) are coded `neutral`.
4. After both columns are complete, the coders (or a third person) resolve each disagreement and record the result in `adjudicated`. Leave `adjudicated` blank where the coders agree.

## Scoring

From the package root:

```bash
python paper_outputs/code/compute_interrater_reliability.py score
```

This writes `paper_outputs/tables/revision_analyses/interrater_reliability.csv` and `manuscript/irr_numbers.tex`.
Recompile the manuscript and the reliability numbers replace the red placeholders.
