# Double-coding sheet for inter-rater reliability

`irr_coding_sheet.csv` holds 400 sentences drawn at random (seed 20261019) from the 1944–2025 PCSI corpus by
`paper_outputs/code/compute_interrater_reliability.py draw`. Classifier predictions are deliberately not included.

## Instructions for coders

Two coders work independently and do not discuss sentences until both have finished.

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
