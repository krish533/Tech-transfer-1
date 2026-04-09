# Manuscript Data Audit

This note keeps the Paper 1 manuscript aligned with the reviewer-facing replication package.

## Canonical NLP-first objects

The reviewer package supports three core objects:

1. `data/raw/policy_sentences_cleaned_combined.csv`
2. `data/derived/sentence_scores_canonical.csv`
3. `data/derived/policy_level_indices_institution_year.csv`

These correspond to the clean replication chain:

- raw cleaned policy sentences
- sentence-level BERT scores
- institution-year policy indices

## Canonical counts

The current canonical counts are:

- raw sentence rows: `87,326`
- canonical deduplicated sentence rows: `86,874`
- verified institutions: `149`
- institution-year observations: `480`
- observed policy documents: `518`

These are the counts the manuscript should use when describing the NLP corpus and the institution-year policy-index object.

## Unit of analysis

Paper 1 should use:

- `Institution-Year (480)`

If an institution has multiple policy documents in the same year, those documents are aggregated together before the final institution-year scores are computed.

## Important distinction

The manuscript should avoid mixing counts from different downstream project files.

In particular, the sentence-level NLP corpus, the document-level object, and the institution-year aggregate are **not** interchangeable. For the replication package prepared here, the correct Paper 1 policy-index object is:

- `policy_level_indices_institution_year.csv`

## Metadata QC

The canonical corpus includes a metadata QC pass before aggregation. The released raw sentence file is already the corrected version. One important example is:

- `Virginia Commonwealth University: 2027 -> 2021`

This is why the canonical year range is:

- `1925–2025`

not `1925–2027`.
