# Securing Title: Stanford v. Roche replication package

This directory documents the standalone empirical project examining how U.S. university intellectual-property policies changed around *Board of Trustees of the Leland Stanford Junior University v. Roche Molecular Systems, Inc.*

## Empirical scope

The project uses the university IP-policy corpus developed in the broader Tech-transfer-1 project. The canonical corpus contains sentence-level policy text and an institution-year panel that records the policy in force in each year. Because a policy is carried forward until a subsequent observed revision, the panel distinguishes directly observed policies from carried-forward observations via `Source_Year` and `Is_Carried_Forward`.

The principal legal event is the September 30, 2009 Federal Circuit decision. The shock year is omitted in specifications using that event. The June 6, 2011 Supreme Court decision is used as an alternative timing specification.

## Preferred identification test

The preferred ex ante exposure design uses FY2008 R&D source-of-funds data, measured before the Federal Circuit decision. For each matched university we construct:

- federal share = federally financed R&D / total R&D;
- industry share = industry-financed R&D / total R&D;
- research scale = log total R&D;
- dual exposure = standardized interaction of federal-share and industry-share exposure.

The preferred model interacts each of these predetermined variables with the post-2009 indicator and includes university and year fixed effects. The coefficient on `dual_z_post` tests whether universities simultaneously exposed to federally funded research and industry-funded research changed assignment-related language differentially after the decision, conditional on the two lower-order exposures and research scale.

The FY2008 source file is `data/external/ncses_fy2008_top150_sources.csv`. It reproduces the source-of-funds columns for the first 150 institutions in the FY2008 NSF rankings reported in Appendix D of the University of Massachusetts President's Office FY2009 R&D Expenditures report, whose underlying source is the NSF Survey of Research and Development Expenditures at Universities and Colleges, FY2008. Dollar values are in thousands.

Only manually audited or canonical-exact institution mappings are admitted to the preferred exposure sample. Fuzzy matches are not used.

## Main outcomes

Assignment-language outcomes are generated transparently from policy text:

- `has_strong_assignment`: indicator that the policy contains present/automatic assignment language;
- `strong_per_1000w`: present-assignment terms per 1,000 words;
- `weak_per_1000w`: future-promise assignment terms per 1,000 words;
- `net_assignment_strength = strong_per_1000w - weak_per_1000w`.

The broader Policy Communication Stance Index (`Mean_Tone_Score`), Legal Load Index, and obligation-modal share are included as broader or falsification-style outcomes rather than treated as the primary response to a narrow ownership/assignment ruling.

## Analysis scripts

Run from the `P1_replication_package` directory with Python 3.11:

```bash
python paper_outputs/code/run_stanford_roche_causal_test.py
python paper_outputs/code/run_federal_rd_exposure_analysis.py
python paper_outputs/code/run_federal_rd_exposure_clean.py
python paper_outputs/code/run_stanford_roche_extended.py
python paper_outputs/code/run_nist_2018_assignment_rule.py
python paper_outputs/code/run_chain_title_risk_2008.py
```

The preferred FY2008 analysis is the final command. Earlier commands reproduce the sequence of alternative identification tests discussed in the manuscript and appendix.

## Output directories

- `paper_outputs/tables/stanford_roche_causal/`: pre-2009 policy-language vulnerability design.
- `paper_outputs/tables/federal_rd_exposure/`: exploratory federal-R&D exposure design.
- `paper_outputs/tables/federal_rd_exposure_clean/`: audited FY2010 federal-share design using the 2011 Supreme Court decision.
- `paper_outputs/tables/stanford_roche_extended/`: research-scale, adoption-timing, alternative-exposure and descriptive results.
- `paper_outputs/tables/nist_2018_assignment_rule/`: 2018 Bayh-Dole regulatory amendment analysis.
- `paper_outputs/tables/chain_title_risk_2008/`: preferred pre-2009 federal x industry exposure analysis.

## Interpretation rule

The package is deliberately designed not to select specifications on statistical significance. The manuscript reports the strong descriptive shift in present-assignment language, the suggestive research-scale result, and the null differential-exposure tests. A causal statement is warranted only if the preferred predetermined exposure design, event-study diagnostics, and placebo tests jointly support it. Otherwise the results are interpreted as evidence of temporally aligned institutional adaptation rather than a point-identified causal effect of the court ruling.

## Software

Core packages: `pandas`, `numpy`, `scipy`, `statsmodels`, `openpyxl`, `lxml`, `html5lib`, `beautifulsoup4`.

## Data provenance

1. University IP-policy corpus and derived sentence/policy measures: project repository data.
2. FY2008 R&D source-of-funds table: NSF Survey of Research and Development Expenditures at Universities and Colleges, FY2008, as reproduced in the University of Massachusetts President's Office FY2009 Annual R&D Expenditures Expanded Report, Appendix D.
3. FY2010 and later exposure tables used in sensitivity analyses: official NSF/NCSES data tables downloaded by the analysis scripts.

## Reproducibility note

The policy corpus contains historical documents assembled from public university sources. Redistribution and licensing of the underlying source documents should follow the terms documented by the parent Tech-transfer-1 project. Derived data and analysis code in this package are intended to make the reported results auditable and reproducible.
