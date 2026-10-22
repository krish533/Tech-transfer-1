# Paper 1: Table 5 rerun and bounded legal-topic sensitivity

**Data:** `P1_replication_package/data/derived/policy_level_indices_institution_year.csv`, restricted to 1944–2025; 4,277 annual policy-in-force rows; 150 institutions; 480 source-document university-years. Topic sensitivity also uses `data/raw/policy_sentences_cleaned_combined.csv` (87,159 sentences).

## 1. Table 5 reproduces exactly

Re-estimating the original three university-level specifications with HC3 robust SE reproduces the rounded published Table 5 entries (N = 150, 145, 144; R² = .074, .111, .161). The original Private R1 coefficients are .02928 (.00934), .01899 (.01207), .02048 (.01362).

## 2. Common-sample Table 5 (N = 144 for all columns)

| Term | Type only | + Institution / urbanicity | + Region |
|:--|--:|--:|--:|
| Private R1 | 0.02894 (0.00941) | 0.01892 (0.01210) | 0.02048 (0.01362) |
| Private R2 | -0.02086 (0.02071) | -0.03064 (0.02267) | -0.03466 (0.02271) |
| Public R2 | 0.01125 (0.01887) | 0.00751 (0.02007) | 0.01091 (0.01840) |
| Land-grant | — | -0.01350 (0.01248) | -0.01067 (0.01329) |
| Medical school | — | 0.01363 (0.01082) | 0.01043 (0.01056) |
| R² | 0.07189 | 0.11112 | 0.16078 |
| Universities | 144 | 144 | 144 |

Reference categories are Public R1, rural, and Midwest. Changes to the Private R1 point estimate cannot be attributed to switching estimation samples. **Do not claim these are causal coefficients.**

Six institutions fall outside the common sample: five with missing urbanicity (SUNY, Texas A&M University System, University of California System, University of Missouri System, University of Oklahoma) and one with no region mapping in the original six-region scheme (University of Hawaii). Names in this audit are for reproducibility, not manuscript Table 2.

## 3. Sensitivity to weighting and calendar year

| Specification | Private R1 | N | Standard errors |
|:--|--:|--:|:--|
| Common-sample institutional cross-section | 0.02048 (0.01362) | 144 universities | HC3 |
| WLS, weighted by number of *direct source* policy-years | 0.02016 (0.01504) | 144 universities | HC3 |
| Policy-in-force panel with year fixed effects | 0.02504 (0.01322) | 4,065 university-years | University-clustered |
| Direct-source-text panel with year fixed effects | 0.03993 (0.01196) | 452 university-years | University-clustered |

The annual panel has repeated carried-forward scores, so the source-only results avoid that particular duplication but are affected by the timing and availability of archived policies. Year-fixed-effects models assign more weight to institutions with more observed years and are not the same estimand as equal-university cross-sectional OLS. Their R² should not be directly compared with cross-sectional R².

**Interpretation:** The Private R1 association attenuates in the original nested university-level models, but is not uniformly imprecise across specifications. The source-only, calendar-year-adjusted association is larger. Do not interpret the difference between these estimators as evidence of a causal institution-type effect.

## 4. Exploratory topic-composition sensitivity — not within-topic validation

We tagged sentences using five **overlapping** case-insensitive keyword groups: ownership/assignment; disclosure; licensing/patenting/commercialization; royalties/revenue; and administrative terms. For each observed institution-source-year we computed the fraction of sentences flagged by each group, inherited the fractions to the policy-in-force years via Source_Year, and averaged them equally across annual years within each university. Adding five shares to the common-sample full institutional regression increases R² from 0.16078 to 0.23607; the Private R1 coefficient is 0.01953 (HC3 SE 0.01476).

This rough subject-matter proxy does **not** hold substantive legal provisions constant and does **not** constitute a within-topic PCSI regression. The keyword features are neither mutually exclusive nor validated legal-content classifications. In particular, the administrative vocabulary and the word 'assign' may reflect drafting style as well as legal content. A genuine within-topic comparison requires scored sentences joined to expert-validated topic labels and ideally comparable legal obligations. Do not use the exploratory result as independent construct validation.

## Reproducibility

- Script: `P1_replication_package/paper_outputs/code/rerun_table5.py`
- Compact independently computed audit output: `verified_private_r1_results.csv`
- Run the script from the repository root after installing pandas and statsmodels. It writes complete coefficients and the Private R1 comparison CSV in this folder.
- The numerical summary here was re-estimated directly from the checked-in canonical files. The reproducibility script is provided for independent execution; its generated outputs must be checked after running it in an environment with the full corpus.

The rerun did **not** fine-tune or recalibrate BERT, alter existing data files, or establish a causal impact of institutional type.
