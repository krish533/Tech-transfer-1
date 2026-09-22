# WP1 final revision notes

This branch consolidates the revisions discussed for the measurement paper.

## Core empirical decisions

- Primary sample: **150 universities, 1944–2025**.
- The raw archive retains one pre-1944 observation: a **one-sentence Caltech fragment from 1925**. It is excluded from the primary manuscript because the policy-in-force procedure would otherwise carry that single sentence through 1943.
- Primary-window source corpus: **87,159 sentences, 518 source policy documents, 480 directly observed institution-years**.
- Policy-in-force panel: **4,277 institution-year observations**.
- The manuscript now distinguishes source documents, observed institution-year policy records, and carried-forward annual policy-in-force observations.

## Measurement and validation

- Renamed the measure **Policy Communication Stance Index (PCSI)** to distinguish it from Canary, Riforgiate, and Montoya's survey-based Policy Communication Index.
- Uses the metrics attached to the released classifier artifact: accuracy **0.9015**, macro-F1 **0.8937**, train N **2,458**, validation N **274**, temperature **0.8462**.
- Removed the older 0.803/0.788 model results from the manuscript because they belong to an earlier model version.
- Removed unreproducible inter-annotator reliability statistics from the manuscript until coder-level annotation records are archived.
- Uses the actual median of the primary PCSI distribution rather than the older “mean of median-based aggregation” value.
- Linguistic regressions now use **university-clustered standard errors**.

## Research Policy feedback

The paper no longer relies on raw heterogeneity alone. It adds:

- a between/within decomposition: about **61%** of observed panel variation is between universities;
- conditional institution-level regressions showing which bivariate regularities persist after joint controls;
- more cautious interpretation of private/public, land-grant, geography, and legal-event patterns.

The technology-transfer outcome regressions remain outside WP1 and belong to the companion paper.

## Lisa Ouellette feedback

- Corrected the description of Lach and Schankerman (2008) using Ouellette and Tutt (2020).
- Added the organizational-practices and faculty–TTO literature (Siegel et al.; Jensen et al.; Owen-Smith and Powell).
- Explicitly distinguishes the textual construct from faculty awareness or perception.
- Discusses three possible pathways: direct faculty exposure, TTO-mediated implementation, and common-cause organizational culture.
- Strengthens replication language and recommends a public archival release before submission.

## Temporal interpretation

- Removed the claim that Stanford v. Roche caused a downward PCSI shift or increased legal density.
- Legal milestones are treated as institutional context only; the paper makes no causal temporal claim.

## Before journal submission

1. Run:
   `python P1_replication_package/pipeline/run_all.py`
   on this branch to regenerate all figures/tables under the 1944–2025 primary window.
2. Archive coder-level annotation data/provenance if available.
3. Make the replication package public (or deposit it in a persistent archive) and add the permanent URL/DOI to the manuscript.
4. Update the companion outcomes paper to use the PCSI name and corrected Lach–Schankerman/Ouellette–Tutt literature discussion.
