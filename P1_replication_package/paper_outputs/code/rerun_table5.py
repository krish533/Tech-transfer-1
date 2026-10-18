"""Rerun Paper 1 institutional regressions from the tracked replication CSVs.

Run from repository root:
  python P1_replication_package/paper_outputs/code/rerun_table5.py

Requires pandas and statsmodels. Does not retrain the classifier or rewrite core data.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/derived/policy_level_indices_institution_year.csv"
SENTENCES = ROOT / "data/raw/policy_sentences_cleaned_combined.csv"
OUT = ROOT / "paper_outputs/tables/table5_rerun"
REGIONS = dict.fromkeys(["ME","NH","VT","MA","RI","CT"], "Northeast")
REGIONS.update(dict.fromkeys(["NY","NJ","PA","DE","MD","DC"], "Mid-Atlantic"))
REGIONS.update(dict.fromkeys(["OH","MI","IN","IL","WI","MN","IA","MO","ND","SD","NE","KS"], "Midwest"))
REGIONS.update(dict.fromkeys(["VA","WV","KY","TN","NC","SC","GA","FL","AL","MS","AR","LA","OK","TX"], "South"))
REGIONS.update(dict.fromkeys(["MT","ID","WY","CO","NM","AZ","UT","NV"], "Southwest"))
REGIONS.update(dict.fromkeys(["WA","OR","CA"], "West Coast"))

TOPIC_PATTERNS = {
    "ownership": r"\b(own(?:ership|ed|er)?|assign(?:ment|ed|able)?|title|vest(?:s|ed|ing)?|proprietary)\b",
    "disclosure": r"\b(disclos(?:ure|e|es|ed|ing)|report(?:s|ing|ed)? an? invention|notify(?:ing)? the (?:university|office))\b",
    "licensing": r"\b(licens(?:e|ing|es|ed)|commercializ(?:ation|e|ing)|patent(?:ing|s)?|market(?:ing|place)?|technology transfer)\b",
    "revenue": r"\b(royalt(?:y|ies)|revenue(?:s)?|income|net proceeds|distribution of proceeds|inventor share|profit shar(?:e|ing))\b",
    "administration": r"\b(committee|appeal|procedure(?:s)?|policy administrator|director|office|review(?:ed|ing)?|approval|administ(?:er|ration|rative))\b",
}

TYPE = 'C(Type, Treatment(reference="Public R1"))'
URBAN = 'C(Urbanicity, Treatment(reference="Rural"))'
REGION = 'C(Region, Treatment(reference="Midwest"))'
BASE = f'pcsi ~ {TYPE}'
MID = BASE + f' + LandGrant + MedicalSchool + {URBAN}'
FULL = MID + f' + {REGION}'

def summarize(model, name: str) -> list[dict]:
    out = []
    for term in model.params.index:
        out.append({
            "model": name, "term": term,
            "coefficient": float(model.params[term]),
            "robust_se": float(model.bse[term]),
            "p_value": float(model.pvalues[term]),
            "n": int(model.nobs), "r_squared": float(model.rsquared),
        })
    return out

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL, low_memory=False).query("1944 <= Year <= 2025").copy()
    assert len(panel) == 4277 and panel.Institution.nunique() == 150
    assert int((panel.Is_Carried_Forward == 0).sum()) == 480

    inst = panel.groupby("Institution", as_index=False).agg(
        pcsi=("Mean_Tone_Score", "mean"), Type=("Type", "first"),
        LandGrant=("Land-Grant Institution", "first"),
        MedicalSchool=("MEDSCHOOL", "first"),
        Urbanicity=("Urbanicity (cat)", "first"),
        STATE=("STATE", "first"),
        n_sources=("Is_Carried_Forward", lambda s: int((s == 0).sum())),
    )
    inst["Urbanicity"] = inst["Urbanicity"].replace(
        {"Urban": "City", "Semi-Urban": "Semi-Rural"}
    )
    inst["Region"] = inst["STATE"].map(REGIONS)
    full_sample = inst.dropna(
        subset=["pcsi", "Type", "LandGrant", "MedicalSchool", "Urbanicity", "Region"]
    ).copy()
    assert len(full_sample) == 144

    results = []
    models = {
        "original_1_n150": smf.ols(BASE, inst).fit(cov_type="HC3"),
        "original_2_n145": smf.ols(MID, inst).fit(cov_type="HC3"),
        "original_3_n144": smf.ols(FULL, inst).fit(cov_type="HC3"),
        "common_1_n144": smf.ols(BASE, full_sample).fit(cov_type="HC3"),
        "common_2_n144": smf.ols(MID, full_sample).fit(cov_type="HC3"),
        "common_3_n144": smf.ols(FULL, full_sample).fit(cov_type="HC3"),
        "coverage_weighted_n144": smf.wls(
            FULL, full_sample, weights=full_sample.n_sources
        ).fit(cov_type="HC3"),
    }

    p = panel[panel.Institution.isin(full_sample.Institution)].copy()
    p["pcsi"] = p["Mean_Tone_Score"]
    p["LandGrant"] = p["Land-Grant Institution"]
    p["MedicalSchool"] = p["MEDSCHOOL"]
    p["Urbanicity"] = p["Urbanicity (cat)"].replace(
        {"Urban": "City", "Semi-Urban": "Semi-Rural"}
    )
    p["Region"] = p.STATE.map(REGIONS)
    models["annual_year_FE_cluster144"] = smf.ols(
        FULL + " + C(Year)", p
    ).fit(cov_type="cluster", cov_kwds={"groups": p.Institution})
    observed = p[p.Is_Carried_Forward == 0].copy()
    assert len(observed) == 452
    models["source_year_FE_cluster144"] = smf.ols(
        FULL + " + C(Year)", observed
    ).fit(cov_type="cluster", cov_kwds={"groups": observed.Institution})

    # Exploratory subject-matter controls, NOT a comparison of identical
    # provisions and NOT a within-topic PCSI estimator.
    raw = pd.read_csv(SENTENCES, low_memory=False).query("1944 <= Year <= 2025").copy()
    assert len(raw) == 87159
    for topic, pat in TOPIC_PATTERNS.items():
        raw["topic_" + topic] = raw.Sentence_Cleaned.fillna("").str.contains(
            pat, case=False, regex=True
        ).astype(float)
    share_cols = ["topic_" + t for t in TOPIC_PATTERNS]
    shares = raw.groupby(["Institution", "Year"], as_index=False)[share_cols].mean()
    shares = shares.rename(columns={"Year": "Source_Year"})
    merged = p.merge(
        shares, on=["Institution", "Source_Year"],
        how="left", validate="many_to_one"
    )
    if merged[share_cols].isna().any().any():
        raise ValueError("One or more policy-in-force records lack topic shares")
    means = merged.groupby("Institution", as_index=False)[share_cols].mean()
    topic_inst = full_sample.merge(means, on="Institution", validate="one_to_one")
    models["exploratory_topic_controls_n144"] = smf.ols(
        FULL + " + " + " + ".join(share_cols), topic_inst
    ).fit(cov_type="HC3")

    for name, model in models.items():
        results.extend(summarize(model, name))
    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUT / "all_model_coefficients.csv", index=False)
    pr1 = results_frame[
        results_frame.term.str.contains("Private R1", regex=False)
    ][["model", "coefficient", "robust_se", "p_value", "n", "r_squared"]]
    pr1.to_csv(OUT / "private_r1_specification_comparison.csv", index=False)
    print(pr1.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("Outputs:", OUT)

if __name__ == "__main__":
    main()
