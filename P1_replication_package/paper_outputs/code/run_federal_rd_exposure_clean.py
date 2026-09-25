from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from run_stanford_roche_causal_test import build_language_measures, attach_language_to_panel
from run_federal_rd_exposure_analysis import TOTAL_URL, FED_URL, read_rank_xlsx

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PACKAGE_ROOT / "data" / "derived" / "policy_level_indices_institution_year.csv"
SENTENCE_PATH = PACKAGE_ROOT / "data" / "derived" / "sentence_scores_canonical.csv"
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "federal_rd_exposure_clean"
SHOCK_YEAR = 2011
EXPOSURE_YEAR = 2010

# Manual mappings are used only where the policy-corpus name and NCSES name differ
# mechanically. No fuzzy matching is allowed in the estimation sample.
SINGLE = {
    "Albert Einstein/Yeshiva": "Yeshiva U.",
    "Brigham Young University": "Brigham Young U., Provo",
    "Colorado State University": "Colorado State U., Fort Collins",
    "Columbia University": "Columbia U. in the City of New York",
    "Dartmouth College": "Dartmouth C. and Dartmouth Hitchcock Medical Center",
    "Indiana University": "Indiana U., Bloomington",
    "Johns Hopkins University": "Johns Hopkins U.a",
    "Louisiana State University": "Louisiana State U., Baton Rouge",
    "Loyola University Medical Center": "Loyola U., Chicago",
    "Medical College of Ohio": "U. Toledo",
    "Montana State University": "Montana State U., Bozeman",
    "Oklahoma State University": "Oklahoma State U., Stillwater",
    "Penn State University": "Pennsylvania State U., University Park and Hershey Medical Center",
    "Purdue University": "Purdue U., West Lafayette",
    "Southern Illinois University": "Southern Illinois U., Carbondale",
    "St. Louis University": "Saint Louis U.",
    "University of Alabama in Birmingham (UAB)": "U. Alabama, Birmingham",
    "University of Alabama in Huntsville (UAH)": "U. Alabama, Huntsville",
    "University of Arkansas": "U. Arkansas, Fayetteville",
    "University of Hawaii": "U. Hawaii, Manoa",
    "University of Massachusetts Medical Center": "U. Massachusetts, Medical School",
    "University of Michigan": "U. Michigan, Ann Arbor",
    "University of Minnesota": "U. Minnesota, Twin Cities",
    "University of Montana": "U. Montana, Missoula",
    "University of Nebraska": "U. Nebraska, Lincoln",
    "University of North Texas": "U. North Texas, Denton",
    "University of Oklahoma": "U. Oklahoma, Norman and Health Science Center",
    "University of Pittsburgh": "U. Pittsburgh, Pittsburgh",
    "University of South Carolina": "U. South Carolina, Columbia",
    "University of Tennessee": "U. Tennessee, Knoxville",
    "University of Texas Houston Hlth. Sci. Ctr_": "U. Texas Health Science Center, Houston",
    "University of Texas Medical Branch (Galveston)": "U. Texas Medical Branch",
    "University of Texas Southwestern Med. Ctr_": "U. Texas Southwestern Medical Center",
    "University of Texas at Austin": "U. Texas, Austin",
    "University of Virginia": "U. Virginia, Charlottesville",
    "University of Washington": "U. Washington, Seattle",
    "Virginia Tech": "Virginia Polytechnic Institute and State U.",
}

# Explicitly system-level policy observations are aggregated across their NCSES
# components. The components are declared transparently rather than inferred.
AGGREGATES_MAIN = {
    "Rutgers the State University of NJ": [
        "Rutgers, State U. New Jersey, New Brunswick",
        "Rutgers, State U. New Jersey, Newark",
        "Rutgers, State U. New Jersey, Camden",
    ],
    "SUNY": [
        "SUNY, U. Buffalo", "SUNY, Stony Brook U.", "SUNY, U. Albany",
        "SUNY, Binghamton U.", "SUNY, Upstate Medical U.",
        "SUNY, Downstate Medical Center", "SUNY, C. of Environmental Science and Forestry",
        "SUNY, C. of Optometry", "SUNY, Geneseo", "SUNY, Buffalo State",
        "SUNY, Oswego", "SUNY, C. Plattsburgh", "SUNY, C. Brockport",
    ],
    "Texas A&M University System": [
        "Texas A&M U., College Station and Health Science Center",
        "Texas A&M U.-Corpus Christi", "Texas A&M U.-Kingsville",
        "Texas A&M International U.", "West Texas A&M U.", "Texas A&M U.-Commerce",
        "Prairie View A&M U.", "Tarleton State U.",
    ],
    "University of California System": [
        "U. California, San Francisco", "U. California, San Diego",
        "U. California, Los Angeles", "U. California, Davis",
        "U. California, Berkeley", "U. California, Irvine",
        "U. California, Santa Barbara", "U. California, Riverside",
        "U. California, Santa Cruz", "U. California, Office of the President",
        "U. California, Merced",
    ],
    "University of Missouri System": [
        "U. Missouri, Columbia", "U. Missouri, Kansas City", "U. Missouri, Saint Louis",
        "Missouri U. of Science and Technology",
    ],
}

# Alternative aggregation for ordinary university names that may represent a
# multi-campus governance unit. Main results use the principal research campus;
# this dictionary is used only in a robustness sample.
AGGREGATES_SENSITIVITY = {
    "Indiana University": ["Indiana U., Bloomington", "Indiana U.-Purdue U., Indianapolis"],
    "Penn State University": [
        "Pennsylvania State U., University Park and Hershey Medical Center",
        "Pennsylvania State U., Behrend", "Pennsylvania State U., Harrisburg",
    ],
    "Purdue University": ["Purdue U., West Lafayette", "Purdue U., Fort Wayne"],
    "Southern Illinois University": ["Southern Illinois U., Carbondale", "Southern Illinois U., Edwardsville"],
    "University of Arkansas": [
        "U. Arkansas, Fayetteville", "U. Arkansas for Medical Sciences",
        "U. Arkansas, Little Rock", "U. Arkansas, Pine Bluff",
    ],
    "University of Hawaii": ["U. Hawaii, Manoa", "U. Hawaii, Hilo"],
    "University of Michigan": ["U. Michigan, Ann Arbor", "U. Michigan, Dearborn", "U. Michigan, Flint"],
    "University of Minnesota": ["U. Minnesota, Twin Cities", "U. Minnesota, Duluth", "U. Minnesota, Morris"],
    "University of Nebraska": [
        "U. Nebraska, Lincoln", "U. Nebraska, Medical Center", "U. Nebraska, Omaha", "U. Nebraska, Kearney",
    ],
    "University of North Texas": ["U. North Texas, Denton", "U. North Texas, Health Science Center"],
    "University of South Carolina": ["U. South Carolina, Columbia", "U. South Carolina, Aiken"],
    "University of Tennessee": [
        "U. Tennessee, Knoxville", "U. Tennessee, Health Science Center",
        "U. Tennessee, Knoxville, Institute of Agriculture", "U. Tennessee, Chattanooga",
    ],
    "University of Washington": ["U. Washington, Seattle", "U. Washington, Bothell", "U. Washington, Tacoma"],
}

# U. Maryland Baltimore is not separately recoverable in the 2010 rank-table
# source used here; matching it to Baltimore County or U. Baltimore would be wrong.
EXCLUDE = {"University of Maryland Baltimore": "2010 NCSES row not cleanly recoverable in the selected tables"}


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def canon(x: object) -> str:
    x = str(x).lower().replace("&", " and ")
    x = re.sub(r"\^[a-z0-9]+", " ", x)
    replacements = [
        (r"\bu\.?\b", " university "), (r"\bc\.?\b", " college "),
        (r"\buniv\.?\b", " university "), (r"\bctr\.?\b", " center "),
        (r"\bsci\.?\b", " science "), (r"\bhlth\.?\b", " health "),
        (r"\bmed\.?\b", " medical "), (r"\binst\.?\b", " institute "),
        (r"\btech\.?\b", " technology "),
    ]
    for p, r in replacements:
        x = re.sub(p, r, x)
    x = re.sub(r"\([^)]*\)", " ", x)
    x = re.sub(r"\b(the|of|in|at)\b", " ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def load_ncses() -> pd.DataFrame:
    total = read_rank_xlsx(TOTAL_URL, "total")
    fed = read_rank_xlsx(FED_URL, "fed")
    n = total.merge(fed, on="ncses_name", how="inner", validate="one_to_one")
    n["canon"] = n["ncses_name"].map(canon)
    return n


def component_sum(n: pd.DataFrame, names: list[str]) -> tuple[float, float]:
    missing = [x for x in names if x not in set(n["ncses_name"])]
    if missing:
        raise ValueError(f"Declared NCSES components not found: {missing}")
    sub = n[n["ncses_name"].isin(names)]
    return float(sub["total_2010"].sum()), float(sub["fed_2010"].sum())


def build_crosswalk(policy_institutions: list[str], n: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    canon_groups = n.groupby("canon")["ncses_name"].apply(list).to_dict()
    for inst in policy_institutions:
        if inst in EXCLUDE:
            rows.append({"Institution": inst, "variant": variant, "method": "excluded", "components": "", "reason": EXCLUDE[inst], "total_2010": np.nan, "fed_2010": np.nan})
            continue
        aggregates = dict(AGGREGATES_MAIN)
        if variant == "system_sensitivity":
            aggregates.update(AGGREGATES_SENSITIVITY)
        if inst in aggregates:
            comps = aggregates[inst]
            total, fed = component_sum(n, comps)
            rows.append({"Institution": inst, "variant": variant, "method": "aggregate", "components": " | ".join(comps), "reason": "declared system aggregation", "total_2010": total, "fed_2010": fed})
            continue
        if inst in SINGLE:
            name = SINGLE[inst]
            if name not in set(n["ncses_name"]):
                raise ValueError(f"Manual NCSES mapping not found for {inst}: {name}")
            r = n[n["ncses_name"] == name].iloc[0]
            rows.append({"Institution": inst, "variant": variant, "method": "manual_single", "components": name, "reason": "verified mechanical/legacy name mapping", "total_2010": r["total_2010"], "fed_2010": r["fed_2010"]})
            continue
        key = canon(inst)
        candidates = canon_groups.get(key, [])
        if len(candidates) == 1:
            r = n[n["ncses_name"] == candidates[0]].iloc[0]
            rows.append({"Institution": inst, "variant": variant, "method": "canonical_exact", "components": candidates[0], "reason": "unique canonical exact match", "total_2010": r["total_2010"], "fed_2010": r["fed_2010"]})
        else:
            rows.append({"Institution": inst, "variant": variant, "method": "excluded", "components": " | ".join(candidates), "reason": f"no unique canonical exact match ({len(candidates)} candidates)", "total_2010": np.nan, "fed_2010": np.nan})
    out = pd.DataFrame(rows)
    out["federal_share_2010"] = out["fed_2010"] / out["total_2010"]
    return out


def make_exposure(crosswalk: pd.DataFrame) -> pd.DataFrame:
    v = crosswalk.dropna(subset=["federal_share_2010"]).copy()
    v = v[(v["federal_share_2010"] >= 0) & (v["federal_share_2010"] <= 1.05)].copy()
    v["federal_share_z"] = (v["federal_share_2010"] - v["federal_share_2010"].mean()) / v["federal_share_2010"].std(ddof=0)
    v["log_federal_rd"] = np.log1p(v["fed_2010"].clip(lower=0))
    v["log_federal_rd_z"] = (v["log_federal_rd"] - v["log_federal_rd"].mean()) / v["log_federal_rd"].std(ddof=0)
    v["high_share_median"] = (v["federal_share_2010"] >= v["federal_share_2010"].median()).astype(int)
    v["high_share_toptercile"] = (v["federal_share_2010"] >= v["federal_share_2010"].quantile(2/3)).astype(int)
    return v


def cluster_fit(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["Institution"]})


def did(panel, exposure, outcome, exp, start, end, observed_only=False):
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["exp_post"] = d[exp] * d["post"]
    d = d.dropna(subset=[outcome, exp, "exp_post"])
    fit = cluster_fit(f"{outcome} ~ exp_post + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["exp_post"]
    return {"outcome": outcome, "exposure": exp, "start": start, "end": end, "observed_only": observed_only,
            "N": int(fit.nobs), "institutions": d["Institution"].nunique(), "coef": fit.params["exp_post"],
            "se": fit.bse["exp_post"], "p": fit.pvalues["exp_post"], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "r2": fit.rsquared}


def event_study(panel, exposure, outcome, start=2006, end=2017, ref=2010):
    exp = "federal_share_z"
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].dropna(subset=[outcome, exp]).copy()
    terms, years = [], []
    for y in range(start, end + 1):
        if y in {ref, SHOCK_YEAR}: continue
        term = f"ev_{y}"; d[term] = (d["Year"] == y).astype(int) * d[exp]; terms.append(term); years.append(y)
    fit = cluster_fit(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)", d)
    rows = []
    for y, term in zip(years, terms):
        ci = fit.conf_int().loc[term]
        rows.append({"outcome": outcome, "year": y, "event_time": y - SHOCK_YEAR, "coef": fit.params[term], "se": fit.bse[term], "p": fit.pvalues[term], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "N": int(fit.nobs), "institutions": d["Institution"].nunique()})
    leads = [f"ev_{y}" for y in years if y < SHOCK_YEAR and y != ref]
    test = fit.wald_test(" = 0, ".join(leads) + " = 0", scalar=True)
    pre = {"outcome": outcome, "n_leads": len(leads), "stat": float(np.asarray(test.statistic).squeeze()), "p": float(np.asarray(test.pvalue).squeeze())}
    return pd.DataFrame(rows), pre


def placebo(panel, exposure, outcome, placebo_year):
    d = panel.merge(exposure[["Institution", "federal_share_z"]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(placebo_year - 4, placebo_year + 4) & (d["Year"] != placebo_year)].copy()
    d["post"] = (d["Year"] > placebo_year).astype(int); d["exp_post"] = d["federal_share_z"] * d["post"]
    fit = cluster_fit(f"{outcome} ~ exp_post + C(Institution) + C(Year)", d)
    return {"outcome": outcome, "placebo_year": placebo_year, "coef": fit.params["exp_post"], "se": fit.bse["exp_post"], "p": fit.pvalues["exp_post"], "N": int(fit.nobs)}


def revision_hazard(panel, exposure):
    d = panel.merge(exposure[["Institution", "federal_share_z"]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(2007, 2016) & (d["Year"] != SHOCK_YEAR)].copy()
    d["revision"] = (d["Is_Carried_Forward"] == 0).astype(int); d["post"] = (d["Year"] > SHOCK_YEAR).astype(int); d["exp_post"] = d["federal_share_z"] * d["post"]
    fit = cluster_fit("revision ~ exp_post + C(Institution) + C(Year)", d); ci = fit.conf_int().loc["exp_post"]
    return {"outcome": "revision_hazard", "coef": fit.params["exp_post"], "se": fit.bse["exp_post"], "p": fit.pvalues["exp_post"], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "N": int(fit.nobs), "institutions": d["Institution"].nunique()}


def main():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel = panel[(panel["Year"] >= 1944) & (panel["Year"] <= 2025)].copy()
    sentences = pd.read_csv(SENTENCE_PATH, low_memory=False)
    panel = attach_language_to_panel(panel, build_language_measures(sentences))
    ncses = load_ncses()
    institutions = sorted(panel.loc[panel["Year"] == EXPOSURE_YEAR, "Institution"].dropna().unique())

    outcomes = ["has_strong_assignment", "strong_per_1000w", "weak_per_1000w", "net_assignment_strength",
                "Mean_Tone_Score", "Tone_Index", "Legal_Load_Index", "obligation_modal_share",
                "restrictive_share", "roche2011_per_1000w"]
    windows = [(2007, 2016), (2009, 2015), (2004, 2018)]
    exposures = ["federal_share_z", "log_federal_rd_z", "high_share_median", "high_share_toptercile"]

    all_did, all_pre, all_event, all_revision, all_placebo = [], [], [], [], []
    audit_frames = []
    for variant in ["principal_main", "system_sensitivity"]:
        cross = build_crosswalk(institutions, ncses, variant)
        audit_frames.append(cross)
        expdf = make_exposure(cross)
        save(expdf, f"exposure_{variant}.csv")
        for start, end in windows:
            for expvar in exposures:
                for outcome in outcomes:
                    r = did(panel, expdf, outcome, expvar, start, end, False); r["variant"] = variant; all_did.append(r)
        for expvar in ["federal_share_z", "high_share_median"]:
            for outcome in outcomes:
                r = did(panel, expdf, outcome, expvar, 2007, 2016, True); r["variant"] = variant; all_did.append(r)
        for outcome in outcomes:
            ev, pre = event_study(panel, expdf, outcome); ev["variant"] = variant; pre["variant"] = variant; all_event.append(ev); all_pre.append(pre)
            for py in [2007, 2015]:
                pr = placebo(panel, expdf, outcome, py); pr["variant"] = variant; all_placebo.append(pr)
        rr = revision_hazard(panel, expdf); rr["variant"] = variant; all_revision.append(rr)

    audit = pd.concat(audit_frames, ignore_index=True); save(audit, "crosswalk_audit.csv")
    dids = pd.DataFrame(all_did); save(dids, "did_results.csv")
    save(pd.concat(all_event, ignore_index=True), "event_study_results.csv")
    save(pd.DataFrame(all_pre), "pretrend_joint_tests.csv")
    save(pd.DataFrame(all_revision), "revision_hazard.csv")
    save(pd.DataFrame(all_placebo), "placebo_results.csv")

    summary_rows = []
    for variant in ["principal_main", "system_sensitivity"]:
        c = audit[audit["variant"] == variant]
        e = make_exposure(c)
        summary_rows.append({"variant": variant, "policy_institutions_2010": len(c), "usable_exposure": len(e),
                             "excluded": int((c["method"] == "excluded").sum()), "canonical_exact": int((c["method"] == "canonical_exact").sum()),
                             "manual_single": int((c["method"] == "manual_single").sum()), "aggregates": int((c["method"] == "aggregate").sum()),
                             "mean_federal_share": e["federal_share_2010"].mean(), "median_federal_share": e["federal_share_2010"].median(), "sd_federal_share": e["federal_share_2010"].std()})
    save(pd.DataFrame(summary_rows), "crosswalk_summary.csv")

    main = dids[(dids["variant"] == "principal_main") & (dids["start"] == 2007) & (dids["end"] == 2016) & (dids["exposure"] == "federal_share_z") & (dids["observed_only"] == False)]
    pre = pd.DataFrame(all_pre); pre = pre[pre["variant"] == "principal_main"]
    lines = ["# Audited federal-R&D exposure design", "", "Shock: June 2011 Stanford v. Roche Supreme Court decision.", "Exposure: FY2010 federally financed R&D / total R&D.", "No fuzzy university matches are used.", "", "## Crosswalk"]
    for r in summary_rows: lines.append(f"- {r['variant']}: usable={r['usable_exposure']}, excluded={r['excluded']}, exact={r['canonical_exact']}, manual={r['manual_single']}, aggregates={r['aggregates']}")
    lines += ["", "## Main 2007-2016 DiD: one-SD higher federal R&D share"]
    for _, r in main.iterrows(): lines.append(f"- {r['outcome']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}, N={int(r['N'])}, universities={int(r['institutions'])}")
    lines += ["", "## Joint pretrend tests"]
    for _, r in pre.iterrows(): lines.append(f"- {r['outcome']}: p={r['p']:.4f}")
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
