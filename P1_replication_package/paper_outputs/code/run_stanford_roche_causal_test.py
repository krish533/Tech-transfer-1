from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.contrast import ContrastResults

PRIMARY_START = 1944
PRIMARY_END = 2025
MAIN_SHOCK_YEAR = 2009  # Federal Circuit decision: Sept. 30, 2009
ALT_SHOCK_YEAR = 2011   # Supreme Court decision: June 6, 2011

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PACKAGE_ROOT / "data" / "derived" / "policy_level_indices_institution_year.csv"
SENTENCE_PATH = PACKAGE_ROOT / "data" / "derived" / "sentence_scores_canonical.csv"
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "stanford_roche_causal"

# The Federal Circuit's assignment-language distinction motivates these deliberately
# transparent dictionaries. These are not used to construct PCSI itself.
STRONG_PATTERNS = [
    r"\bhereby\s+assign(?:s|ed|ing)?\b",
    r"\bdoes\s+hereby\s+assign\b",
    r"\bhereby\s+transfer(?:s|red|ring)?\b",
    r"\bassign(?:s)?\s+and\s+hereby\s+assign(?:s)?\b",
    r"\bautomatically\s+(?:assign(?:s|ed)?|vest(?:s|ed)?)\b",
    r"\bis\s+hereby\s+assigned\b",
    r"\bare\s+hereby\s+assigned\b",
]
WEAK_PATTERNS = [
    r"\bagree(?:s|d)?\s+to\s+assign\b",
    r"\bwill\s+assign\b",
    r"\bshall\s+assign\b",
    r"\bpromise(?:s|d)?\s+to\s+assign\b",
    r"\bobligat(?:e|ed)\s+to\s+assign\b",
    r"\brequired\s+to\s+assign\b",
]
ASSIGNMENT_PAT = re.compile(r"\bassign(?:ment|s|ed|ing)?\b", flags=re.I)
STRONG_RE = re.compile("|".join(f"(?:{p})" for p in STRONG_PATTERNS), flags=re.I)
WEAK_RE = re.compile("|".join(f"(?:{p})" for p in WEAK_PATTERNS), flags=re.I)


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def clustered_fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["Institution"]}
    )


def build_language_measures(sentences: pd.DataFrame) -> pd.DataFrame:
    s = sentences.copy()
    s = s[(s["Year"] >= PRIMARY_START) & (s["Year"] <= PRIMARY_END)].copy()
    text = s["Sentence_Cleaned"].fillna("").astype(str)
    s["strong_assignment_sentence"] = text.str.contains(STRONG_RE, regex=True).astype(int)
    s["weak_assignment_sentence"] = text.str.contains(WEAK_RE, regex=True).astype(int)
    s["assignment_sentence"] = text.str.contains(ASSIGNMENT_PAT, regex=True).astype(int)
    s["sentence_words"] = text.str.split().str.len().clip(lower=1)

    obs = (
        s.groupby(["Institution", "Year"], as_index=False)
        .agg(
            n_policy_sentences=("Sentence_Cleaned", "size"),
            sentence_words=("sentence_words", "sum"),
            strong_assignment_count=("strong_assignment_sentence", "sum"),
            weak_assignment_count=("weak_assignment_sentence", "sum"),
            assignment_sentence_count=("assignment_sentence", "sum"),
        )
    )
    denom = obs["sentence_words"].replace(0, np.nan)
    obs["strong_per_1000w"] = 1000 * obs["strong_assignment_count"] / denom
    obs["weak_per_1000w"] = 1000 * obs["weak_assignment_count"] / denom
    obs["assignment_per_1000w"] = 1000 * obs["assignment_sentence_count"] / denom
    obs["net_assignment_strength"] = obs["strong_per_1000w"] - obs["weak_per_1000w"]
    obs["has_strong_assignment"] = (obs["strong_assignment_count"] > 0).astype(int)
    obs["has_weak_assignment"] = (obs["weak_assignment_count"] > 0).astype(int)
    return obs


def attach_language_to_panel(panel: pd.DataFrame, obs_lang: pd.DataFrame) -> pd.DataFrame:
    language_cols = [c for c in obs_lang.columns if c not in {"Institution", "Year"}]
    p = panel.merge(
        obs_lang.rename(columns={"Year": "Source_Year"}),
        on=["Institution", "Source_Year"],
        how="left",
        validate="many_to_one",
    )
    if p[language_cols].isna().all(axis=None):
        raise ValueError("Language measures failed to merge to policy panel.")
    return p


def build_exposure(panel: pd.DataFrame, baseline_year: int = 2008) -> pd.DataFrame:
    base = panel.loc[panel["Year"] == baseline_year].copy()
    keep = [
        "Institution", "Source_Year", "has_strong_assignment", "has_weak_assignment",
        "strong_per_1000w", "weak_per_1000w", "net_assignment_strength",
        "Mean_Tone_Score", "Legal_Load_Index", "obligation_modal_share",
        "Private", "Carnegie R1", "MEDSCHOOL", "Land-Grant Institution", "Type",
    ]
    base = base[keep].dropna(subset=["has_strong_assignment", "has_weak_assignment"]).copy()
    base["strict_vulnerable"] = (
        (base["has_weak_assignment"] == 1) & (base["has_strong_assignment"] == 0)
    ).astype(int)
    base["broad_vulnerable"] = (base["has_strong_assignment"] == 0).astype(int)
    base["weak_any"] = base["has_weak_assignment"].astype(int)
    # Continuous exposure is higher when the baseline policy relies more on future-promise
    # wording relative to present-assignment wording.
    base["continuous_exposure"] = (
        base["weak_per_1000w"] - base["strong_per_1000w"]
    )
    return base


def treatment_summary(exposure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["strict_vulnerable", "broad_vulnerable", "weak_any"]:
        rows.append({
            "exposure_definition": col,
            "N_institutions": len(exposure),
            "treated": int(exposure[col].sum()),
            "control": int((1 - exposure[col]).sum()),
            "treated_share": float(exposure[col].mean()),
        })
    return pd.DataFrame(rows)


def did_one(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exposure_col: str,
            shock_year: int, start_year: int, end_year: int, exclude_shock: bool = True) -> dict:
    d = panel.merge(exposure[["Institution", exposure_col]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start_year, end_year)].copy()
    if exclude_shock:
        d = d[d["Year"] != shock_year].copy()
    d["post"] = (d["Year"] > shock_year).astype(int)
    d["treat_post"] = d[exposure_col] * d["post"]
    d = d.dropna(subset=[outcome, exposure_col, "treat_post"])
    if d["Institution"].nunique() < 10 or d[exposure_col].nunique() < 2:
        return {
            "outcome": outcome, "exposure": exposure_col, "shock_year": shock_year,
            "start_year": start_year, "end_year": end_year, "N": len(d),
            "institutions": d["Institution"].nunique(), "coef": np.nan, "se": np.nan,
            "p": np.nan, "ci_low": np.nan, "ci_high": np.nan,
        }
    fit = clustered_fit(f"{outcome} ~ treat_post + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["treat_post"]
    return {
        "outcome": outcome,
        "exposure": exposure_col,
        "shock_year": shock_year,
        "start_year": start_year,
        "end_year": end_year,
        "N": int(fit.nobs),
        "institutions": int(d["Institution"].nunique()),
        "treated_institutions": int(d.loc[d[exposure_col] == 1, "Institution"].nunique()),
        "control_institutions": int(d.loc[d[exposure_col] == 0, "Institution"].nunique()),
        "coef": float(fit.params["treat_post"]),
        "se": float(fit.bse["treat_post"]),
        "p": float(fit.pvalues["treat_post"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "r2": float(fit.rsquared),
    }


def run_dids(panel: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    outcomes = [
        "has_strong_assignment",
        "strong_per_1000w",
        "weak_per_1000w",
        "net_assignment_strength",
        "Mean_Tone_Score",
        "Legal_Load_Index",
        "obligation_modal_share",
    ]
    exposures = ["strict_vulnerable", "broad_vulnerable", "weak_any"]
    specs = [
        (MAIN_SHOCK_YEAR, 2004, 2014),
        (MAIN_SHOCK_YEAR, 2006, 2013),
        (MAIN_SHOCK_YEAR, 2000, 2016),
        (ALT_SHOCK_YEAR, 2006, 2016),
    ]
    rows = []
    for shock, start, end in specs:
        for exp in exposures:
            for outcome in outcomes:
                rows.append(did_one(panel, exposure, outcome, exp, shock, start, end))
    return pd.DataFrame(rows)


def event_study(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str,
                exposure_col: str = "strict_vulnerable", shock_year: int = MAIN_SHOCK_YEAR,
                start_year: int = 2004, end_year: int = 2015, reference_year: int = 2008) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = panel.merge(exposure[["Institution", exposure_col]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start_year, end_year) & (d["Year"] != shock_year)].copy()
    d = d.dropna(subset=[outcome, exposure_col])

    term_names = []
    years = [y for y in range(start_year, end_year + 1) if y not in {reference_year, shock_year}]
    for y in years:
        term = f"event_{y}"
        d[term] = ((d["Year"] == y).astype(int) * d[exposure_col]).astype(int)
        term_names.append(term)

    formula = f"{outcome} ~ " + " + ".join(term_names) + " + C(Institution) + C(Year)"
    fit = clustered_fit(formula, d)

    rows = []
    for y, term in zip(years, term_names):
        ci = fit.conf_int().loc[term]
        rows.append({
            "outcome": outcome,
            "exposure": exposure_col,
            "shock_year": shock_year,
            "reference_year": reference_year,
            "year": y,
            "event_time": y - shock_year,
            "coef": float(fit.params[term]),
            "se": float(fit.bse[term]),
            "p": float(fit.pvalues[term]),
            "ci_low": float(ci.iloc[0]),
            "ci_high": float(ci.iloc[1]),
            "N": int(fit.nobs),
            "institutions": int(d["Institution"].nunique()),
        })

    # Joint test that all available pre-treatment interactions except the reference are zero.
    lead_terms = [f"event_{y}" for y in years if y < shock_year]
    lead_terms = [t for t in lead_terms if int(t.split("_")[1]) != reference_year]
    if lead_terms:
        hypothesis = " = 0, ".join(lead_terms) + " = 0"
        test = fit.wald_test(hypothesis, scalar=True)
        pre = pd.DataFrame([{
            "outcome": outcome,
            "exposure": exposure_col,
            "pretrend_start": start_year,
            "pretrend_end": shock_year - 1,
            "reference_year": reference_year,
            "F_or_chi2": float(np.asarray(test.statistic).squeeze()),
            "p": float(np.asarray(test.pvalue).squeeze()),
            "n_leads_tested": len(lead_terms),
        }])
    else:
        pre = pd.DataFrame()
    return pd.DataFrame(rows), pre


def revision_hazard_did(panel: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    d = panel.merge(exposure[["Institution", "strict_vulnerable"]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(2004, 2014) & (d["Year"] != MAIN_SHOCK_YEAR)].copy()
    d["revision_observed"] = (d["Is_Carried_Forward"] == 0).astype(int)
    d["post"] = (d["Year"] > MAIN_SHOCK_YEAR).astype(int)
    d["treat_post"] = d["strict_vulnerable"] * d["post"]
    fit = clustered_fit("revision_observed ~ treat_post + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["treat_post"]
    return pd.DataFrame([{
        "outcome": "revision_observed",
        "exposure": "strict_vulnerable",
        "coef": float(fit.params["treat_post"]),
        "se": float(fit.bse["treat_post"]),
        "p": float(fit.pvalues["treat_post"]),
        "ci_low": float(ci.iloc[0]),
        "ci_high": float(ci.iloc[1]),
        "N": int(fit.nobs),
        "institutions": int(d["Institution"].nunique()),
    }])


def write_summary(exposure: pd.DataFrame, did: pd.DataFrame, pretests: pd.DataFrame,
                  revision: pd.DataFrame) -> None:
    main = did[
        (did["shock_year"] == MAIN_SHOCK_YEAR)
        & (did["start_year"] == 2004)
        & (did["end_year"] == 2014)
        & (did["exposure"] == "strict_vulnerable")
    ].copy()
    lines = [
        "# Stanford–Roche exploratory causal validation",
        "",
        "This analysis treats the September 30, 2009 Federal Circuit decision as the primary external shock.",
        "The shock year itself is excluded from the main specification. Exposure is fixed using the policy in force in 2008.",
        "",
        "## Exposure counts",
    ]
    for col in ["strict_vulnerable", "broad_vulnerable", "weak_any"]:
        lines.append(f"- {col}: {int(exposure[col].sum())} treated of {len(exposure)} baseline institutions")
    lines += ["", "## Main 2004–2014 DiD, strict exposure"]
    for _, r in main.iterrows():
        lines.append(
            f"- {r['outcome']}: beta={r['coef']:.4f}, SE={r['se']:.4f}, p={r['p']:.4g}, "
            f"95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )
    lines += ["", "## Event-study pretrend tests"]
    for _, r in pretests.iterrows():
        lines.append(f"- {r['outcome']}: joint pretrend p={r['p']:.4g} ({int(r['n_leads_tested'])} leads)")
    if not revision.empty:
        r = revision.iloc[0]
        lines += ["", "## Revision hazard", f"- beta={r['coef']:.4f}, SE={r['se']:.4f}, p={r['p']:.4g}"]
    lines += [
        "",
        "## Interpretation guardrail",
        "Treat these estimates as a design-feasibility diagnostic, not a final causal claim. The design is credible only if exposure is substantively defensible, pretrends are acceptable, and results are robust to alternative exposure definitions/windows.",
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    sentences = pd.read_csv(SENTENCE_PATH, low_memory=False)
    panel = panel[(panel["Year"] >= PRIMARY_START) & (panel["Year"] <= PRIMARY_END)].copy()

    obs_lang = build_language_measures(sentences)
    save(obs_lang, "observed_policy_assignment_language.csv")
    panel = attach_language_to_panel(panel, obs_lang)

    exposure = build_exposure(panel, baseline_year=2008)
    save(exposure, "baseline_2008_exposure.csv")
    save(treatment_summary(exposure), "treatment_counts.csv")

    did = run_dids(panel, exposure)
    save(did, "did_results.csv")

    event_frames, pre_frames = [], []
    for outcome in ["has_strong_assignment", "strong_per_1000w", "net_assignment_strength", "Mean_Tone_Score", "Legal_Load_Index", "obligation_modal_share"]:
        ev, pre = event_study(panel, exposure, outcome=outcome)
        event_frames.append(ev)
        if not pre.empty:
            pre_frames.append(pre)
    events = pd.concat(event_frames, ignore_index=True)
    pretests = pd.concat(pre_frames, ignore_index=True) if pre_frames else pd.DataFrame()
    save(events, "event_study_results.csv")
    save(pretests, "pretrend_joint_tests.csv")

    revision = revision_hazard_did(panel, exposure)
    save(revision, "revision_hazard_did.csv")

    write_summary(exposure, did, pretests, revision)
    print((OUTDIR / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
