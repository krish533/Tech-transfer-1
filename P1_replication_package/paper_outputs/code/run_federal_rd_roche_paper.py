from __future__ import annotations

import re
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PACKAGE_ROOT / "data" / "derived" / "policy_level_indices_institution_year.csv"
SENTENCE_PATH = PACKAGE_ROOT / "data" / "derived" / "sentence_scores_canonical.csv"
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "federal_rd_roche_paper"

TOTAL_URL = "https://ncsesdata.nsf.gov/herd/2014/html/HERD2014_DST_17.html"
FED_URL = "https://ncsesdata.nsf.gov/herd/2014/html/HERD2014_DST_20.html"
SHOCK_YEAR = 2009
PRE_YEARS = [2006, 2007, 2008]

STRONG_RE = re.compile(
    r"\bhereby\s+assign(?:s|ed|ing)?\b|\bdoes\s+hereby\s+assign\b|"
    r"\bhereby\s+transfer(?:s|red|ring)?\b|\bis\s+hereby\s+assigned\b|"
    r"\bare\s+hereby\s+assigned\b|\bautomatically\s+(?:assign(?:s|ed)?|vest(?:s|ed)?)\b",
    flags=re.I,
)
WEAK_RE = re.compile(
    r"\bagree(?:s|d)?\s+to\s+assign\b|\bwill\s+assign\b|\bshall\s+assign\b|"
    r"\bpromise(?:s|d)?\s+to\s+assign\b|\brequired\s+to\s+assign\b|"
    r"\bobligat(?:e|ed)\s+to\s+assign\b",
    flags=re.I,
)
OWNERSHIP_RE = re.compile(r"\bownership\b|\btitle\b|\bowned by\b|\bshall belong to\b", flags=re.I)
BAYH_RE = re.compile(r"bayh[- ]dole|federally funded|federal funding|government rights|subject inventions?", flags=re.I)

ALIASES = {
    "california institute of technology": "caltech",
    "massachusetts institute of technology": "mit",
    "georgia institute of technology": "georgia tech",
    "university of california berkeley": "uc berkeley",
    "university of california los angeles": "ucla",
    "university of california san diego": "uc san diego",
    "university of california san francisco": "uc san francisco",
    "university of california davis": "uc davis",
    "university of california irvine": "uc irvine",
    "university of california santa barbara": "uc santa barbara",
    "university of california santa cruz": "uc santa cruz",
    "university of north carolina chapel hill": "unc chapel hill",
    "university of illinois urbana champaign": "illinois urbana champaign",
    "university of wisconsin madison": "wisconsin madison",
    "university of michigan ann arbor": "michigan ann arbor",
    "university of washington seattle": "washington seattle",
}


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def norm_name(x: object) -> str:
    s = str(x).lower()
    s = re.sub(r"\^\{.*?\}|\*|†|‡", " ", s)
    s = s.replace("univ.", "university").replace("u.", "university ")
    s = s.replace("univ ", "university ")
    s = re.sub(r"\buniversity of\b", "university of", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bthe\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for col in out.columns:
            parts = [str(v) for v in col if str(v) != "nan" and not str(v).startswith("Unnamed")]
            cols.append(" ".join(dict.fromkeys(parts)))
        out.columns = cols
    else:
        out.columns = [str(c) for c in out.columns]
    return out


def parse_rank_table(url: str, prefix: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    candidates = []
    for t in tables:
        f = flatten_columns(t)
        text = " ".join(f.columns)
        if "Institution" in text and all(str(y) in text for y in PRE_YEARS):
            candidates.append(f)
    if not candidates:
        raise RuntimeError(f"Could not identify institution table at {url}")
    df = max(candidates, key=len).copy()
    inst_col = next(c for c in df.columns if "Institution" in c)
    out = pd.DataFrame({"ncses_name": df[inst_col].astype(str)})
    for y in PRE_YEARS:
        matches = [c for c in df.columns if re.search(rf"(^|\s){y}(\s|$)", c)]
        if not matches:
            raise RuntimeError(f"No {y} column in {url}: {df.columns.tolist()}")
        vals = df[matches[0]].astype(str).str.replace(",", "", regex=False)
        vals = vals.replace({"NA": np.nan, "na": np.nan, "ne": np.nan, "i": np.nan, "nan": np.nan})
        out[f"{prefix}_{y}"] = pd.to_numeric(vals, errors="coerce")
    out = out[~out["ncses_name"].str.contains("All institutions|All other surveyed", case=False, na=False)].copy()
    out["ncses_norm"] = out["ncses_name"].map(norm_name)
    return out


def best_match(name: str, choices: list[str]) -> tuple[str, float]:
    n = norm_name(name)
    exact = [c for c in choices if c == n]
    if exact:
        return exact[0], 1.0
    # token overlap + sequence ratio; deliberately conservative
    nt = set(n.split())
    best_c, best_s = "", -1.0
    for c in choices:
        ct = set(c.split())
        j = len(nt & ct) / max(1, len(nt | ct))
        seq = SequenceMatcher(None, n, c).ratio()
        score = 0.62 * seq + 0.38 * j
        if score > best_s:
            best_c, best_s = c, score
    return best_c, best_s


def build_ncses_exposure(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = parse_rank_table(TOTAL_URL, "total")
    fed = parse_rank_table(FED_URL, "fed")
    rd = total.merge(fed.drop(columns=["ncses_name"]), on="ncses_norm", how="outer", suffixes=("", "_fedname"))
    rd["ncses_name"] = rd["ncses_name"].fillna(rd.get("ncses_name_fedname"))
    choices = sorted(rd["ncses_norm"].dropna().unique())

    institutions = sorted(panel.loc[panel["Year"] == 2008, "Institution"].dropna().unique())
    rows = []
    for inst in institutions:
        match, score = best_match(inst, choices)
        rows.append({"Institution": inst, "panel_norm": norm_name(inst), "ncses_norm": match, "match_score": score})
    crosswalk = pd.DataFrame(rows)
    crosswalk["accepted"] = crosswalk["match_score"] >= 0.72
    crosswalk = crosswalk.merge(rd, on="ncses_norm", how="left")

    for y in PRE_YEARS:
        crosswalk[f"fed_share_{y}"] = crosswalk[f"fed_{y}"] / crosswalk[f"total_{y}"].replace(0, np.nan)
    crosswalk["pre_fed_share"] = crosswalk[[f"fed_share_{y}" for y in PRE_YEARS]].mean(axis=1)
    crosswalk["pre_fed_rd_mean"] = crosswalk[[f"fed_{y}" for y in PRE_YEARS]].mean(axis=1)
    crosswalk["pre_total_rd_mean"] = crosswalk[[f"total_{y}" for y in PRE_YEARS]].mean(axis=1)
    crosswalk["pre_fed_rd_log"] = np.log1p(crosswalk["pre_fed_rd_mean"])
    crosswalk.loc[~crosswalk["accepted"], ["pre_fed_share", "pre_fed_rd_mean", "pre_total_rd_mean", "pre_fed_rd_log"]] = np.nan

    # Standardized continuous exposures and high/low alternatives.
    for col in ["pre_fed_share", "pre_fed_rd_log"]:
        mu = crosswalk[col].mean(skipna=True)
        sd = crosswalk[col].std(skipna=True, ddof=1)
        crosswalk[f"z_{col}"] = (crosswalk[col] - mu) / sd
        med = crosswalk[col].median(skipna=True)
        crosswalk[f"high_{col}"] = np.where(crosswalk[col].notna(), (crosswalk[col] >= med).astype(float), np.nan)
    return crosswalk, rd


def build_text_outcomes(sentences: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    s = sentences.copy()
    text = s["Sentence_Cleaned"].fillna("").astype(str)
    s["words"] = text.str.split().str.len().clip(lower=1)
    s["strong"] = text.str.contains(STRONG_RE, regex=True).astype(int)
    s["weak"] = text.str.contains(WEAK_RE, regex=True).astype(int)
    s["ownership"] = text.str.contains(OWNERSHIP_RE, regex=True).astype(int)
    s["bayh"] = text.str.contains(BAYH_RE, regex=True).astype(int)
    obs = s.groupby(["Institution", "Year"], as_index=False).agg(
        words=("words", "sum"), strong=("strong", "sum"), weak=("weak", "sum"),
        ownership=("ownership", "sum"), bayh=("bayh", "sum")
    )
    for c in ["strong", "weak", "ownership", "bayh"]:
        obs[f"{c}_per_1000w"] = 1000 * obs[c] / obs["words"].replace(0, np.nan)
        obs[f"has_{c}"] = (obs[c] > 0).astype(int)
    obs["net_assignment_strength"] = obs["strong_per_1000w"] - obs["weak_per_1000w"]
    p = panel.merge(obs.rename(columns={"Year": "Source_Year"}), on=["Institution", "Source_Year"], how="left", validate="many_to_one")
    return p


def fit_cluster(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["Institution"]})


def did_continuous(panel: pd.DataFrame, x: pd.DataFrame, outcome: str, exposure: str, start: int, end: int, observed_only=False) -> dict:
    d = panel.merge(x[["Institution", exposure]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d = d.dropna(subset=[outcome, exposure])
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["xp"] = d[exposure] * d["post"]
    if d["Institution"].nunique() < 20:
        return {"outcome": outcome, "exposure": exposure, "start": start, "end": end, "observed_only": observed_only, "N": len(d), "institutions": d["Institution"].nunique(), "coef": np.nan, "se": np.nan, "p": np.nan}
    fit = fit_cluster(f"{outcome} ~ xp + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["xp"]
    return {"outcome": outcome, "exposure": exposure, "start": start, "end": end, "observed_only": observed_only,
            "N": int(fit.nobs), "institutions": int(d["Institution"].nunique()), "coef": float(fit.params["xp"]),
            "se": float(fit.bse["xp"]), "p": float(fit.pvalues["xp"]), "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]), "r2": float(fit.rsquared)}


def event_study(panel: pd.DataFrame, x: pd.DataFrame, outcome: str, exposure: str, start=2004, end=2015, reference=2008) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = panel.merge(x[["Institution", exposure]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].dropna(subset=[outcome, exposure]).copy()
    years = [y for y in range(start, end + 1) if y not in {reference, SHOCK_YEAR}]
    terms = []
    for y in years:
        t = f"ev_{y}"
        d[t] = (d["Year"].eq(y).astype(int) * d[exposure])
        terms.append(t)
    fit = fit_cluster(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)", d)
    rows = []
    for y, t in zip(years, terms):
        ci = fit.conf_int().loc[t]
        rows.append({"outcome": outcome, "exposure": exposure, "year": y, "event_time": y-SHOCK_YEAR,
                     "coef": float(fit.params[t]), "se": float(fit.bse[t]), "p": float(fit.pvalues[t]),
                     "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1])})
    leads = [f"ev_{y}" for y in years if y < SHOCK_YEAR]
    if leads:
        test = fit.wald_test(" = 0, ".join(leads) + " = 0", scalar=True)
        pre = pd.DataFrame([{"outcome": outcome, "exposure": exposure, "n_leads": len(leads),
                             "stat": float(np.asarray(test.statistic).squeeze()), "p": float(np.asarray(test.pvalue).squeeze())}])
    else:
        pre = pd.DataFrame()
    return pd.DataFrame(rows), pre


def placebo_tests(panel: pd.DataFrame, x: pd.DataFrame, outcome: str, exposure: str) -> pd.DataFrame:
    rows = []
    for fake in [2006, 2007, 2008, 2012, 2013]:
        d = panel.merge(x[["Institution", exposure]], on="Institution", how="inner", validate="many_to_one")
        d = d[d["Year"].between(fake-4, fake+4) & (d["Year"] != fake)].dropna(subset=[outcome, exposure]).copy()
        d["post"] = (d["Year"] > fake).astype(int)
        d["xp"] = d[exposure] * d["post"]
        fit = fit_cluster(f"{outcome} ~ xp + C(Institution) + C(Year)", d)
        rows.append({"fake_shock": fake, "outcome": outcome, "exposure": exposure,
                     "coef": float(fit.params["xp"]), "se": float(fit.bse["xp"]), "p": float(fit.pvalues["xp"]),
                     "N": int(fit.nobs), "institutions": int(d["Institution"].nunique())})
    return pd.DataFrame(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel = panel[panel["Year"].between(1944, 2025)].copy()
    sentences = pd.read_csv(SENTENCE_PATH, low_memory=False)
    p = build_text_outcomes(sentences, panel)
    crosswalk, rd = build_ncses_exposure(p)
    save(crosswalk, "ncses_crosswalk_exposure.csv")
    save(rd, "ncses_2006_2008_source_extract.csv")

    diagnostics = pd.DataFrame([{
        "panel_institutions_2008": int(p.loc[p["Year"]==2008, "Institution"].nunique()),
        "accepted_matches": int(crosswalk["accepted"].sum()),
        "share_accepted": float(crosswalk["accepted"].mean()),
        "with_fed_share": int(crosswalk["pre_fed_share"].notna().sum()),
        "mean_match_score": float(crosswalk["match_score"].mean()),
        "median_match_score": float(crosswalk["match_score"].median()),
        "mean_pre_fed_share": float(crosswalk["pre_fed_share"].mean()),
        "sd_pre_fed_share": float(crosswalk["pre_fed_share"].std(ddof=1)),
    }])
    save(diagnostics, "merge_diagnostics.csv")

    outcomes = [
        "Mean_Tone_Score", "Tone_Index", "Legal_Load_Index", "obligation_modal_share",
        "strong_per_1000w", "weak_per_1000w", "net_assignment_strength",
        "ownership_per_1000w", "bayh_per_1000w", "has_strong", "has_ownership", "has_bayh"
    ]
    exposures = ["z_pre_fed_share", "z_pre_fed_rd_log", "high_pre_fed_share", "high_pre_fed_rd_log"]
    windows = [(2004, 2014), (2006, 2013), (2000, 2016), (2004, 2012)]
    rows = []
    for outcome in outcomes:
        for exposure in exposures:
            for start, end in windows:
                for observed_only in [False, True]:
                    rows.append(did_continuous(p, crosswalk, outcome, exposure, start, end, observed_only))
    did = pd.DataFrame(rows)
    save(did, "did_results_full.csv")

    # Event studies for the main paper outcomes under both continuous exposure definitions.
    es_rows, pre_rows = [], []
    for outcome in ["Mean_Tone_Score", "Legal_Load_Index", "obligation_modal_share", "strong_per_1000w", "ownership_per_1000w", "bayh_per_1000w"]:
        for exposure in ["z_pre_fed_share", "z_pre_fed_rd_log"]:
            es, pre = event_study(p, crosswalk, outcome, exposure)
            es_rows.append(es); pre_rows.append(pre)
    save(pd.concat(es_rows, ignore_index=True), "event_study_results.csv")
    save(pd.concat(pre_rows, ignore_index=True), "pretrend_joint_tests.csv")

    placebo = []
    for outcome in ["Mean_Tone_Score", "strong_per_1000w", "ownership_per_1000w", "bayh_per_1000w"]:
        for exposure in ["z_pre_fed_share", "z_pre_fed_rd_log"]:
            placebo.append(placebo_tests(p, crosswalk, outcome, exposure))
    save(pd.concat(placebo, ignore_index=True), "placebo_tests.csv")

    # Compact main-results table for reading.
    main = did[(did["start"]==2004) & (did["end"]==2014) & (~did["observed_only"]) & did["exposure"].isin(["z_pre_fed_share", "z_pre_fed_rd_log"])].copy()
    save(main, "main_results.csv")

    # Simple descriptive bins.
    cx = crosswalk.dropna(subset=["pre_fed_share"]).copy()
    cx["fed_share_quartile"] = pd.qcut(cx["pre_fed_share"], 4, labels=[1,2,3,4], duplicates="drop")
    desc = cx.groupby("fed_share_quartile", observed=False).agg(
        institutions=("Institution", "nunique"), mean_fed_share=("pre_fed_share", "mean"),
        mean_fed_rd=("pre_fed_rd_mean", "mean")
    ).reset_index()
    save(desc, "exposure_quartiles.csv")

    lines = ["# Federal R&D exposure × Stanford–Roche analysis", "",
             f"Accepted NCSES matches: {int(crosswalk['accepted'].sum())} / {len(crosswalk)}.", "",
             "## Main 2004–2014 estimates (1 SD higher exposure × post-2009)"]
    for _, r in main.sort_values(["outcome","exposure"]).iterrows():
        lines.append(f"- {r['outcome']} | {r['exposure']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}, N={int(r['N'])}, universities={int(r['institutions'])}")
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote results to", OUTDIR)
    print(diagnostics.to_string(index=False))
    print(main[["outcome","exposure","coef","se","p","N","institutions"]].to_string(index=False))


if __name__ == "__main__":
    main()
