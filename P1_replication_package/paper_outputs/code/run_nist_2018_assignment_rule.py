from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from run_federal_rd_exposure_clean import SINGLE, AGGREGATES_MAIN, EXCLUDE, canon

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PACKAGE_ROOT / "data" / "derived" / "policy_level_indices_institution_year.csv"
SENTENCE_PATH = PACKAGE_ROOT / "data" / "derived" / "sentence_scores_canonical.csv"
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "nist_2018_assignment_rule"
TOTAL_URL = "https://ncses.nsf.gov/pubs/nsf26304/assets/data-tables/tables/nsf26304-tab013.xlsx"
FED_URL = "https://ncses.nsf.gov/pubs/nsf26304/assets/data-tables/tables/nsf26304-tab016.xlsx"
SHOCK_YEAR = 2018
PRE_YEARS = [2015, 2016, 2017]

ASSIGN_RE = re.compile(r"\bassign(?:s|ed|ing|ment)?\b", re.I)
OBLIG_ASSIGN_RE = re.compile(
    r"\b(?:hereby\s+assign(?:s)?|does\s+hereby\s+assign|agree(?:s|d)?\s+to\s+assign|shall\s+assign|will\s+assign|must\s+assign|required\s+to\s+assign|obligat(?:e|ed)\s+to\s+assign)\b",
    re.I,
)
RIGHT_TITLE_INTEREST_RE = re.compile(r"\bright\b.{0,80}\btitle\b.{0,80}\binterest\b", re.I)
SUBJECT_INV_RE = re.compile(r"\bsubject\s+invention(?:s)?\b|\bbayh[- ]dole\b|\bfunding\s+agreement\b|\bfederally\s+funded\b", re.I)
WRITTEN_AGREEMENT_RE = re.compile(r"\bwritten\s+agreement\b|\bemployment\s+agreement\b|\bpatent\s+agreement\b", re.I)


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(",", "", regex=False).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def read_rank_table(url: str, prefix: str) -> pd.DataFrame:
    raw = pd.read_excel(url, header=None, engine="openpyxl")
    header_row = None
    for i in range(min(25, len(raw))):
        vals = [str(v) for v in raw.iloc[i].tolist()]
        if any("Institution" in v for v in vals) and all(any(str(y) in v for v in vals) for y in PRE_YEARS):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not locate header in {url}")
    header = [str(v).strip() for v in raw.iloc[header_row].tolist()]
    seen, cols = {}, []
    for j, h in enumerate(header):
        h = h if h and h.lower() != "nan" else f"col_{j}"
        n = seen.get(h, 0); seen[h] = n + 1
        cols.append(h if n == 0 else f"{h}_{n}")
    d = raw.iloc[header_row + 1:].copy(); d.columns = cols
    inst_col = next(c for c in d.columns if "Institution" in c)
    out = pd.DataFrame({"ncses_name": d[inst_col].astype(str)})
    for y in PRE_YEARS:
        ycands = [c for c in d.columns if re.search(rf"(?<!\d){y}(?!\d)", c)]
        if not ycands:
            raise ValueError(f"No {y} column in {url}")
        scored = [(numeric(d[c]).notna().sum(), c) for c in ycands]
        _, yc = max(scored, key=lambda z: z[0])
        out[f"{prefix}_{y}"] = numeric(d[yc])
    out = out[~out["ncses_name"].str.contains("All institutions|All other surveyed", case=False, na=False)]
    return out.dropna(subset=[f"{prefix}_{y}" for y in PRE_YEARS], how="all").drop_duplicates("ncses_name")


def component_sum(n: pd.DataFrame, names: list[str], prefix: str, year: int) -> float:
    name_set = set(n["ncses_name"])
    missing = [x for x in names if x not in name_set]
    if missing:
        raise ValueError(f"Missing declared NCSES components: {missing}")
    return float(n.loc[n["ncses_name"].isin(names), f"{prefix}_{year}"].sum())


def build_exposure(policy_institutions: list[str]) -> pd.DataFrame:
    total = read_rank_table(TOTAL_URL, "total")
    fed = read_rank_table(FED_URL, "fed")
    n = total.merge(fed, on="ncses_name", validate="one_to_one")
    n["canon"] = n["ncses_name"].map(canon)
    canon_groups = n.groupby("canon")["ncses_name"].apply(list).to_dict()
    rows = []
    for inst in policy_institutions:
        if inst in EXCLUDE:
            continue
        method = None; names = None
        if inst in AGGREGATES_MAIN:
            names = AGGREGATES_MAIN[inst]; method = "aggregate"
        elif inst in SINGLE:
            names = [SINGLE[inst]]; method = "manual"
        else:
            hits = canon_groups.get(canon(inst), [])
            if len(hits) == 1:
                names = [hits[0]]; method = "canonical_exact"
        if not names:
            continue
        row = {"Institution": inst, "match_method": method, "ncses_components": " | ".join(names)}
        ok = True
        for y in PRE_YEARS:
            try:
                row[f"total_{y}"] = component_sum(n, names, "total", y)
                row[f"fed_{y}"] = component_sum(n, names, "fed", y)
            except ValueError:
                ok = False; break
        if ok:
            rows.append(row)
    x = pd.DataFrame(rows)
    x["avg_total_pre"] = x[[f"total_{y}" for y in PRE_YEARS]].mean(axis=1)
    x["avg_fed_pre"] = x[[f"fed_{y}" for y in PRE_YEARS]].mean(axis=1)
    x["fed_share_pre"] = x[[f"fed_{y}" for y in PRE_YEARS]].sum(axis=1) / x[[f"total_{y}" for y in PRE_YEARS]].sum(axis=1)
    x["log_total_pre"] = np.log1p(x["avg_total_pre"])
    x["log_fed_pre"] = np.log1p(x["avg_fed_pre"])
    for c in ["fed_share_pre", "log_total_pre", "log_fed_pre"]:
        x[c + "_z"] = (x[c] - x[c].mean()) / x[c].std(ddof=0)
    x["high_fed_share"] = (x["fed_share_pre"] >= x["fed_share_pre"].median()).astype(int)
    x["high_fed_volume"] = (x["avg_fed_pre"] >= x["avg_fed_pre"].median()).astype(int)
    return x


def build_outcomes(sentences: pd.DataFrame) -> pd.DataFrame:
    s = sentences.copy()
    text = s["Sentence_Cleaned"].fillna("").astype(str)
    s["words"] = text.str.split().str.len().clip(lower=1)
    s["assignment_obligation"] = text.str.contains(OBLIG_ASSIGN_RE).astype(int)
    s["right_title_interest"] = (text.str.contains(ASSIGN_RE) & text.str.contains(RIGHT_TITLE_INTEREST_RE)).astype(int)
    s["federal_assignment"] = (text.str.contains(ASSIGN_RE) & text.str.contains(SUBJECT_INV_RE)).astype(int)
    s["written_assignment"] = (text.str.contains(ASSIGN_RE) & text.str.contains(WRITTEN_AGREEMENT_RE)).astype(int)
    g = s.groupby(["Institution", "Year"], as_index=False).agg(
        words=("words", "sum"),
        assignment_obligation_count=("assignment_obligation", "sum"),
        rti_assignment_count=("right_title_interest", "sum"),
        federal_assignment_count=("federal_assignment", "sum"),
        written_assignment_count=("written_assignment", "sum"),
    )
    for c in ["assignment_obligation", "rti_assignment", "federal_assignment", "written_assignment"]:
        g[f"{c}_per_1000w"] = 1000 * g[f"{c}_count"] / g["words"]
        g[f"has_{c}"] = (g[f"{c}_count"] > 0).astype(int)
    return g


def attach(panel: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(obs.rename(columns={"Year": "Source_Year"}), on=["Institution", "Source_Year"], how="left", validate="many_to_one")


def fit_cluster(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["Institution"]})


def did(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exp: str, start: int, end: int, observed_only: bool=False) -> dict:
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["xp"] = d[exp] * d["post"]
    d = d.dropna(subset=[outcome, exp])
    if d["Institution"].nunique() < 20 or d["post"].nunique() < 2:
        return {"outcome": outcome, "exposure": exp, "start": start, "end": end, "observed_only": observed_only, "coef": np.nan, "se": np.nan, "p": np.nan, "N": len(d), "institutions": d["Institution"].nunique()}
    fit = fit_cluster(f"{outcome} ~ xp + C(Institution) + C(Year)", d)
    return {"outcome": outcome, "exposure": exp, "start": start, "end": end, "observed_only": observed_only, "coef": float(fit.params["xp"]), "se": float(fit.bse["xp"]), "p": float(fit.pvalues["xp"]), "N": int(fit.nobs), "institutions": int(d["Institution"].nunique())}


def event_study(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exp: str, start: int=2013, end: int=2025, ref: int=2017):
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy().dropna(subset=[outcome, exp])
    years = [y for y in range(start, end + 1) if y not in {ref, SHOCK_YEAR}]
    terms=[]
    for y in years:
        t=f"evt_{y}"; d[t] = (d["Year"].eq(y).astype(int) * d[exp]); terms.append(t)
    fit = fit_cluster(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)", d)
    rows=[]
    for y,t in zip(years,terms):
        rows.append({"outcome":outcome,"exposure":exp,"year":y,"event_time":y-SHOCK_YEAR,"coef":float(fit.params[t]),"se":float(fit.bse[t]),"p":float(fit.pvalues[t])})
    leads=[f"evt_{y}" for y in years if y<SHOCK_YEAR]
    hypothesis=" = 0, ".join(leads)+" = 0"
    test=fit.wald_test(hypothesis, scalar=True)
    pre={"outcome":outcome,"exposure":exp,"pretrend_p":float(np.asarray(test.pvalue).squeeze()),"n_leads":len(leads)}
    return pd.DataFrame(rows), pre


def placebo(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exp: str, fake_shock: int) -> dict:
    d = panel.merge(exposure[["Institution",exp]], on="Institution", how="inner", validate="many_to_one")
    d=d[d["Year"].between(2012,2017) & (d["Year"]!=fake_shock)].copy().dropna(subset=[outcome,exp])
    d["post"]=(d["Year"]>fake_shock).astype(int); d["xp"]=d[exp]*d["post"]
    fit=fit_cluster(f"{outcome} ~ xp + C(Institution)+C(Year)",d)
    return {"outcome":outcome,"exposure":exp,"fake_shock":fake_shock,"coef":float(fit.params["xp"]),"se":float(fit.bse["xp"]),"p":float(fit.pvalues["xp"]),"N":int(fit.nobs)}


def main():
    panel=pd.read_csv(PANEL_PATH,low_memory=False)
    sentences=pd.read_csv(SENTENCE_PATH,low_memory=False)
    exposure=build_exposure(sorted(panel["Institution"].unique()))
    obs=build_outcomes(sentences)
    p=attach(panel,obs)
    outcomes=[
        "assignment_obligation_per_1000w","has_assignment_obligation",
        "rti_assignment_per_1000w","has_rti_assignment",
        "federal_assignment_per_1000w","has_federal_assignment",
        "written_assignment_per_1000w","has_written_assignment",
        "Mean_Tone_Score","Legal_Load_Index","obligation_modal_share","roche2011_per_1000w"
    ]
    exposures=["fed_share_pre_z","log_fed_pre_z","log_total_pre_z","high_fed_share","high_fed_volume"]
    windows=[(2014,2024),(2013,2025),(2015,2023)]
    rows=[]
    for start,end in windows:
        for exp in exposures:
            for outcome in outcomes:
                rows.append(did(p,exposure,outcome,exp,start,end,False))
    for exp in ["fed_share_pre_z","log_fed_pre_z"]:
        for outcome in outcomes[:8]:
            rows.append(did(p,exposure,outcome,exp,2013,2025,True))
    did_df=pd.DataFrame(rows); save(did_df,"did_results.csv")

    evs=[]; pre=[]
    for exp in ["fed_share_pre_z","log_fed_pre_z","log_total_pre_z"]:
        for outcome in ["assignment_obligation_per_1000w","rti_assignment_per_1000w","federal_assignment_per_1000w","has_assignment_obligation","Mean_Tone_Score"]:
            e,pr=event_study(p,exposure,outcome,exp); evs.append(e); pre.append(pr)
    save(pd.concat(evs,ignore_index=True),"event_study.csv"); save(pd.DataFrame(pre),"pretrend_tests.csv")

    pls=[]
    for fake in [2014,2015,2016]:
        for exp in ["fed_share_pre_z","log_fed_pre_z","log_total_pre_z"]:
            for outcome in ["assignment_obligation_per_1000w","rti_assignment_per_1000w","federal_assignment_per_1000w"]:
                pls.append(placebo(p,exposure,outcome,exp,fake))
    save(pd.DataFrame(pls),"placebos.csv")
    save(exposure,"exposure_crosswalk.csv")
    period=p[p["Year"].between(2013,2025) & (p["Is_Carried_Forward"]==0)].copy()
    period["period"]=pd.cut(period["Year"],[2012,2015,2017,2018,2021,2025],labels=["2013_2015","2016_2017","2018","2019_2021","2022_2025"])
    summary=period.groupby("period",observed=True).agg(N=("Institution","size"),institutions=("Institution","nunique"),assign_rate=("assignment_obligation_per_1000w","mean"),rti_rate=("rti_assignment_per_1000w","mean"),fed_assign_rate=("federal_assignment_per_1000w","mean"),pcsi=("Mean_Tone_Score","mean")).reset_index()
    save(summary,"observed_period_summary.csv")

    main=did_df[(did_df.start==2014)&(did_df.end==2024)&(~did_df.observed_only)&(did_df.exposure.isin(["fed_share_pre_z","log_fed_pre_z","log_total_pre_z"]))]
    print("Matched institutions:",len(exposure))
    print("Exposure correlations:")
    print(exposure[["fed_share_pre_z","log_fed_pre_z","log_total_pre_z"]].corr().to_string())
    print("\nMain 2014-2024 results:")
    print(main[["outcome","exposure","coef","se","p","N","institutions"]].to_string(index=False))
    print("\nPretrend tests:")
    print(pd.DataFrame(pre).to_string(index=False))
    print("\nObserved-policy period summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
