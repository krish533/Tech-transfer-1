from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rapidfuzz import fuzz, process

from run_stanford_roche_causal_test import build_language_measures, attach_language_to_panel

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = PACKAGE_ROOT / "data" / "derived" / "policy_level_indices_institution_year.csv"
SENTENCE_PATH = PACKAGE_ROOT / "data" / "derived" / "sentence_scores_canonical.csv"
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "federal_rd_exposure"
TOTAL_URL = "https://ncsesdata.nsf.gov/herd/2014/html/HERD2014_DST_17.html"
FED_URL = "https://ncsesdata.nsf.gov/herd/2014/html/HERD2014_DST_20.html"
PRE_YEARS = [2006, 2007, 2008]
MAIN_SHOCK = 2009
ALT_SHOCK = 2011


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def cluster_fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["Institution"]})


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cols = []
    for c in d.columns:
        parts = list(c) if isinstance(c, tuple) else [c]
        parts = [str(x) for x in parts if "Unnamed" not in str(x) and str(x) != "nan"]
        cols.append(" ".join(parts).strip())
    d.columns = cols
    return d


def numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(",", "", regex=False)
    x = x.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def read_ncses(url: str, prefix: str) -> pd.DataFrame:
    tables = pd.read_html(url)
    candidates = []
    for raw in tables:
        d = flatten_columns(raw)
        inst = [c for c in d.columns if "Institution" in c]
        if inst and sum(any(str(y) in c for c in d.columns) for y in PRE_YEARS) >= 2:
            candidates.append((len(d), d, inst[0]))
    if not candidates:
        raise ValueError(f"No suitable NCSES table found: {url}")
    _, d, inst_col = max(candidates, key=lambda z: z[0])
    out = pd.DataFrame({"ncses_name": d[inst_col].astype(str)})
    for y in PRE_YEARS:
        year_cols = [c for c in d.columns if re.search(rf"(?<!\d){y}(?!\d)", c)]
        if not year_cols:
            out[f"{prefix}_{y}"] = np.nan
            continue
        scored = [(numeric(d[c]).notna().sum(), c, numeric(d[c])) for c in year_cols]
        _, _, vals = max(scored, key=lambda z: z[0])
        out[f"{prefix}_{y}"] = vals
    out = out[~out["ncses_name"].str.contains("All institutions|All other surveyed", case=False, na=False)].copy()
    out = out.drop_duplicates("ncses_name")
    return out


def norm_name(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"\^\{[^}]*\}|\^[a-z]", " ", x)
    x = x.replace("&", " and ")
    x = re.sub(r"\bu\.\s*", " university ", x)
    x = re.sub(r"\buniv\.?\b", " university ", x)
    x = re.sub(r"\binst\.?\b", " institute ", x)
    x = re.sub(r"\btech\.?\b", " technology ", x)
    repl = {"st.": "state", "calif.": "california", "penn.": "pennsylvania", "wash.": "washington", "mass.": "massachusetts"}
    for a, b in repl.items():
        x = x.replace(a, b)
    x = re.sub(r"\bthe\b", " ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def build_ncses_exposure(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = read_ncses(TOTAL_URL, "total")
    fed = read_ncses(FED_URL, "fed")
    n = total.merge(fed, on="ncses_name", how="outer", validate="one_to_one")
    n["norm"] = n["ncses_name"].map(norm_name)
    n = n.sort_values("ncses_name").drop_duplicates("norm")

    institutions = sorted(panel.loc[panel["Year"] == 2008, "Institution"].dropna().unique())
    choices = n["norm"].tolist()
    rows = []
    for inst in institutions:
        key = norm_name(inst)
        exact = n[n["norm"] == key]
        if len(exact):
            match_norm, score = key, 100.0
        else:
            best = process.extractOne(key, choices, scorer=fuzz.token_set_ratio)
            match_norm, score = (best[0], float(best[1])) if best else (None, np.nan)
        if match_norm is None:
            rows.append({"Institution": inst, "policy_norm": key, "ncses_name": None, "match_score": np.nan, "accepted": 0})
            continue
        candidate = n[n["norm"] == match_norm].iloc[0]
        accepted = int(score >= 88)
        row = {"Institution": inst, "policy_norm": key, "ncses_name": candidate["ncses_name"], "match_score": score, "accepted": accepted}
        for y in PRE_YEARS:
            row[f"total_{y}"] = candidate.get(f"total_{y}", np.nan)
            row[f"fed_{y}"] = candidate.get(f"fed_{y}", np.nan)
        rows.append(row)
    m = pd.DataFrame(rows)
    for y in PRE_YEARS:
        m[f"share_{y}"] = m[f"fed_{y}"] / m[f"total_{y}"]
    m["federal_share_pre"] = m[[f"share_{y}" for y in PRE_YEARS]].mean(axis=1, skipna=True)
    m["federal_rd_pre"] = m[[f"fed_{y}" for y in PRE_YEARS]].mean(axis=1, skipna=True)
    m["total_rd_pre"] = m[[f"total_{y}" for y in PRE_YEARS]].mean(axis=1, skipna=True)
    m.loc[m["accepted"] == 0, ["federal_share_pre", "federal_rd_pre", "total_rd_pre"]] = np.nan
    valid = m.dropna(subset=["federal_share_pre"]).copy()
    valid = valid[(valid["federal_share_pre"] >= 0) & (valid["federal_share_pre"] <= 1.05)].copy()
    valid["federal_share_z"] = (valid["federal_share_pre"] - valid["federal_share_pre"].mean()) / valid["federal_share_pre"].std(ddof=0)
    valid["log_federal_rd"] = np.log1p(valid["federal_rd_pre"].clip(lower=0))
    valid["log_federal_rd_z"] = (valid["log_federal_rd"] - valid["log_federal_rd"].mean()) / valid["log_federal_rd"].std(ddof=0)
    med = valid["federal_share_pre"].median()
    q67 = valid["federal_share_pre"].quantile(2/3)
    valid["high_share_median"] = (valid["federal_share_pre"] >= med).astype(int)
    valid["high_share_toptercile"] = (valid["federal_share_pre"] >= q67).astype(int)
    return m, valid


def did(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exp: str, shock: int, start: int, end: int, observed_only: bool=False) -> dict:
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != shock)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d["post"] = (d["Year"] > shock).astype(int)
    d["exp_post"] = d[exp] * d["post"]
    d = d.dropna(subset=[outcome, exp, "exp_post"])
    if d["Institution"].nunique() < 15 or d[exp].std() == 0:
        return {"outcome":outcome,"exposure":exp,"shock":shock,"start":start,"end":end,"observed_only":observed_only,"N":len(d),"institutions":d["Institution"].nunique(),"coef":np.nan,"se":np.nan,"p":np.nan}
    fit = cluster_fit(f"{outcome} ~ exp_post + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["exp_post"]
    return {"outcome":outcome,"exposure":exp,"shock":shock,"start":start,"end":end,"observed_only":observed_only,"N":int(fit.nobs),"institutions":d["Institution"].nunique(),"coef":fit.params["exp_post"],"se":fit.bse["exp_post"],"p":fit.pvalues["exp_post"],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"r2":fit.rsquared}


def event_study(panel: pd.DataFrame, exposure: pd.DataFrame, outcome: str, exp: str="federal_share_z", shock: int=MAIN_SHOCK, start: int=2004, end: int=2015, ref: int=2008) -> tuple[pd.DataFrame,pd.DataFrame]:
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start,end) & (d["Year"] != shock)].dropna(subset=[outcome,exp]).copy()
    terms=[]; years=[]
    for y in range(start,end+1):
        if y in {ref,shock}: continue
        term=f"ev_{y}"; d[term]=(d["Year"]==y).astype(int)*d[exp]; terms.append(term); years.append(y)
    fit=cluster_fit(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)",d)
    rows=[]
    for y,t in zip(years,terms):
        ci=fit.conf_int().loc[t]
        rows.append({"outcome":outcome,"exposure":exp,"year":y,"event_time":y-shock,"coef":fit.params[t],"se":fit.bse[t],"p":fit.pvalues[t],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()})
    leads=[f"ev_{y}" for y in years if y<shock and y!=ref]
    hyp=" = 0, ".join(leads)+" = 0"
    test=fit.wald_test(hyp,scalar=True)
    pre=pd.DataFrame([{"outcome":outcome,"exposure":exp,"n_leads":len(leads),"stat":float(np.asarray(test.statistic).squeeze()),"p":float(np.asarray(test.pvalue).squeeze())}])
    return pd.DataFrame(rows),pre


def revision_hazard(panel: pd.DataFrame, exposure: pd.DataFrame, exp: str="federal_share_z") -> dict:
    d=panel.merge(exposure[["Institution",exp]],on="Institution",how="inner",validate="many_to_one")
    d=d[d["Year"].between(2004,2014)&(d["Year"]!=MAIN_SHOCK)].copy()
    d["revision"]=(d["Is_Carried_Forward"]==0).astype(int); d["post"]=(d["Year"]>MAIN_SHOCK).astype(int); d["exp_post"]=d[exp]*d["post"]
    fit=cluster_fit("revision ~ exp_post + C(Institution) + C(Year)",d)
    ci=fit.conf_int().loc["exp_post"]
    return {"outcome":"revision_hazard","exposure":exp,"coef":fit.params["exp_post"],"se":fit.bse["exp_post"],"p":fit.pvalues["exp_post"],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()}


def main() -> None:
    panel=pd.read_csv(PANEL_PATH,low_memory=False)
    panel=panel[(panel["Year"]>=1944)&(panel["Year"]<=2025)].copy()
    sentences=pd.read_csv(SENTENCE_PATH,low_memory=False)
    lang=build_language_measures(sentences)
    panel=attach_language_to_panel(panel,lang)
    matches,exposure=build_ncses_exposure(panel)
    save(matches,"ncses_match_audit.csv"); save(exposure,"federal_rd_exposure.csv")

    outcomes=["has_strong_assignment","strong_per_1000w","weak_per_1000w","net_assignment_strength","Mean_Tone_Score","Tone_Index","Legal_Load_Index","obligation_modal_share","restrictive_share","roche2011_per_1000w"]
    exposures=["federal_share_z","log_federal_rd_z","high_share_median","high_share_toptercile"]
    specs=[(MAIN_SHOCK,2004,2014),(MAIN_SHOCK,2006,2013),(MAIN_SHOCK,2000,2016),(ALT_SHOCK,2006,2016)]
    rows=[]
    for shock,start,end in specs:
        for exp in exposures:
            for out in outcomes:
                rows.append(did(panel,exposure,out,exp,shock,start,end,False))
    for exp in ["federal_share_z","high_share_median"]:
        for out in outcomes:
            rows.append(did(panel,exposure,out,exp,MAIN_SHOCK,2004,2014,True))
    dids=pd.DataFrame(rows); save(dids,"did_results.csv")

    evs=[]; pres=[]
    for out in outcomes:
        e,p=event_study(panel,exposure,out)
        evs.append(e); pres.append(p)
    save(pd.concat(evs,ignore_index=True),"event_study_results.csv")
    save(pd.concat(pres,ignore_index=True),"pretrend_joint_tests.csv")
    save(pd.DataFrame([revision_hazard(panel,exposure)]),"revision_hazard.csv")

    desc=pd.DataFrame([{
        "matched_policy_institutions":len(exposure),
        "mean_federal_share":exposure["federal_share_pre"].mean(),
        "median_federal_share":exposure["federal_share_pre"].median(),
        "sd_federal_share":exposure["federal_share_pre"].std(),
        "min_federal_share":exposure["federal_share_pre"].min(),
        "max_federal_share":exposure["federal_share_pre"].max(),
        "median_match_score":exposure["match_score"].median(),
    }]); save(desc,"exposure_summary.csv")

    mainres=dids[(dids["shock"]==MAIN_SHOCK)&(dids["start"]==2004)&(dids["end"]==2014)&(dids["exposure"]=="federal_share_z")&(dids["observed_only"]==False)]
    lines=["# Federal R&D exposure × Stanford–Roche exploratory results","",f"Matched institutions with usable pre-2009 federal-share exposure: {len(exposure)}.","","## Main continuous-exposure DiD (one-SD higher pre-2009 federal R&D share)"]
    for _,r in mainres.iterrows(): lines.append(f"- {r['outcome']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}, N={int(r['N'])}, universities={int(r['institutions'])}")
    preall=pd.concat(pres,ignore_index=True)
    lines += ["","## Joint pretrend tests"]
    for _,r in preall.iterrows(): lines.append(f"- {r['outcome']}: p={r['p']:.4f}")
    (OUTDIR/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("Matched institutions:",len(exposure)); print(mainres[["outcome","coef","se","p","N","institutions"]].to_string(index=False)); print(preall[["outcome","p"]].to_string(index=False))


if __name__ == "__main__":
    main()
