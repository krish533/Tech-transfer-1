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
TOTAL_URL = "https://ncses.nsf.gov/pubs/nsf21314/assets/data-tables/tables/nsf21314-tab020.xlsx"
FED_URL = "https://ncses.nsf.gov/pubs/nsf21314/assets/data-tables/tables/nsf21314-tab023.xlsx"
EXPOSURE_YEAR = 2010
SHOCK_YEAR = 2011


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def cluster_fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["Institution"]})


def numeric(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.replace(",", "", regex=False)
    x = x.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(x, errors="coerce")


def read_rank_xlsx(url: str, prefix: str) -> pd.DataFrame:
    raw = pd.read_excel(url, header=None, engine="openpyxl")
    header_row = None
    for i in range(min(25, len(raw))):
        vals = [str(v) for v in raw.iloc[i].tolist()]
        if any("Institution" in str(v) for v in vals) and any("2010" in str(v) for v in vals):
            header_row = i
            break
    if header_row is None:
        # Some NCSES workbooks use a two-line header. Locate Institution first and combine it with next row.
        for i in range(min(25, len(raw) - 1)):
            vals = [str(v) for v in raw.iloc[i].tolist()]
            next_vals = [str(v) for v in raw.iloc[i + 1].tolist()]
            if any("Institution" in str(v) for v in vals) and any("2010" in str(v) for v in next_vals):
                header_row = i
                combined = []
                for a, b in zip(vals, next_vals):
                    aa = "" if a.lower() == "nan" else a.strip()
                    bb = "" if b.lower() == "nan" else b.strip()
                    combined.append((aa + " " + bb).strip())
                header = combined
                d = raw.iloc[i + 2 :].copy()
                break
        else:
            raise ValueError(f"Could not locate header row in {url}; preview={raw.head(12).astype(str).to_dict(orient='split')}")
    else:
        header = [str(v).strip() for v in raw.iloc[header_row].tolist()]
        d = raw.iloc[header_row + 1 :].copy()
    # Ensure unique column labels for pandas 3.x.
    seen = {}
    unique_header = []
    for j, h in enumerate(header):
        h = h if h and h.lower() != "nan" else f"col_{j}"
        n = seen.get(h, 0)
        seen[h] = n + 1
        unique_header.append(h if n == 0 else f"{h}_{n}")
    d.columns = unique_header
    inst_candidates = [c for c in d.columns if "Institution" in str(c)]
    year_candidates = [c for c in d.columns if re.search(r"(?<!\d)2010(?!\d)", str(c))]
    if not inst_candidates or not year_candidates:
        raise ValueError(f"Institution/2010 columns not found in {url}: {list(d.columns)}")
    inst_col = inst_candidates[0]
    scored = [(numeric(d[c]).notna().sum(), c) for c in year_candidates]
    _, year_col = max(scored, key=lambda z: z[0])
    out = pd.DataFrame({"ncses_name": d[inst_col].astype(str), f"{prefix}_2010": numeric(d[year_col])})
    out = out[~out["ncses_name"].str.contains("All institutions|All other surveyed", case=False, na=False)]
    return out.dropna(subset=[f"{prefix}_2010"]).drop_duplicates("ncses_name")


def norm_name(x: str) -> str:
    x = str(x).lower()
    x = re.sub(r"\^\{[^}]*\}|\^[a-z]", " ", x)
    x = x.replace("&", " and ")
    x = re.sub(r"\bu\.\s*", " university ", x)
    x = re.sub(r"\buniv\.?\b", " university ", x)
    x = re.sub(r"\binst\.?\b", " institute ", x)
    x = re.sub(r"\btech\.?\b", " technology ", x)
    x = x.replace("st.", "state").replace("calif.", "california").replace("penn.", "pennsylvania")
    x = x.replace("wash.", "washington").replace("mass.", "massachusetts")
    x = re.sub(r"\bthe\b", " ", x)
    x = re.sub(r"[^a-z0-9]+", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def build_exposure(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = read_rank_xlsx(TOTAL_URL, "total")
    fed = read_rank_xlsx(FED_URL, "fed")
    n = total.merge(fed, on="ncses_name", how="inner", validate="one_to_one")
    n["norm"] = n["ncses_name"].map(norm_name)
    n = n.sort_values("ncses_name").drop_duplicates("norm")
    institutions = sorted(panel.loc[panel["Year"] == EXPOSURE_YEAR, "Institution"].dropna().unique())
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
        accepted = int(score >= 90)
        rows.append({"Institution": inst, "policy_norm": key, "ncses_name": candidate["ncses_name"], "match_score": score,
                     "accepted": accepted, "total_2010": candidate["total_2010"], "fed_2010": candidate["fed_2010"]})
    m = pd.DataFrame(rows)
    m["federal_share_2010"] = m["fed_2010"] / m["total_2010"]
    m.loc[m["accepted"] == 0, "federal_share_2010"] = np.nan
    v = m.dropna(subset=["federal_share_2010"]).copy()
    v = v[(v["federal_share_2010"] >= 0) & (v["federal_share_2010"] <= 1.05)].copy()
    v["federal_share_z"] = (v["federal_share_2010"] - v["federal_share_2010"].mean()) / v["federal_share_2010"].std(ddof=0)
    v["log_federal_rd"] = np.log1p(v["fed_2010"].clip(lower=0))
    v["log_federal_rd_z"] = (v["log_federal_rd"] - v["log_federal_rd"].mean()) / v["log_federal_rd"].std(ddof=0)
    v["high_share_median"] = (v["federal_share_2010"] >= v["federal_share_2010"].median()).astype(int)
    v["high_share_toptercile"] = (v["federal_share_2010"] >= v["federal_share_2010"].quantile(2/3)).astype(int)
    return m, v


def did(panel, exposure, outcome, exp, start, end, observed_only=False):
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["exp_post"] = d[exp] * d["post"]
    d = d.dropna(subset=[outcome, exp, "exp_post"])
    if d["Institution"].nunique() < 15 or d[exp].std() == 0:
        return {"outcome":outcome,"exposure":exp,"start":start,"end":end,"observed_only":observed_only,"N":len(d),"institutions":d["Institution"].nunique(),"coef":np.nan,"se":np.nan,"p":np.nan}
    fit = cluster_fit(f"{outcome} ~ exp_post + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["exp_post"]
    return {"outcome":outcome,"exposure":exp,"start":start,"end":end,"observed_only":observed_only,
            "N":int(fit.nobs),"institutions":d["Institution"].nunique(),"coef":fit.params["exp_post"],"se":fit.bse["exp_post"],
            "p":fit.pvalues["exp_post"],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"r2":fit.rsquared}


def event_study(panel, exposure, outcome, exp="federal_share_z", start=2006, end=2017, ref=2010):
    d = panel.merge(exposure[["Institution", exp]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start,end) & (d["Year"] != SHOCK_YEAR)].dropna(subset=[outcome,exp]).copy()
    terms=[]; years=[]
    for y in range(start,end+1):
        if y in {ref,SHOCK_YEAR}: continue
        term=f"ev_{y}"; d[term]=(d["Year"]==y).astype(int)*d[exp]; terms.append(term); years.append(y)
    fit=cluster_fit(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)",d)
    rows=[]
    for y,t in zip(years,terms):
        ci=fit.conf_int().loc[t]
        rows.append({"outcome":outcome,"year":y,"event_time":y-SHOCK_YEAR,"coef":fit.params[t],"se":fit.bse[t],"p":fit.pvalues[t],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()})
    leads=[f"ev_{y}" for y in years if y<SHOCK_YEAR and y!=ref]
    test=fit.wald_test(" = 0, ".join(leads)+" = 0",scalar=True)
    pre=pd.DataFrame([{"outcome":outcome,"n_leads":len(leads),"stat":float(np.asarray(test.statistic).squeeze()),"p":float(np.asarray(test.pvalue).squeeze())}])
    return pd.DataFrame(rows),pre


def revision_hazard(panel, exposure):
    d=panel.merge(exposure[["Institution","federal_share_z"]],on="Institution",how="inner",validate="many_to_one")
    d=d[d["Year"].between(2007,2016)&(d["Year"]!=SHOCK_YEAR)].copy()
    d["revision"]=(d["Is_Carried_Forward"]==0).astype(int); d["post"]=(d["Year"]>SHOCK_YEAR).astype(int); d["exp_post"]=d["federal_share_z"]*d["post"]
    fit=cluster_fit("revision ~ exp_post + C(Institution) + C(Year)",d); ci=fit.conf_int().loc["exp_post"]
    return {"outcome":"revision_hazard","coef":fit.params["exp_post"],"se":fit.bse["exp_post"],"p":fit.pvalues["exp_post"],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()}


def main():
    panel=pd.read_csv(PANEL_PATH,low_memory=False)
    panel=panel[(panel["Year"]>=1944)&(panel["Year"]<=2025)].copy()
    sentences=pd.read_csv(SENTENCE_PATH,low_memory=False)
    panel=attach_language_to_panel(panel,build_language_measures(sentences))
    matches,exposure=build_exposure(panel)
    save(matches,"ncses_match_audit.csv"); save(exposure,"federal_rd_exposure.csv")

    outcomes=["has_strong_assignment","strong_per_1000w","weak_per_1000w","net_assignment_strength","Mean_Tone_Score","Tone_Index","Legal_Load_Index","obligation_modal_share","restrictive_share","roche2011_per_1000w"]
    exposures=["federal_share_z","log_federal_rd_z","high_share_median","high_share_toptercile"]
    windows=[(2007,2016),(2009,2015),(2004,2018)]
    rows=[]
    for start,end in windows:
        for exp in exposures:
            for out in outcomes:
                rows.append(did(panel,exposure,out,exp,start,end,False))
    for exp in ["federal_share_z","high_share_median"]:
        for out in outcomes:
            rows.append(did(panel,exposure,out,exp,2007,2016,True))
    dids=pd.DataFrame(rows); save(dids,"did_results.csv")

    evs=[]; pres=[]
    for out in outcomes:
        e,p=event_study(panel,exposure,out); evs.append(e); pres.append(p)
    save(pd.concat(evs,ignore_index=True),"event_study_results.csv")
    pre=pd.concat(pres,ignore_index=True); save(pre,"pretrend_joint_tests.csv")
    save(pd.DataFrame([revision_hazard(panel,exposure)]),"revision_hazard.csv")

    desc=pd.DataFrame([{"matched_policy_institutions":len(exposure),"mean_federal_share":exposure["federal_share_2010"].mean(),
                        "median_federal_share":exposure["federal_share_2010"].median(),"sd_federal_share":exposure["federal_share_2010"].std(),
                        "min_federal_share":exposure["federal_share_2010"].min(),"max_federal_share":exposure["federal_share_2010"].max(),
                        "median_match_score":exposure["match_score"].median()}]); save(desc,"exposure_summary.csv")

    mainres=dids[(dids["start"]==2007)&(dids["end"]==2016)&(dids["exposure"]=="federal_share_z")&(dids["observed_only"]==False)]
    lines=["# 2010 Federal R&D exposure × 2011 Stanford–Roche Supreme Court decision","",f"Usable matched universities: {len(exposure)}.","","## Main DiD: one-SD higher 2010 federal R&D share"]
    for _,r in mainres.iterrows(): lines.append(f"- {r['outcome']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}, N={int(r['N'])}, universities={int(r['institutions'])}")
    lines += ["","## Joint pretrend tests"]
    for _,r in pre.iterrows(): lines.append(f"- {r['outcome']}: p={r['p']:.4f}")
    (OUTDIR/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("Usable matched institutions:",len(exposure)); print(mainres[["outcome","coef","se","p","N","institutions"]].to_string(index=False)); print(pre[["outcome","p"]].to_string(index=False))


if __name__ == "__main__":
    main()
