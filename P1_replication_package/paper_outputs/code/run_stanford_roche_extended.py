from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg

from run_stanford_roche_causal_test import build_language_measures, attach_language_to_panel
from run_federal_rd_exposure_clean import (
    PANEL_PATH, SENTENCE_PATH, OUTDIR as CLEAN_OUTDIR, SHOCK_YEAR,
    load_ncses, build_crosswalk, make_exposure, cluster_fit,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = PACKAGE_ROOT / "paper_outputs" / "tables" / "stanford_roche_extended"


def save(df: pd.DataFrame, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTDIR / name, index=False)


def build_data():
    panel = pd.read_csv(PANEL_PATH, low_memory=False)
    panel = panel[(panel["Year"] >= 1944) & (panel["Year"] <= 2025)].copy()
    sentences = pd.read_csv(SENTENCE_PATH, low_memory=False)
    panel = attach_language_to_panel(panel, build_language_measures(sentences))
    ncses = load_ncses()
    institutions = sorted(panel.loc[panel["Year"] == 2010, "Institution"].dropna().unique())
    cross = build_crosswalk(institutions, ncses, "principal_main")
    exp = make_exposure(cross)
    exp["log_total_rd"] = np.log1p(exp["total_2010"].clip(lower=0))
    exp["log_total_rd_z"] = (exp["log_total_rd"] - exp["log_total_rd"].mean()) / exp["log_total_rd"].std(ddof=0)
    return panel, exp


def did_single(panel, expdf, outcome, exposure, start=2007, end=2016, observed_only=False):
    d = panel.merge(expdf[["Institution", exposure]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].copy()
    if observed_only:
        d = d[d["Is_Carried_Forward"] == 0].copy()
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["xpost"] = d[exposure] * d["post"]
    d = d.dropna(subset=[outcome, exposure])
    fit = cluster_fit(f"{outcome} ~ xpost + C(Institution) + C(Year)", d)
    ci = fit.conf_int().loc["xpost"]
    return {"outcome":outcome,"exposure":exposure,"start":start,"end":end,"observed_only":observed_only,
            "coef":fit.params["xpost"],"se":fit.bse["xpost"],"p":fit.pvalues["xpost"],
            "ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()}


def did_two_exposures(panel, expdf, outcome, x1, x2, start=2007, end=2016):
    d = panel.merge(expdf[["Institution", x1, x2]], on="Institution", how="inner", validate="many_to_one")
    d = d[d["Year"].between(start, end) & (d["Year"] != SHOCK_YEAR)].dropna(subset=[outcome,x1,x2]).copy()
    d["post"] = (d["Year"] > SHOCK_YEAR).astype(int)
    d["x1post"] = d[x1] * d["post"]; d["x2post"] = d[x2] * d["post"]
    fit = cluster_fit(f"{outcome} ~ x1post + x2post + C(Institution) + C(Year)", d)
    rows=[]
    for term,var in [("x1post",x1),("x2post",x2)]:
        ci=fit.conf_int().loc[term]
        rows.append({"outcome":outcome,"model":f"{x1}+{x2}","term":var,"coef":fit.params[term],"se":fit.bse[term],"p":fit.pvalues[term],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1],"N":int(fit.nobs),"institutions":d["Institution"].nunique()})
    return rows


def event_study(panel, expdf, outcome, exposure, start=2006, end=2017, ref=2010):
    d=panel.merge(expdf[["Institution",exposure]],on="Institution",how="inner",validate="many_to_one")
    d=d[d["Year"].between(start,end)&(d["Year"]!=SHOCK_YEAR)].dropna(subset=[outcome,exposure]).copy()
    years=[]; terms=[]
    for y in range(start,end+1):
        if y in {ref,SHOCK_YEAR}: continue
        t=f"ev_{y}"; d[t]=(d["Year"]==y).astype(int)*d[exposure]; years.append(y); terms.append(t)
    fit=cluster_fit(f"{outcome} ~ {' + '.join(terms)} + C(Institution) + C(Year)",d)
    rows=[]
    for y,t in zip(years,terms):
        ci=fit.conf_int().loc[t]
        rows.append({"outcome":outcome,"exposure":exposure,"year":y,"event_time":y-SHOCK_YEAR,"coef":fit.params[t],"se":fit.bse[t],"p":fit.pvalues[t],"ci_low":ci.iloc[0],"ci_high":ci.iloc[1]})
    leads=[f"ev_{y}" for y in years if y<SHOCK_YEAR and y!=ref]
    wt=fit.wald_test(" = 0, ".join(leads)+" = 0",scalar=True)
    pre={"outcome":outcome,"exposure":exposure,"n_leads":len(leads),"stat":float(np.asarray(wt.statistic).squeeze()),"p":float(np.asarray(wt.pvalue).squeeze())}
    return pd.DataFrame(rows),pre


def placebo(panel, expdf, outcome, exposure, year):
    d=panel.merge(expdf[["Institution",exposure]],on="Institution",how="inner",validate="many_to_one")
    d=d[d["Year"].between(year-4,year+4)&(d["Year"]!=year)].dropna(subset=[outcome,exposure]).copy()
    d["post"]=(d["Year"]>year).astype(int); d["xpost"]=d[exposure]*d["post"]
    fit=cluster_fit(f"{outcome} ~ xpost + C(Institution) + C(Year)",d)
    return {"outcome":outcome,"exposure":exposure,"placebo_year":year,"coef":fit.params["xpost"],"se":fit.bse["xpost"],"p":fit.pvalues["xpost"],"N":int(fit.nobs)}


def descriptive_adoption(panel):
    observed=panel[panel["Is_Carried_Forward"]==0].copy()
    def period(y):
        if y <= 2008: return "through_2008"
        if y <= 2011: return "2009_2011"
        if y <= 2015: return "2012_2015"
        if y <= 2020: return "2016_2020"
        return "2021_2025"
    observed["period"]=observed["Year"].map(period)
    order=["through_2008","2009_2011","2012_2015","2016_2020","2021_2025"]
    out=(observed.groupby("period").agg(
        N=("Institution","size"), institutions=("Institution","nunique"),
        strong_share=("has_strong_assignment","mean"), weak_share=("has_weak_assignment","mean"),
        mean_strong_rate=("strong_per_1000w","mean"), mean_weak_rate=("weak_per_1000w","mean"),
        mean_net_strength=("net_assignment_strength","mean"), mean_pcsi=("Mean_Tone_Score","mean")
    ).reindex(order).reset_index())
    annual=(observed[observed["Year"]>=2000].groupby("Year").agg(
        observed_policies=("Institution","size"), strong_share=("has_strong_assignment","mean"),
        weak_share=("has_weak_assignment","mean"), net_strength=("net_assignment_strength","mean")
    ).reset_index())
    force=(panel[panel["Year"]>=2000].groupby("Year").agg(
        institutions=("Institution","size"), strong_share=("has_strong_assignment","mean"),
        weak_share=("has_weak_assignment","mean"), net_strength=("net_assignment_strength","mean")
    ).reset_index())
    return out,annual,force


def first_adoption(panel, expdf):
    base=panel[panel["Year"]==2010][["Institution","has_strong_assignment"]].merge(
        expdf[["Institution","log_federal_rd_z","federal_share_z","log_total_rd_z"]],on="Institution",how="inner")
    base=base[base["has_strong_assignment"]==0].copy()
    obs=panel[(panel["Is_Carried_Forward"]==0)&(panel["Year"]>=2012)][["Institution","Year","has_strong_assignment"]].copy()
    adopters=obs[obs["has_strong_assignment"]==1].groupby("Institution")["Year"].min().rename("adoption_year")
    a=base.merge(adopters,on="Institution",how="left")
    a["event"]=a["adoption_year"].notna().astype(int)
    a["duration"]=np.where(a["event"]==1,a["adoption_year"]-SHOCK_YEAR,2025-SHOCK_YEAR)
    a["duration"]=a["duration"].clip(lower=1)
    a["high_federal_volume"]=(a["log_federal_rd_z"]>=a["log_federal_rd_z"].median()).astype(int)
    summary=[]
    for g,df in a.groupby("high_federal_volume"):
        summary.append({"high_federal_volume":int(g),"N":len(df),"adopt_by_2015":(df["adoption_year"]<=2015).mean(),"adopt_by_2020":(df["adoption_year"]<=2020).mean(),"adopt_by_2025":df["event"].mean(),"median_adoption_year_among_adopters":df.loc[df["event"]==1,"adoption_year"].median()})
    cox_rows=[]
    for vars_,label in [(["log_federal_rd_z"],"federal_volume"),(["log_federal_rd_z","log_total_rd_z"],"federal_volume_plus_total_size"),(["federal_share_z","log_total_rd_z"],"federal_share_plus_total_size")]:
        X=a[vars_].astype(float)
        fit=PHReg(a["duration"].astype(float),X,status=a["event"].astype(int),ties="efron").fit(disp=0)
        for i,v in enumerate(vars_):
            cox_rows.append({"model":label,"term":v,"coef":fit.params[i],"se":fit.bse[i],"p":fit.pvalues[i],"hazard_ratio":np.exp(fit.params[i]),"N":len(a),"events":int(a["event"].sum())})
    # First actual post-shock policy revision conditional on having one.
    firstpost=obs.sort_values(["Institution","Year"]).groupby("Institution").first().reset_index()
    f=a.merge(firstpost[["Institution","Year","has_strong_assignment"]],on="Institution",how="inner",suffixes=("","_firstpost"))
    lpm_rows=[]
    for formula,label in [
        ("has_strong_assignment_firstpost ~ log_federal_rd_z","federal_volume"),
        ("has_strong_assignment_firstpost ~ log_federal_rd_z + log_total_rd_z","federal_volume_plus_total_size"),
        ("has_strong_assignment_firstpost ~ federal_share_z + log_total_rd_z","federal_share_plus_total_size")]:
        fit=smf.ols(formula,data=f).fit(cov_type="HC1")
        for term in fit.params.index:
            if term=="Intercept": continue
            lpm_rows.append({"model":label,"term":term,"coef":fit.params[term],"se":fit.bse[term],"p":fit.pvalues[term],"N":int(fit.nobs)})
    return a,pd.DataFrame(summary),pd.DataFrame(cox_rows),pd.DataFrame(lpm_rows)


def main():
    panel,expdf=build_data()
    outcomes=["has_strong_assignment","strong_per_1000w","weak_per_1000w","net_assignment_strength","Mean_Tone_Score","Tone_Index","Legal_Load_Index","obligation_modal_share","restrictive_share","roche2011_per_1000w"]

    # Add total R&D size to the saved exposure audit.
    save(expdf,"exposure_with_total_size.csv")

    single=[]; multi=[]; events=[]; pre=[]; placebos=[]
    for out in outcomes:
        for exposure in ["federal_share_z","log_federal_rd_z","log_total_rd_z"]:
            for start,end in [(2007,2016),(2009,2015),(2004,2018)]:
                single.append(did_single(panel,expdf,out,exposure,start,end,False))
            single.append(did_single(panel,expdf,out,exposure,2007,2016,True))
        multi += did_two_exposures(panel,expdf,out,"federal_share_z","log_total_rd_z")
        multi += did_two_exposures(panel,expdf,out,"log_federal_rd_z","log_total_rd_z")
        for exposure in ["federal_share_z","log_federal_rd_z","log_total_rd_z"]:
            ev,pr=event_study(panel,expdf,out,exposure); events.append(ev); pre.append(pr)
            for py in [2007,2015]: placebos.append(placebo(panel,expdf,out,exposure,py))
    save(pd.DataFrame(single),"single_exposure_did.csv")
    save(pd.DataFrame(multi),"two_exposure_did.csv")
    save(pd.concat(events,ignore_index=True),"event_studies.csv")
    save(pd.DataFrame(pre),"pretrend_tests.csv")
    save(pd.DataFrame(placebos),"placebo_tests.csv")

    bins,annual,force=descriptive_adoption(panel)
    save(bins,"observed_policy_periods.csv"); save(annual,"observed_policy_annual.csv"); save(force,"policy_in_force_annual.csv")
    adoption,adopt_summary,cox,lpm=first_adoption(panel,expdf)
    save(adoption,"first_adoption_data.csv"); save(adopt_summary,"first_adoption_summary.csv"); save(cox,"first_adoption_cox.csv"); save(lpm,"first_post_revision_lpm.csv")

    main=pd.DataFrame(single)
    main=main[(main["outcome"]=="net_assignment_strength")&(main["start"]==2007)&(main["end"]==2016)&(main["observed_only"]==False)]
    size=pd.DataFrame(multi); size=size[size["outcome"]=="net_assignment_strength"]
    ptest=pd.DataFrame(pre); ptest=ptest[ptest["outcome"]=="net_assignment_strength"]
    lines=["# Extended Stanford–Roche empirical results","","## Net assignment strength: main 2007-2016 DiD"]
    for _,r in main.iterrows(): lines.append(f"- {r['exposure']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}")
    lines += ["","## Net assignment strength: exposure decomposition with total R&D size"]
    for _,r in size.iterrows(): lines.append(f"- {r['model']} / {r['term']}: beta={r['coef']:.5f}, SE={r['se']:.5f}, p={r['p']:.4f}")
    lines += ["","## Net assignment strength: event-study pretrend tests"]
    for _,r in ptest.iterrows(): lines.append(f"- {r['exposure']}: p={r['p']:.4f}")
    lines += ["","## First adoption Cox models"]
    for _,r in cox.iterrows(): lines.append(f"- {r['model']} / {r['term']}: HR={r['hazard_ratio']:.3f}, p={r['p']:.4f}")
    (OUTDIR/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))
    print("\nObserved-policy periods:\n",bins.to_string(index=False))
    print("\nAdoption summary:\n",adopt_summary.to_string(index=False))


if __name__=="__main__":
    main()
