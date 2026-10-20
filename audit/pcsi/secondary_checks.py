#!/usr/bin/env python3
"""Independent *secondary* PCSI checks. No published output files are imported.
Reads canonical panel and complete sentence dataset, saves reproducible CSV/JSON.
"""
from pathlib import Path
import json,itertools,math
import numpy as np,pandas as pd
import statsmodels.formula.api as smf

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/"P1_replication_package"
OUT=P/"paper_outputs"/"tables"/"independent_pcsi_audit"
OUT.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(P/"data/derived/policy_level_indices_institution_year.csv",low_memory=False)
p=p[p.Year.between(1944,2025)].copy()
s=pd.read_csv(P/"data/derived/sentence_scores_canonical.csv",low_memory=False)
s=s[s.Year.between(1944,2025)].copy()
assert len(p)==4277 and len(s)==87159
obs=p[p.Is_Carried_Forward.eq(0)].copy()
res={"panel_n":len(p),"observed_n":len(obs),"n_sentences":len(s)}
for name,d in [("all",p),("observed",obs)]:
    x=d.Mean_Tone_Score
    mu=x.mean()
    between=((d.groupby("Institution").Mean_Tone_Score.transform("mean")-mu)**2).sum()
    variance=((x-mu)**2).sum()
    res[name+"_between_share"]=float(between/variance)
    res[name+"_mean"]=float(mu)
    res[name+"_sd"]=float(x.std())
    res[name+"_below_midpoint"]=float((x<.5).mean())
# Transparent Shapley of R-squared. All predictor subsets receive exact OLS.
for name,d in [("all",p),("observed",obs)]:
    xs=["Tone_Index","Clarity_Index","Legal_Load_Index"]
    t=d[["Mean_Tone_Score"]+xs].dropna()
    r2={frozenset():0.0}
    for n in range(1,len(xs)+1):
        for subset in itertools.combinations(xs,n):
            key=frozenset(subset)
            r2[key]=smf.ols("Mean_Tone_Score ~ "+" + ".join(subset),t).fit().rsquared
    tot=r2[frozenset(xs)]
    vals={}
    for v in xs:
        incr=0
        for k in range(len(xs)):
            for others in itertools.combinations([x for x in xs if x!=v],k):
                base=frozenset(others)
                w=math.factorial(k)*math.factorial(len(xs)-k-1)/math.factorial(len(xs))
                incr+=w*(r2[base|{v}]-r2[base])
        vals[v]=float(incr/tot)
    res[name+"_full_R2"]=float(tot)
    res[name+"_Shapley_shares"]=vals
# Temporal regressions from observed policy-years; FE and university-clustered SE.
trends=[]
for start in [1980,1990,2000]:
    sub=obs[obs.Year.ge(start)].copy()
    mod=smf.ols("Mean_Tone_Score ~ Year + C(Institution)",sub).fit(cov_type="cluster",cov_kwds={"groups":sub.Institution})
    trends.append(dict(start=start,N=int(mod.nobs),universities=int(sub.Institution.nunique()),coef=float(mod.params["Year"]),
                       se=float(mod.bse["Year"]),p=float(mod.pvalues["Year"])))
res["observed_FE_trends"]=trends
# Sentence deontic cells from published *coding rules*, independent variance arithmetic
# (rules are not replaced with inferred codes from the saved output tables).
import sys
sys.path.insert(0,str(P/"paper_outputs"/"code"))
from generate_revision_analyses import tag_deontic,tag_topics
d=tag_deontic(s.copy())
g=d.groupby("Deontic x sanction").ToneScore_0_1
means=g.transform("mean")
res["sentence_rule_form_R2"]=float(((means-s.ToneScore_0_1.mean())**2).sum()/((s.ToneScore_0_1-s.ToneScore_0_1.mean())**2).sum())
res["obligation_sentence_mean"]=float(d.loc[d.Deontic.eq("Obligation"),"ToneScore_0_1"].mean())
res["no_operator_sentence_mean"]=float(d.loc[d.Deontic.eq("No deontic operator"),"ToneScore_0_1"].mean())
res["obligation_sentence_share"]=float(d.Deontic.eq("Obligation").mean())
t=tag_topics(s.copy())
sub=t[t.Year.ge(1980)]
res["since_1980_share_topics_matched"]=float(sub["Primary topic"].ne("Other / general").mean())
# Re-estimate multiple topic-specific source-policy-year university-FE trends;
# exclude singleton universities to match the published reference specification.
trendrows=[]
for c in ["Ownership & assignment","Disclosure","Revenue sharing","Administration & disputes","Other / general"]:
    m=sub["Primary topic"].eq(c) if c=="Other / general" else sub[c]
    g=sub[m].groupby(["Institution","Year"],as_index=False).ToneScore_0_1.mean()
    g=g[g.groupby("Institution").Year.transform("nunique").gt(1)]
    fit=smf.ols("ToneScore_0_1 ~ Year + C(Institution)",g).fit(cov_type="cluster",cov_kwds={"groups":g.Institution})
    trendrows.append(dict(topic=c,coef=float(fit.params["Year"]),se=float(fit.bse["Year"]),p=float(fit.pvalues["Year"]),n=int(fit.nobs),university_count=int(g.Institution.nunique())))
res["topic_FE_trends"]=trendrows
checks={
 "panel_between":abs(res["all_between_share"]-0.61008404)<1e-7,
 "observed_between":abs(res["observed_between_share"]-0.48689514)<1e-7,
 "panel_full_R2":abs(res["all_full_R2"]-0.23628378)<1e-7,
 "observed_full_R2":abs(res["observed_full_R2"]-0.15074460)<1e-7,
 "tone_shapley":abs(res["all_Shapley_shares"]["Tone_Index"]-0.88617675)<1e-6,
 "observed_1980_FE_slope":abs(trends[0]["coef"]+0.00117393)<1e-7,
 "observed_1980_FE_se":abs(trends[0]["se"]-0.00039396)<2e-6,
 "sentence_deontic_R2":abs(res["sentence_rule_form_R2"]-0.04789042)<2e-6,
 "obligation_mean":abs(res["obligation_sentence_mean"]-0.329192)<2e-5,
 "topic_ownership_FE":abs(trendrows[0]["coef"]+0.000311)<1e-5,
 "topic_disclosure_FE":abs(trendrows[1]["coef"]+0.002227)<1e-5,
}
res["checks"]=checks
(OUT/"secondary_checks.json").write_text(json.dumps(res,indent=2)+"\n")
pd.DataFrame(trends).to_csv(OUT/"observed_fixed_effects_reproduced.csv",index=False)
pd.DataFrame(trendrows).to_csv(OUT/"topic_fixed_effects_reproduced.csv",index=False)
print("BEGIN_SECONDARY_AUDIT_JSON")
print(json.dumps(res,indent=2))
print("END_SECONDARY_AUDIT_JSON")
print("PASSED",sum(checks.values()),"OF",len(checks))
assert all(checks.values()),"Mismatch vs previously independently reported results"
