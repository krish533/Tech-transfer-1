#!/usr/bin/env python3
"""PCSI canonical scored sentences -> directly observed policy-years -> carried-forward panel.
Independent checks against original source-level CSV; never updates canonical data.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]/"P1_replication_package"
OUT=ROOT/"paper_outputs"/"tables"/"independent_pcsi_audit"
OUT.mkdir(exist_ok=True,parents=True)
s=pd.read_csv(ROOT/"data/derived/sentence_scores_canonical.csv",low_memory=False)
s=s[s.Year.between(1944,2025)].copy()
p=pd.read_csv(ROOT/"data/derived/policy_level_indices_institution_year.csv",low_memory=False)
p=p[p.Year.between(1944,2025)].copy()
columns=["P_supportive_calib","P_neutral_calib","P_restrictive_calib"]
assert all(c in s for c in columns),f"Expected calibrated fields absent: {s.columns.tolist()}"
prob_sum=s[columns].sum(axis=1)
assert (prob_sum-1).abs().max()<1e-5, prob_sum.describe()
score=s.P_supportive_calib+0.5*s.P_neutral_calib
sentence_err=(score-s.ToneScore_0_1).abs()
assert float(sentence_err.max())<1e-8,sentence_err.max()
source=s.assign(Rebuilt_Score=score).groupby(["Institution","Year"],as_index=False).agg(rebuilt_pcsi=("Rebuilt_Score","mean"),rebuilt_sentence_n=("Rebuilt_Score","size"))
obs=p[p.Is_Carried_Forward.eq(0)][["Institution","Year","Mean_Tone_Score","n_sentences"]].copy()
j=obs.merge(source,on=["Institution","Year"],how="outer",indicator=True,validate="one_to_one")
assert len(j)==480 and j._merge.eq("both").all(),j._merge.value_counts()
max_mean_error=(j.Mean_Tone_Score-j.rebuilt_pcsi).abs().max()
assert max_mean_error<1e-10,max_mean_error
assert j.n_sentences.eq(j.rebuilt_sentence_n).all()
source=source.sort_values(["Institution","Year"]).copy()
expanded=[]
for inst,g in source.groupby("Institution"):
    periods=pd.DataFrame({"Year":list(range(int(g.Year.min()),2026))})
    m=periods.merge(g,on="Year",how="left",validate="one_to_one")
    m["Institution"]=inst
    m["Rebuilt_Score_Fill"]=m.rebuilt_pcsi.ffill()
    m["Source_Year_Rebuilt"]=m.Year.where(m.rebuilt_pcsi.notna()).ffill()
    m["Rebuilt_Carried"]=(m.Year!=m.Source_Year_Rebuilt).astype(int)
    expanded.append(m[["Institution","Year","Rebuilt_Score_Fill","Source_Year_Rebuilt","Rebuilt_Carried"]])
reconstructed=pd.concat(expanded,ignore_index=True)
recon=p[["Institution","Year","Mean_Tone_Score","Source_Year","Is_Carried_Forward"]].merge(reconstructed,on=["Institution","Year"],how="outer",indicator=True,validate="one_to_one")
assert recon._merge.eq("both").all(),recon._merge.value_counts()
assert len(recon)==4277,len(recon)
max_fill_error=(recon.Mean_Tone_Score-recon.Rebuilt_Score_Fill).abs().max()
assert max_fill_error<1e-10,max_fill_error
assert recon.Source_Year.eq(recon.Source_Year_Rebuilt).all()
assert recon.Is_Carried_Forward.eq(recon.Rebuilt_Carried).all()
result={
"sentence_rows":len(s),"source_university_years":len(source),"policy_panel_rows":len(recon),
"source_distinct_documents":int(s["File"].nunique()) if "File" in s.columns else None,
"max_prob_sum_error":float((prob_sum-1).abs().max()),
"max_sentence_score_error":float(sentence_err.max()),
"max_source_year_pcsi_error":float(max_mean_error),
"max_policy_panel_pcsi_error":float(max_fill_error),
"carried_forward_rows":int(recon.Is_Carried_Forward.sum()),
"all_sentence_count_matches":bool(j.n_sentences.eq(j.rebuilt_sentence_n).all()),
"all_source_year_links_match":bool(recon.Source_Year.eq(recon.Source_Year_Rebuilt).all()),
"all_carry_flags_match":bool(recon.Is_Carried_Forward.eq(recon.Rebuilt_Carried).all()),
"status":"PASS"}
(OUT/"independent_source_to_panel_reconstruction.json").write_text(json.dumps(result,indent=2)+"\n")
print("BEGIN_FULL_SOURCE_TO_PANEL_JSON\n"+json.dumps(result,indent=2)+"\nEND_FULL_SOURCE_TO_PANEL_JSON")
print("PCSI_SOURCE_TO_PANEL_RECONSTRUCTION_PASS")
