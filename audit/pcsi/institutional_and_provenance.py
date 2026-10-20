#!/usr/bin/env python3
"""Final PCSI source-document provenance and institutional-regression verification.
Writes audit-only outputs; does not change study estimates.
"""
from pathlib import Path
import json
import pandas as pd
import statsmodels.formula.api as smf
ROOT=Path(__file__).resolve().parents[2]/"P1_replication_package"
OUT=ROOT/"paper_outputs"/"tables"/"independent_pcsi_audit"
s=pd.read_csv(ROOT/"data/derived/sentence_scores_canonical.csv",low_memory=False)
s=s[s.Year.between(1944,2025)].copy()
p=pd.read_csv(ROOT/"data/derived/policy_level_indices_institution_year.csv",low_memory=False)
p=p[p.Year.between(1944,2025)].copy()
from sys import path
path.insert(0,str(ROOT/"paper_outputs"/"code"))
from generate_paper_outputs import REGIONS
d={
"sentence_rows":len(s),
"unique_filename_strings":int(s.File.nunique()),
"unique_institution_file_pairs":int(s[["Institution","File"]].drop_duplicates().shape[0]),
"unique_institution_year_file_triples":int(s[["Institution","Year","File"]].drop_duplicates().shape[0]),
"unique_university_source_year_pairs":int(s[["Institution","Year"]].drop_duplicates().shape[0]),
}
d["sample_count_warning"]="Filename strings alone do not necessarily identify unique source documents; compare the institution-year-file triples against the published number of 518."
o=p[p.Is_Carried_Forward.eq(0)].copy()
o["Urb"]=o["Urbanicity (cat)"]
o["Region"]=o.STATE.map(REGIONS)
o["LG"]=o["Land-Grant Institution"]
o["Med"]=o["MEDSCHOOL"]
o["yr"]=o.Year.astype(int)
o=o.dropna(subset=["Urb","Region"])
formula="Mean_Tone_Score ~ C(Type, Treatment('Public R1')) + LG + Med + C(Urb, Treatment('Rural')) + C(Region, Treatment('Midwest')) + C(yr)"
model=smf.ols(formula,data=o).fit(cov_type="cluster",cov_kwds={"groups":o.Institution})
term="C(Type, Treatment('Public R1'))[T.Private R1]"
d["source_year_full_controls"]={
"n":int(model.nobs),"universities":int(o.Institution.nunique()),
"private_R1_coef":float(model.params[term]),
"private_R1_se":float(model.bse[term]),
"private_R1_p":float(model.pvalues[term]),
"land_grant_coef":float(model.params["LG"]),
"land_grant_se":float(model.bse["LG"]),
"land_grant_p":float(model.pvalues["LG"]),
}
assert model.nobs==452,model.nobs
assert abs(model.params[term]-0.039931)<0.0001,model.params[term]
assert abs(model.params["LG"]+0.004874)<0.0001,model.params["LG"]
d["regression_status"]="PASS"
(OUT/"institutional_and_document_count.json").write_text(json.dumps(d,indent=2)+"\n")
print("BEGIN_INSTITUTIONAL_PROVENANCE_AUDIT\n"+json.dumps(d,indent=2)+"\nEND_INSTITUTIONAL_PROVENANCE_AUDIT")
print("INSTITUTIONAL_REGRESSION_REPLICATION_PASS")
