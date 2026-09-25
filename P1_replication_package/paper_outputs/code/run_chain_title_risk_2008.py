from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rapidfuzz import fuzz, process
from run_stanford_roche_causal_test import build_language_measures, attach_language_to_panel

ROOT=Path(__file__).resolve().parents[2]
PANEL=ROOT/'data/derived/policy_level_indices_institution_year.csv'
SENT=ROOT/'data/derived/sentence_scores_canonical.csv'
SRC=ROOT/'data/external/ncses_fy2008_top150_sources.csv'
OUT=ROOT/'paper_outputs/tables/chain_title_risk_2008'
SHOCK=2009

def canon(x):
    x=str(x).lower().replace('&',' and ')
    repl={r'\buniversity\b':' u ',r'\bu\.?\b':' u ',r'\bcollege\b':' c ',r'\bc\.?\b':' c ',r'\binstitute\b':' inst ',r'\btechnology\b':' tech ',r'\bmedical\b':' med ',r'\bcenter\b':' ctr ',r'\bscience\b':' sci ',r'\bhealth\b':' health ',r'\bthe\b':' '}
    for a,b in repl.items(): x=re.sub(a,b,x)
    x=re.sub(r'\ball campuses\b|\bmain campus\b',' ',x)
    x=re.sub(r'[^a-z0-9]+',' ',x)
    return re.sub(r'\s+',' ',x).strip()

MANUAL={
'Albert Einstein/Yeshiva':'Yeshiva U.','Brigham Young University':None,'Colorado State University':'CO State U.',
'Columbia University':'Columbia U. in the City of New York','Dartmouth College':'Dartmouth C.','Indiana University':'IN U. all campuses',
'Johns Hopkins University':'Johns Hopkins U., Thea','Louisiana State University':'LA State U. all campuses','Medical College of Ohio':None,
'Montana State University':'MT State U. Bozeman','Oklahoma State University':'OK State U. all campuses','Penn State University':'PA State U. all campuses',
'Purdue University':'Purdue U. all campuses','Rutgers the State University of NJ':'Rutgers, The State U. NJ all campuses',
'Southern Illinois University':None,'SUNY':'SUNY Buffalo all campuses','Texas A&M University System':'TX A&M U.',
'University of Alabama in Birmingham (UAB)':'U. AL Birmingham, The','University of Arkansas':'U. AR main campus',
'University of California System':None,'University of Cincinnati':'U. Cincinnati all campuses','University of Colorado':'U. CO all campuses',
'University of Connecticut':'U. CT all campuses','University of Hawaii':'U. HI Manoa','University of Kansas':'U. KS all campuses',
'University of Kentucky':'U. KY all campuses','University of Maryland Baltimore':'U. MD Baltimore','University of Michigan':'U. MI all campuses',
'University of Minnesota':'U. MN all campuses','University of Missouri System':'U. MO-Columbia','University of Nebraska':'U. NE all campuses',
'University of Oklahoma':'U. OK all campuses','University of Pittsburgh':'U. Pittsburgh all campuses','University of South Carolina':'U. SC all campuses',
'University of Tennessee':'U. TN all campuses','University of Texas at Austin':'U. TX Austin','University of Virginia':'U. VA all campuses',
'University of Washington':'U. WA','Virginia Tech':'VA Polytechnic Institute and State U.','Washington State University':'WA State U.',
'University of Wisconsin-Madison':'U. WI Madison','Ohio State University':'OH State U. all campuses','Cornell University':'Cornell U. all campuses',
'University of Pennsylvania':'U. PA','University of Florida':'U. FL','University of Arizona':'U. AZ','University of Southern California':'U. Southern CA',
'University of Illinois Urbana-Champaign':'U. IL Urbana-Champaign','University of North Carolina at Chapel Hill':'U. NC Chapel Hill',
'Georgia Institute of Technology':'GA Institute of Technology all campuses','University of Maryland College Park':'U. MD College Park',
'University of Rochester':'U. Rochester','North Carolina State University':'NC State U.','Michigan State University':'MI State U.',
'University of Georgia':'U. GA','University of Iowa':'U. IA','University of South Florida':'U. South FL','Arizona State University':'AZ State U.',
'University of Utah':'U. UT','Wayne State University':'Wayne State U.','University of Miami':'U. Miami','Iowa State University':'IA State U.',
'Mississippi State University':'MS State U.','Carnegie Mellon University':'Carnegie Mellon U.','Wake Forest University':'Wake Forest U.',
'University of New Mexico':'U. NM main campus','Clemson University':'Clemson U.','Princeton University':'Princeton U.','Oregon State University':'OR State U.',
'Florida State University':'FL State U.','Brown University':'Brown U.','George Washington University':'George Washington U.',
'Tulane University':'Tulane U.','University of Massachusetts Amherst':'U. MA Amherst','University of Louisville':'U. Louisville',
'Virginia Commonwealth University':'VA Commonwealth U.','Auburn University':'Auburn U. all campuses','Utah State University':'UT State U.',
'Georgetown University':'Georgetown U.','West Virginia University':'WV U.','Tufts University':'Tufts U.','Kansas State University':'KS State U.',
'University of Delaware':'U. DE','University of Vermont':'U. VT','University of New Hampshire':'U. NH','University of Central Florida':'U. Central FL',
'Drexel University':'Drexel U.','University of Notre Dame':'U. of Notre Dame','Florida International University':'FL International U.',
'Temple University':'Temple U.','University of Houston':'U. Houston','Rice University':'Rice U.','Rensselaer Polytechnic Institute':'Rensselaer Polytechnic Institute'
}

def build_crosswalk(insts, src):
    names=src.ncses_name.tolist(); can={canon(n):n for n in names}
    rows=[]
    for inst in sorted(insts):
        target=MANUAL.get(inst,'__NO_MANUAL__')
        if target is None:
            rows.append({'Institution':inst,'ncses_name':None,'method':'excluded_manual','score':np.nan}); continue
        if target!='__NO_MANUAL__' and target in names:
            rows.append({'Institution':inst,'ncses_name':target,'method':'manual','score':100}); continue
        key=canon(inst)
        if key in can:
            rows.append({'Institution':inst,'ncses_name':can[key],'method':'canonical_exact','score':100}); continue
        best=process.extractOne(key, {n:canon(n) for n in names}, scorer=fuzz.WRatio)
        # process dict returns (value, score, key); value is canonical, key is original name
        match=best[2]; score=float(best[1])
        if score>=88:
            rows.append({'Institution':inst,'ncses_name':match,'method':'high_confidence_fuzzy','score':score})
        else:
            rows.append({'Institution':inst,'ncses_name':None,'method':'unmatched','score':score})
    return pd.DataFrame(rows)

def z(s): return (s-s.mean())/s.std(ddof=0)
def fit(formula,d): return smf.ols(formula,d).fit(cov_type='cluster',cov_kwds={'groups':d.Institution})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    panel=pd.read_csv(PANEL); sent=pd.read_csv(SENT); src=pd.read_csv(SRC)
    lang=build_language_measures(sent); p=attach_language_to_panel(panel,lang)
    cw=build_crosswalk(p.Institution.unique(),src)
    cw=cw.merge(src,on='ncses_name',how='left')
    cw['federal_share']=cw.federal_rd_2008/cw.total_rd_2008
    cw['industry_share']=cw.industry_rd_2008/cw.total_rd_2008
    cw['log_total_rd']=np.log1p(cw.total_rd_2008)
    use=cw.dropna(subset=['federal_share','industry_share','log_total_rd']).copy()
    for c in ['federal_share','industry_share','log_total_rd'] : use[c+'_z']=z(use[c])
    use['dual_z']=z(use.federal_share_z*use.industry_share_z)
    use['dual_raw']=use.federal_share*use.industry_share
    cw.to_csv(OUT/'crosswalk_all.csv',index=False); use.to_csv(OUT/'exposure_analysis_sample.csv',index=False)
    outcomes=['has_strong_assignment','strong_per_1000w','weak_per_1000w','net_assignment_strength','Mean_Tone_Score','Legal_Load_Index','obligation_modal_share']
    windows=[(2004,2014),(2006,2013),(2000,2016)]
    rows=[]
    for start,end in windows:
      for obs in [False,True]:
       d=p.merge(use[['Institution','federal_share_z','industry_share_z','log_total_rd_z','dual_z']],on='Institution',how='inner')
       d=d[d.Year.between(start,end)&(d.Year!=SHOCK)].copy()
       if obs: d=d[d.Is_Carried_Forward==0].copy()
       d['post']=(d.Year>SHOCK).astype(int)
       for x in ['federal_share_z','industry_share_z','log_total_rd_z','dual_z']: d[x+'_post']=d[x]*d.post
       for y in outcomes:
        dd=d.dropna(subset=[y]).copy()
        if dd.Institution.nunique()<20: continue
        m=fit(f'{y} ~ federal_share_z_post + industry_share_z_post + log_total_rd_z_post + dual_z_post + C(Institution)+C(Year)',dd)
        for term in ['federal_share_z_post','industry_share_z_post','log_total_rd_z_post','dual_z_post']:
          ci=m.conf_int().loc[term]
          rows.append({'start':start,'end':end,'observed_only':obs,'outcome':y,'term':term,'coef':m.params[term],'se':m.bse[term],'p':m.pvalues[term],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1],'N':m.nobs,'institutions':dd.Institution.nunique()})
    pd.DataFrame(rows).to_csv(OUT/'did_results.csv',index=False)
    # Event study for net assignment strength: dual exposure coefficient each year, controlling flexibly for lower-order exposure by year.
    d=p.merge(use[['Institution','federal_share_z','industry_share_z','log_total_rd_z','dual_z']],on='Institution',how='inner')
    d=d[d.Year.between(2004,2015)&(d.Year!=SHOCK)].dropna(subset=['net_assignment_strength']).copy()
    ref=2008; dual_terms=[]; lower=[]
    for yr in sorted(d.Year.unique()):
      if yr in [ref,SHOCK]: continue
      for v in ['dual_z','federal_share_z','industry_share_z','log_total_rd_z']:
        nm=f'{v}_y{int(yr)}'; d[nm]=d[v]*(d.Year==yr).astype(int)
        (dual_terms if v=='dual_z' else lower).append(nm)
    m=fit('net_assignment_strength ~ '+' + '.join(dual_terms+lower)+' + C(Institution)+C(Year)',d)
    er=[]
    for term in dual_terms:
      yr=int(term.split('y')[-1]); ci=m.conf_int().loc[term]
      er.append({'year':yr,'event_time':yr-SHOCK,'coef':m.params[term],'se':m.bse[term],'p':m.pvalues[term],'ci_low':ci.iloc[0],'ci_high':ci.iloc[1]})
    pd.DataFrame(er).to_csv(OUT/'event_study_dual.csv',index=False)
    leads=[x for x in dual_terms if int(x.split('y')[-1])<SHOCK]
    if leads:
      test=m.wald_test(' = 0, '.join(leads)+' = 0',scalar=True)
      pd.DataFrame([{'joint_pretrend_p':float(test.pvalue),'n_leads':len(leads),'N':m.nobs,'institutions':d.Institution.nunique()}]).to_csv(OUT/'pretrend_test.csv',index=False)
    # placebo shocks
    pr=[]
    for sh in [2005,2007]:
      d=p.merge(use[['Institution','federal_share_z','industry_share_z','log_total_rd_z','dual_z']],on='Institution',how='inner')
      d=d[d.Year.between(sh-4,sh+4)&(d.Year!=sh)].dropna(subset=['net_assignment_strength']).copy(); d['post']=(d.Year>sh).astype(int)
      for x in ['federal_share_z','industry_share_z','log_total_rd_z','dual_z']: d[x+'_post']=d[x]*d.post
      m=fit('net_assignment_strength ~ federal_share_z_post + industry_share_z_post + log_total_rd_z_post + dual_z_post + C(Institution)+C(Year)',d)
      pr.append({'placebo_shock':sh,'coef':m.params['dual_z_post'],'se':m.bse['dual_z_post'],'p':m.pvalues['dual_z_post'],'N':m.nobs,'institutions':d.Institution.nunique()})
    pd.DataFrame(pr).to_csv(OUT/'placebo_results.csv',index=False)
    print('matched',len(use),'of',cw.Institution.nunique())
    print(pd.DataFrame(rows).query("start==2004 and end==2014 and observed_only==False and term=='dual_z_post'")[['outcome','coef','se','p','N','institutions']].to_string(index=False))

if __name__=='__main__': main()
