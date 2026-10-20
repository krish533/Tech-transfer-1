#!/usr/bin/env python3
"""Independent PCSI replication audit, never overwrites manuscript output.

Reads the *complete* versioned CSV stored on GitHub and re-calculates
descriptive quantities and exact within/composition (shift/share) identities.
The category definitions are imported from the published paper code; the
decomposition arithmetic here is independent.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "P1_replication_package"
CODE = PACKAGE / "paper_outputs" / "code"
OUT = PACKAGE / "paper_outputs" / "tables" / "independent_pcsi_audit"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(CODE))
from generate_revision_analyses import tag_topics, tag_deontic


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def decompose(frame, category):
    # The target is the average change in each university's sentence-weighted
    # mean, weighting the 63 continuing universities equally.
    d = frame.loc[frame.Year.between(1980, 1999) | frame.Year.between(2010, 2025)].copy()
    d["Period"] = np.where(d.Year <= 1999, "early", "late")
    both = sorted(set(d.loc[d.Period.eq("early"), "Institution"]) &
                  set(d.loc[d.Period.eq("late"), "Institution"]))
    d = d[d.Institution.isin(both)]
    per_inst = []
    entries = []
    for university, g in d.groupby("Institution", sort=True):
        tab = g.groupby(["Period",category])["ToneScore_0_1"].agg(["mean","size"])
        scores = {p: float(g.loc[g.Period.eq(p), "ToneScore_0_1"].mean()) for p in ("early","late")}
        totals = {p: int(g.Period.eq(p).sum()) for p in ("early","late")}
        cats = set(g[category].dropna())
        within = composition = 0.0
        for c in cats:
            present_e = ("early",c) in tab.index
            present_l = ("late",c) in tab.index
            share_e = float(tab.loc[("early",c),"size"])/totals["early"] if present_e else 0.0
            share_l = float(tab.loc[("late",c),"size"])/totals["late"] if present_l else 0.0
            # A category absent in one period has an unobserved conditional mean.
            # Set this mean equal to the observed-period mean: all entry/exit
            # effects are allocated to composition, not within-category changes.
            mean_e = float(tab.loc[("early",c),"mean"]) if present_e else float(tab.loc[("late",c),"mean"])
            mean_l = float(tab.loc[("late",c),"mean"]) if present_l else float(tab.loc[("early",c),"mean"])
            w = 0.5 * (share_e + share_l) * (mean_l-mean_e)
            comp = 0.5 * (mean_l + mean_e) * (share_l-share_e)
            entries.append(dict(Institution=university,Category=c,
                                early_share=share_e,late_share=share_l,
                                early_mean=mean_e,late_mean=mean_l,
                                within=w,composition=comp))
            within += w
            composition += comp
        change = scores["late"]-scores["early"]
        assert np.isclose(within+composition,change,rtol=0,atol=1e-11), \
               f"{category}: identity failed for {university}: {within+composition} != {change}"
        per_inst.append(dict(Institution=university,early_mean=scores["early"],
                             late_mean=scores["late"],delta=change,
                             within=within,composition=composition,
                             early_sentences=totals["early"],late_sentences=totals["late"]))
    u = pd.DataFrame(per_inst).sort_values("Institution")
    e = pd.DataFrame(entries)
    by_cat = e.groupby("Category")[["within","composition"]].sum().div(len(u)).reset_index()
    total = float(u.delta.mean())
    w = float(by_cat.within.sum())
    c = float(by_cat.composition.sum())
    assert abs(w+c-total)<1e-11
    return {"universities":len(u), "sentences_early":int(d.Period.eq("early").sum()),
            "sentences_late":int(d.Period.eq("late").sum()),
            "pcsi_early_mean":float(u.early_mean.mean()),
            "pcsi_late_mean":float(u.late_mean.mean()),
            "change":total, "within":w, "composition":c,
            "identity_error":w+c-total},by_cat,u,e


def main():
    src = PACKAGE/"data"/"derived"/"sentence_scores_canonical.csv"
    policy = PACKAGE/"data"/"derived"/"policy_level_indices_institution_year.csv"
    if not src.exists() or not policy.exists():
        raise FileNotFoundError("Required canonical source data absent from checkout")
    s = pd.read_csv(src,low_memory=False)
    p = pd.read_csv(policy,low_memory=False)
    for c in ("Institution","Year","ToneScore_0_1","Sentence_Cleaned"):
        if c not in s.columns:
            raise ValueError(f"sentence file missing {c}: {list(s.columns)}")
    for c in ("Institution","Year","Mean_Tone_Score","Is_Carried_Forward"):
        if c not in p.columns:
            raise ValueError(f"panel file missing {c}: {list(p.columns)}")
    s["Year"] = pd.to_numeric(s["Year"],errors="raise")
    s["ToneScore_0_1"] = pd.to_numeric(s["ToneScore_0_1"],errors="raise")
    s = s[s.Year.between(1944,2025)].copy()
    p = p[p.Year.between(1944,2025)].copy()
    observed = p[p.Is_Carried_Forward.eq(0)]
    summary = {
        "source_sha256":file_hash(src),
        "policy_sha256":file_hash(policy),
        "sentence_rows_1944_2025":int(len(s)),
        "sentences_universities":int(s.Institution.nunique()),
        "panel_rows":int(len(p)),
        "universities":int(p.Institution.nunique()),
        "source_policy_years":int(len(observed)),
        "mean_pcsi":float(p.Mean_Tone_Score.mean()),
        "median_pcsi":float(p.Mean_Tone_Score.median()),
        "sd_pcsi":float(p.Mean_Tone_Score.std(ddof=1)),
        "share_below_midpoint":float(p.Mean_Tone_Score.lt(0.5).mean()),
        "source_policy_year_mean":float(observed.Mean_Tone_Score.mean()),
        "source_policy_year_share_below_midpoint":float(observed.Mean_Tone_Score.lt(0.5).mean()),
    }
    summary["checks"]={
        "150_universities":summary["universities"]==150,
        "4277_panel_rows":summary["panel_rows"]==4277,
        "480_observed_policy_years":summary["source_policy_years"]==480,
        "mean_pcsi_matches_prior_audit":abs(summary["mean_pcsi"]-0.43684415)<1e-7,
    }
    audits = {}
    for name,transformed,c in (
        ("topic",tag_topics(s.copy()),"Primary topic"),
        ("deontic",tag_deontic(s.copy()),"Deontic x sanction"),
    ):
        results,categories,by_institution,by_university_category=decompose(transformed,c)
        assert results["universities"]==63, results
        assert results["sentences_early"]==8353, results
        assert results["sentences_late"]==28175, results
        assert abs(results["change"]+0.0288161081)<1e-8, results
        categories.to_csv(OUT/f"{name}_exact_category_contributions.csv",index=False,float_format="%.12f")
        by_institution.to_csv(OUT/f"{name}_exact_university_contributions.csv",index=False,float_format="%.12f")
        by_university_category.to_csv(OUT/f"{name}_cells.csv",index=False,float_format="%.12f")
        audits[name]=results
    summary["decompositions"]=audits
    (OUT/"independent_replication_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("BEGIN_PCSI_INDEPENDENT_AUDIT_JSON")
    print(json.dumps(summary,indent=2))
    print("END_PCSI_INDEPENDENT_AUDIT_JSON")
    for name, d in audits.items():
        print(f"CHECKED {name}: change={d['change']:.12f}, within={d['within']:.12f}, composition={d['composition']:.12f}, identity error={d['identity_error']:.3g}")
    assert all(summary["checks"].values())
    print("PCSI_AUDIT_PASS: all requested panel checks and exact university-level shift/share identities")


if __name__=="__main__":
    main()
