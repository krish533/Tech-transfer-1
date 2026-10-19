"""Revision analyses for the October 2026 manuscript revision.

These exercises respond to referee-style feedback on the September 2026 draft:

1. Institutional correlates (Table 5): the published specifications, Column (1)
   re-estimated on the common 144-university sample, weighted versions, and
   era-adjusted versions that remove the secular PCSI trend before comparing
   universities whose archives come from different decades.
2. Linguistic decomposition (Table 6): Shapley (dominance-analysis) shares of
   R^2 for Tone, Clarity and Legal Load, which do not depend on entry order.
3. Clause-topic decomposition of the modern PCSI decline: is the drift toward
   restrictive stance concentrated in ownership/assignment language (the
   post-Stanford v. Roche "hereby assign" revision) or visible within topics?
4. Rule form vs. framing: sentences coded by deontic operator (obligation,
   prohibition, permission) and sanction language, following the
   Institutional Grammar; PCSI net of rule form; shift-share over forms.
5. Classifier uncertainty: PCSI from hard labels and from high-confidence
   sentences only, and model confidence by decade.

All outputs are written to paper_outputs/tables/revision_analyses/ and
paper_outputs/figures/. Nothing here changes the construction of PCSI.

Run from the package root:
    python paper_outputs/code/generate_revision_analyses.py
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from generate_paper_outputs import REGIONS

TYPE = "C(Type, Treatment('Public R1'))"
URB = "C(Urb, Treatment('Rural'))"
REG = "C(Region, Treatment('Midwest'))"
SPECS = {
    "(1) Type only": f"pcsi ~ {TYPE}",
    "(2) + LG, Med, Urbanicity": f"pcsi ~ {TYPE} + LG + Med + {URB}",
    "(3) + Region FE": f"pcsi ~ {TYPE} + LG + Med + {URB} + {REG}",
}
KEEP_TERMS = {
    f"{TYPE}[T.Private R1]": "Private R1",
    f"{TYPE}[T.Private R2]": "Private R2",
    f"{TYPE}[T.Public R2]": "Public R2",
    "LG": "Land-grant",
    "Med": "Medical school",
}

# Clause topics. A sentence can match several topics (multi-label); the
# exclusive "primary" topic uses the order below as priority.
TOPICS = {
    "Ownership & assignment": r"\bassign|\bownership\b|\bowned\b|\bowns\b|\btitle\b|\bvest|belong|property of the univ",
    "Disclosure": r"disclos|invention report|report (any|all|the) invention",
    "Revenue sharing": r"royalt|revenue|net (income|proceeds)|\bincome\b|distribut|\bequity\b",
    "Licensing & commercialization": r"licens|commerciali|technology transfer|technology licensing|\bott\b|\btto\b|start-?up|patent application|marketab",
    "Sponsored research & third parties": r"sponsor|\bgrant\b|\bgrants\b|federal|government|contract|third part",
    "Copyright & software": r"copyright|software|courseware|scholarly work|works? of authorship",
    "Administration & disputes": r"appeal|dispute|committee|review|waive|waiver|grievance|amend",
    "Definitions & scope": r"\bmeans\b|\bdefined\b|definition|\bshall mean\b|appl(y|ies) to|\bpurpose\b|\bscope\b",
}


def cluster_fit(formula: str, data: pd.DataFrame, weights=None):
    model = smf.wls(formula, data=data, weights=weights) if weights is not None else smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["Institution"]})


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def load(package_root: Path):
    panel = pd.read_csv(package_root / "data/derived/policy_level_indices_primary_1944_2025.csv", low_memory=False)
    if len(panel) != 4277 or panel["Institution"].nunique() != 150:
        raise ValueError("Primary panel does not match the manuscript sample (4,277 rows, 150 universities).")
    panel["Urb"] = panel["Urbanicity (cat)"]
    panel["Region"] = panel["STATE"].map(REGIONS)
    panel["LG"] = panel["Land-Grant Institution"]
    panel["Med"] = panel["MEDSCHOOL"]
    sentences = pd.read_csv(package_root / "data/derived/sentence_scores_canonical.csv", low_memory=False)
    sentences = sentences[sentences["Year"].between(1944, 2025)].copy()
    return panel, sentences


def institution_frame(panel: pd.DataFrame, value: str = "Mean_Tone_Score") -> pd.DataFrame:
    return (
        panel.groupby("Institution")
        .agg(
            pcsi=(value, "mean"),
            Type=("Type", "first"),
            LG=("LG", "first"),
            Med=("Med", "first"),
            Urb=("Urb", "first"),
            Region=("Region", "first"),
            n_source_years=("Source_Year", "nunique"),
            mean_source_year=("Source_Year", "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 1. Institutional correlates
# ---------------------------------------------------------------------------
def institutional_robustness(panel: pd.DataFrame, sentences: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    inst = institution_frame(panel)
    n_sent = sentences.groupby("Institution").size().rename("n_sentences")
    inst = inst.merge(n_sent, on="Institution", how="left")
    common = inst.dropna(subset=["Urb", "Region"])

    # Era adjustment A: remove calendar-year means from the policy-in-force panel
    # before averaging within university.
    adj = panel.copy()
    adj["pcsi_adj"] = adj["Mean_Tone_Score"] - adj.groupby("Year")["Mean_Tone_Score"].transform("mean") + adj["Mean_Tone_Score"].mean()
    inst_adj = institution_frame(adj, "pcsi_adj")
    common_adj = inst_adj.dropna(subset=["Urb", "Region"])

    # Era adjustment B: source-text years only, with source-year fixed effects.
    obs = panel[panel["Is_Carried_Forward"] == 0].copy().rename(columns={"Mean_Tone_Score": "pcsi"})
    obs["yr"] = obs["Year"].astype(int)
    obs_common = obs.dropna(subset=["Urb", "Region"])

    runs = []
    for label, formula in SPECS.items():
        sample = inst if label.startswith("(1)") else inst.dropna(subset=["Urb"]) if label.startswith("(2)") else common
        runs.append(("Published (HC3)", label, smf.ols(formula, sample).fit(cov_type="HC3")))
        runs.append(("Common 144 sample (HC3)", label, smf.ols(formula, common).fit(cov_type="HC3")))
        runs.append(
            ("WLS: weight = source policy years (HC3)", label,
             smf.wls(formula, common, weights=common["n_source_years"]).fit(cov_type="HC3"))
        )
        runs.append(
            ("WLS: weight = sentences (HC3)", label,
             smf.wls(formula, common, weights=common["n_sentences"]).fit(cov_type="HC3"))
        )
        runs.append(("Era-adjusted means: year-demeaned panel (HC3)", label, smf.ols(formula, common_adj).fit(cov_type="HC3")))
        runs.append(
            ("Source-text years + year FE (clustered)", label,
             cluster_fit(formula + " + C(yr)", obs_common))
        )
        runs.append(
            ("Source-text years 1980+ + year FE (clustered)", label,
             cluster_fit(formula + " + C(yr)", obs_common[obs_common["yr"] >= 1980]))
        )

    rows = []
    for sample, spec, fit in runs:
        for term, name in KEEP_TERMS.items():
            if term in fit.params.index:
                rows.append(
                    {
                        "Sample / estimator": sample,
                        "Specification": spec,
                        "Term": name,
                        "Coef": round(float(fit.params[term]), 4),
                        "SE": round(float(fit.bse[term]), 4),
                        "p": round(float(fit.pvalues[term]), 4),
                        "N": int(fit.nobs),
                        "R2": round(float(fit.rsquared), 3),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "table5_institutional_robustness.csv", index=False)

    # Does archive vintage correlate with institutional type? (Motivates era adjustment.)
    vintage = inst.groupby("Type")["mean_source_year"].agg(["mean", "std", "size"]).round(1)
    vintage.to_csv(outdir / "archive_vintage_by_type.csv")
    return out


# ---------------------------------------------------------------------------
# 2. Shapley R^2 decomposition
# ---------------------------------------------------------------------------
def shapley_r2(data: pd.DataFrame, y: str, xs: list[str]) -> dict[str, float]:
    cache: dict[tuple, float] = {(): 0.0}

    def r2(subset: tuple) -> float:
        if subset not in cache:
            cache[subset] = smf.ols(f"{y} ~ " + " + ".join(subset), data).fit().rsquared
        return cache[subset]

    k = len(xs)
    shares = {}
    for x in xs:
        others = [v for v in xs if v != x]
        total = 0.0
        for size in range(k):
            for combo in itertools.combinations(others, size):
                base = tuple(sorted(combo))
                with_x = tuple(sorted(combo + (x,)))
                weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
                total += weight * (r2(with_x) - r2(base))
        shares[x] = total
    shares["Total R2"] = r2(tuple(sorted(xs)))
    return shares


def linguistic_shapley(panel: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    xs = ["Tone_Index", "Clarity_Index", "Legal_Load_Index"]
    rows = []
    for label, frame in [
        ("Policy-in-force panel (N=4,277)", panel),
        ("Source-text years (N=480)", panel[panel["Is_Carried_Forward"] == 0]),
    ]:
        d = frame[["Mean_Tone_Score"] + xs].dropna()
        sh = shapley_r2(d, "Mean_Tone_Score", xs)
        total = sh["Total R2"]
        for x in xs:
            rows.append({"Sample": label, "Feature": x.replace("_Index", "").replace("_", " "),
                         "Shapley R2": round(sh[x], 4), "Share of explained": round(sh[x] / total, 3)})
        rows.append({"Sample": label, "Feature": "Total", "Shapley R2": round(total, 4), "Share of explained": 1.0})
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "table6_shapley_r2.csv", index=False)
    return out



def shift_share(s: pd.DataFrame, category: str) -> tuple[pd.DataFrame, str]:
    """Within-category vs composition decomposition of the change in mean
    sentence score between 1980-1999 and 2010-2025.

    Uses universities with source text in both periods and weights each
    university equally: shares and category means are computed within
    university-period and then averaged across universities.
    """
    s = s[s["Year"].between(1980, 2025)].copy()
    s["Period"] = np.select([s["Year"].between(1980, 1999), s["Year"].between(2010, 2025)], ["early", "late"], "mid")
    both = set(s.loc[s["Period"] == "early", "Institution"]) & set(s.loc[s["Period"] == "late", "Institution"])
    b = s[s["Institution"].isin(both) & s["Period"].isin(["early", "late"])]
    up = b.groupby(["Institution", "Period", category])["ToneScore_0_1"].agg(["mean", "size"]).reset_index()
    up["share"] = up["size"] / up.groupby(["Institution", "Period"])["size"].transform("sum")
    shares = (
        up.pivot_table(index=["Institution", category], columns="Period", values="share")
        .reindex(pd.MultiIndex.from_product([sorted(both), sorted(up[category].unique())], names=["Institution", category]))
        .fillna(0)
        .groupby(category)
        .mean()
    )
    means = up.groupby(["Period", category])["mean"].mean().unstack(0)
    cats = shares.index.intersection(means.index)
    shares, means = shares.loc[cats], means.loc[cats]
    avg_share = 0.5 * (shares["late"] + shares["early"])
    centre = 0.5 * (means["late"] + means["early"])
    within = avg_share * (means["late"] - means["early"])
    # Centre category means on the grand mean so each composition term reads as
    # "more sentences of a below-average (or above-average) category".
    comp = (centre - (centre * avg_share).sum()) * (shares["late"] - shares["early"])
    out = pd.DataFrame({"Early share": shares["early"].round(3), "Late share": shares["late"].round(3),
                        "Early mean": means["early"].round(3), "Late mean": means["late"].round(3),
                        "Within contribution": within.round(4), "Composition contribution": comp.round(4)})
    out.loc["Total"] = [round(shares["early"].sum(), 3), round(shares["late"].sum(), 3), np.nan, np.nan,
                        round(within.sum(), 4), round(comp.sum(), 4)]
    note = (f"Universities with source text in both 1980-1999 and 2010-2025: {len(both)}\n"
            f"Sentences: early={int((b.Period == 'early').sum())}, late={int((b.Period == 'late').sum())}\n")
    return out, note

# ---------------------------------------------------------------------------
# 3. Clause-topic decomposition of the temporal decline
# ---------------------------------------------------------------------------
def tag_topics(sentences: pd.DataFrame) -> pd.DataFrame:
    text = sentences["Sentence_Cleaned"].fillna("").str.lower()
    for topic, pattern in TOPICS.items():
        sentences[topic] = text.str.contains(re.sub(r"\((?!\?)", "(?:", pattern), regex=True)
    flags = sentences[list(TOPICS)]
    sentences["Primary topic"] = np.where(flags.any(axis=1), flags.idxmax(axis=1), "Other / general")
    return sentences


def topic_decomposition(sentences: pd.DataFrame, outdir: Path, figdir: Path) -> None:
    s = tag_topics(sentences.copy())
    s = s[s["Year"] >= 1980]

    # (a) Topic coverage and mean stance.
    cover = []
    for topic in list(TOPICS) + ["Other / general"]:
        mask = s["Primary topic"].eq(topic) if topic == "Other / general" else s[topic]
        cover.append({"Topic": topic, "Share of sentences (multi-label)": round(mask.mean(), 3),
                      "Share as primary topic": round(s["Primary topic"].eq(topic).mean(), 3),
                      "Mean sentence score": round(s.loc[mask, "ToneScore_0_1"].mean(), 3),
                      "Restrictive share": round((s.loc[mask, "Pred_Label"].astype(str).str.lower().str.startswith("r")).mean(), 3)})
    pd.DataFrame(cover).to_csv(outdir / "topic_coverage.csv", index=False)

    # (b) Within-topic university-FE trends on source-text years, 1980+.
    trends = []
    for topic in list(TOPICS) + ["Other / general"]:
        mask = s["Primary topic"].eq(topic) if topic == "Other / general" else s[topic]
        g = s[mask].groupby(["Institution", "Year"], as_index=False)["ToneScore_0_1"].mean()
        multi = g.groupby("Institution")["Year"].transform("nunique") > 1
        g = g[multi]
        if g["Institution"].nunique() < 10:
            continue
        fit = cluster_fit("ToneScore_0_1 ~ Year + C(Institution)", g)
        trends.append({"Topic": topic, "FE trend per year": round(fit.params["Year"], 5),
                       "SE": round(fit.bse["Year"], 5), "p": round(fit.pvalues["Year"], 3),
                       "University-years": len(g), "Universities": g["Institution"].nunique()})
    # All sentences, same construction, for comparison.
    g = s.groupby(["Institution", "Year"], as_index=False)["ToneScore_0_1"].mean()
    g = g[g.groupby("Institution")["Year"].transform("nunique") > 1]
    fit = cluster_fit("ToneScore_0_1 ~ Year + C(Institution)", g)
    trends.append({"Topic": "All sentences (= PCSI)", "FE trend per year": round(fit.params["Year"], 5),
                   "SE": round(fit.bse["Year"], 5), "p": round(fit.pvalues["Year"], 3),
                   "University-years": len(g), "Universities": g["Institution"].nunique()})
    trend_df = pd.DataFrame(trends)
    trend_df.to_csv(outdir / "topic_within_trends.csv", index=False)

    # (c) Shift-share: universities with source text in both 1980-1999 and 2010-2025.
    ss, note = shift_share(s, "Primary topic")
    ss.to_csv(outdir / "topic_shift_share.csv")
    (outdir / "topic_shift_share_note.txt").write_text(note)

    # Figure: within-topic trends with 95% CIs.
    t = trend_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ypos = np.arange(len(t))
    colors = ["#0072B2" if not x.startswith("All") else "#D55E00" for x in t["Topic"]]
    for yv, est, se, col in zip(ypos, t["FE trend per year"], t["SE"], colors):
        ax.errorbar(est * 1000, yv, xerr=1.96 * se * 1000, fmt="none", ecolor=col, capsize=3)
    ax.scatter(t["FE trend per year"] * 1000, ypos, c=colors, zorder=3)
    ax.axvline(0, color="#999999", linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(t["Topic"])
    ax.set_xlabel("Within-university trend in mean sentence score (x 1,000 per year)")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(figdir / "fig_topic_trends.pdf", bbox_inches="tight")
    fig.savefig(figdir / "fig_topic_trends.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Rule form vs. framing (Institutional Grammar)
# ---------------------------------------------------------------------------
# Deontic operator of each sentence, following Crawford and Ostrom (1995):
# prohibition (must not) > obligation (must) > permission (may) > none.
# "Or else" marks sanction language.
DEONTIC = {
    "Prohibition": r"\b(?:shall|must|may|will|can|does|do)\s+not\b|\bcannot\b|prohibit|forbid|not (?:be )?permitted",
    "Obligation": r"\bshall\b|\bmust\b|\b(?:is|are|be) required\b|\brequires?\b|obligat|\bagrees? to\b|(?:is|are) expected to",
    "Permission": r"\bmay\b|\bpermit|\bentitled\b|discretion|\bcan\b|\ballow",
}
SANCTION = r"violat|disciplin|sanction|forfeit|terminat|breach|penalt|dismiss|revok"
DEONTIC_ORDER = ["Obligation", "Prohibition", "Permission", "No deontic operator"]


def tag_deontic(sentences: pd.DataFrame) -> pd.DataFrame:
    text = sentences["Sentence_Cleaned"].fillna("").str.lower()
    deontic = pd.Series("No deontic operator", index=sentences.index)
    for label in ["Permission", "Obligation", "Prohibition"]:  # later assignments take priority
        deontic[text.str.contains(DEONTIC[label], regex=True)] = label
    sentences["Deontic"] = deontic
    sentences["Sanction"] = text.str.contains(SANCTION, regex=True)
    sentences["Deontic x sanction"] = sentences["Deontic"] + np.where(sentences["Sanction"], " + sanction", "")
    return sentences


def observed_frame(panel: pd.DataFrame) -> pd.DataFrame:
    obs = panel[panel["Is_Carried_Forward"] == 0].copy()
    return obs[["Institution", "Year", "Mean_Tone_Score", "Type", "LG", "Med", "Urb", "Region"]]


def version_summary(obs: pd.DataFrame, col: str) -> dict[str, float]:
    """Headline statistics for one PCSI construction on source-text university-years."""
    d = obs.dropna(subset=[col]).copy()
    grand = d[col].mean()
    between = (d.groupby("Institution")[col].transform("mean") - grand).pow(2).sum()
    total = (d[col] - grand).pow(2).sum()
    tr = d[(d["Year"] >= 1980)]
    tr = tr[tr.groupby("Institution")["Year"].transform("nunique") > 1]
    trend = cluster_fit(f"{col} ~ Year + C(Institution)", tr)
    inst = d.dropna(subset=["Urb", "Region"]).copy()
    inst["yr"] = inst["Year"].astype(int)
    reg = cluster_fit(f"{col} ~ {TYPE} + LG + Med + {URB} + {REG} + C(yr)", inst)
    term = f"{TYPE}[T.Private R1]"
    return {
        "Mean": round(grand, 3),
        "Share below 0.50": round((d[col] < 0.5).mean(), 3),
        "Between-university share": round(between / total, 3),
        "FE trend 1980+": round(trend.params["Year"], 5),
        "FE trend SE": round(trend.bse["Year"], 5),
        "FE trend p": round(trend.pvalues["Year"], 3),
        "Private R1 (year FE, full controls)": round(reg.params[term], 4),
        "Private R1 SE": round(reg.bse[term], 4),
        "Private R1 p": round(reg.pvalues[term], 3),
        "University-years": int(len(d)),
    }


def deontic_analysis(sentences: pd.DataFrame, panel: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    s = tag_deontic(sentences.copy())

    # (a) Coverage: how restrictive is each rule form?
    rows = []
    for label in DEONTIC_ORDER + ["Sanction language (any form)"]:
        mask = s["Sanction"] if label.startswith("Sanction") else s["Deontic"].eq(label)
        sub = s[mask]
        rows.append({"Rule form": label, "Share of sentences": round(mask.mean(), 3),
                     "Mean sentence score": round(sub["ToneScore_0_1"].mean(), 3),
                     "SD": round(sub["ToneScore_0_1"].std(), 3),
                     "Restrictive share": round(sub["Pred_Label"].eq("restrictive").mean(), 3),
                     "Supportive share": round(sub["Pred_Label"].eq("supportive").mean(), 3)})
    pd.DataFrame(rows).to_csv(outdir / "deontic_coverage.csv", index=False)

    # (b) Framing net of rule form: residualize sentence scores on the eight
    # deontic-by-sanction cells, then aggregate like PCSI.
    cell_means = s.groupby("Deontic x sanction")["ToneScore_0_1"].transform("mean")
    s["Framing_net_of_form"] = s["ToneScore_0_1"] - cell_means + s["ToneScore_0_1"].mean()
    uy = s.groupby(["Institution", "Year"]).agg(
        Framing_net_of_form=("Framing_net_of_form", "mean"),
        obligation=("Deontic", lambda x: (x == "Obligation").mean()),
        prohibition=("Deontic", lambda x: (x == "Prohibition").mean()),
        permission=("Deontic", lambda x: (x == "Permission").mean()),
        sanction=("Sanction", "mean"),
    ).reset_index()
    obs = observed_frame(panel).merge(uy, on=["Institution", "Year"], how="left")

    form_fit = smf.ols("Mean_Tone_Score ~ obligation + prohibition + permission + sanction", obs).fit(cov_type="HC3")
    with open(outdir / "deontic_form_regression.txt", "w") as f:
        f.write("PCSI on rule-form shares, source-text university-years (HC3)\n")
        f.write(form_fit.summary().as_text())
    corr = obs[["Mean_Tone_Score", "Framing_net_of_form"]].corr().iloc[0, 1]

    # (c) Shift-share over rule forms.
    ss, note = shift_share(s, "Deontic x sanction")
    ss.to_csv(outdir / "deontic_shift_share.csv")
    (outdir / "deontic_shift_share_note.txt").write_text(note)

    summary = pd.DataFrame(
        {
            "Statistic": ["R2 of PCSI on rule-form shares", "Correlation, PCSI and framing net of form",
                          "Sentence-level R2 of score on deontic x sanction cells"],
            "Value": [round(form_fit.rsquared, 3), round(corr, 3),
                      round(1 - ((s["ToneScore_0_1"] - cell_means) ** 2).sum()
                            / ((s["ToneScore_0_1"] - s["ToneScore_0_1"].mean()) ** 2).sum(), 3)],
        }
    )
    summary.to_csv(outdir / "deontic_summary.csv", index=False)
    return obs[["Institution", "Year", "Framing_net_of_form"]]


# ---------------------------------------------------------------------------
# 5. Classifier uncertainty
# ---------------------------------------------------------------------------
def uncertainty_analysis(sentences: pd.DataFrame, panel: pd.DataFrame, framing: pd.DataFrame, outdir: Path) -> None:
    s = sentences.copy()
    probs = s[["P_restrictive_calib", "P_neutral_calib", "P_supportive_calib"]]
    s["max_prob"] = probs.max(axis=1)
    s["hard_score"] = s["Pred_Label"].map({"restrictive": 0.0, "neutral": 0.5, "supportive": 1.0})

    # Confidence by decade: is the model less certain on older (possibly noisier) text?
    s["Decade"] = (s["Year"] // 10 * 10).astype(int)
    conf = s.groupby("Decade").agg(Sentences=("max_prob", "size"), Mean_max_probability=("max_prob", "mean"),
                                   Share_above_0_9=("max_prob", lambda x: (x >= 0.9).mean())).round(3)
    conf.to_csv(outdir / "classifier_confidence_by_decade.csv")

    uy = s.groupby(["Institution", "Year"]).agg(
        PCSI_hard_labels=("hard_score", "mean"),
    ).reset_index()
    hi = (s[s["max_prob"] >= 0.9].groupby(["Institution", "Year"])["ToneScore_0_1"].mean()
          .rename("PCSI_high_confidence").reset_index())
    obs = (observed_frame(panel).merge(uy, on=["Institution", "Year"], how="left")
           .merge(hi, on=["Institution", "Year"], how="left")
           .merge(framing, on=["Institution", "Year"], how="left"))

    versions = {
        "Baseline PCSI (calibrated probabilities)": "Mean_Tone_Score",
        "Hard labels": "PCSI_hard_labels",
        "High-confidence sentences only (max p >= 0.9)": "PCSI_high_confidence",
        "Framing net of rule form": "Framing_net_of_form",
    }
    rows = []
    for label, col in versions.items():
        row = {"Construction": label}
        row.update(version_summary(obs, col))
        if col != "Mean_Tone_Score":
            row["Correlation with baseline"] = round(obs[["Mean_Tone_Score", col]].corr().iloc[0, 1], 3)
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "alternative_constructions.csv", index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir")
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parents[2]
    outdir = Path(args.outdir) if args.outdir else package_root / "paper_outputs/tables/revision_analyses"
    figdir = package_root / "paper_outputs/figures"
    outdir.mkdir(parents=True, exist_ok=True)

    panel, sentences = load(package_root)
    institutional_robustness(panel, sentences, outdir)
    linguistic_shapley(panel, outdir)
    topic_decomposition(sentences, outdir, figdir)
    framing = deontic_analysis(sentences, panel, outdir)
    uncertainty_analysis(sentences, panel, framing, outdir)
    print(f"Revision analyses written to {outdir}")


if __name__ == "__main__":
    main()
