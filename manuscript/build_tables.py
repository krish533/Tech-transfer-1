"""Write every manuscript table as a LaTeX fragment in manuscript/tables/.

All numbers come from the replication package: the primary 1944--2025 panel,
the canonical sentence scores, and the CSV outputs of
generate_strengthened_results.py and generate_revision_analyses.py. Run the
replication pipeline first, then, from the repository root:

    python manuscript/build_tables.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "P1_replication_package"
DERIVED = PKG / "data" / "derived"
STRONG = PKG / "paper_outputs" / "tables" / "strengthened_results"
REV = PKG / "paper_outputs" / "tables" / "revision_analyses"
MODEL = PKG / "model" / "srn_cls_model"
OUT = Path(__file__).resolve().parent / "tables"

TYPE_ORDER = ["Private R1", "Public R1", "Private R2", "Public R2"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def coef(b: float, p: float, d: int = 3) -> str:
    return f"{signed(b, d)}{stars(p)}"


def se(x: float, d: int = 3) -> str:
    return f"({x:.{d}f})"


def table(name: str, caption: str, label: str, colspec: str, header: list[str], body: list[str], notes: str,
          size: str = "\\small", wide: bool = False) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        size,
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{threeparttable}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
        *header,
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\begin{tablenotes}[flushleft]\\footnotesize",
        f"\\item \\textit{{Notes:}} {notes}",
        "\\end{tablenotes}",
        "\\end{threeparttable}",
        "\\end{table}",
    ]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def load():
    panel = pd.read_csv(DERIVED / "policy_level_indices_primary_1944_2025.csv", low_memory=False)
    assert len(panel) == 4277 and panel["Institution"].nunique() == 150
    panel["Urb"] = panel["Urbanicity (cat)"]
    sentences = pd.read_csv(DERIVED / "sentence_scores_canonical.csv", low_memory=False)
    sentences = sentences[sentences["Year"].between(1944, 2025)]
    return panel, sentences


# ---------------------------------------------------------------------------
def t_sample(panel, sentences):
    obs = panel[panel["Is_Carried_Forward"] == 0]
    docs = sentences.groupby(["Institution", "Year", "File"]).ngroups
    rows = [
        ("Universities", f"{panel['Institution'].nunique()}"),
        ("Analysis period", "1944--2025"),
        ("Source policy documents", f"{docs:,}"),
        ("Cleaned sentences", f"{len(sentences):,}"),
        ("University-years with source policy text", f"{len(obs):,}"),
        ("Annual policy-in-force university-years", f"{len(panel):,}"),
    ]
    table("tab_sample", "Corpus and Analysis Sample, 1944--2025", "tab:sample", "lr",
          ["Quantity & Value \\\\"], [f"{a} & {b} \\\\" for a, b in rows],
          "When several source documents are available for the same university and year, their sentences are pooled "
          "before the university-year score is constructed. In the annual panel, that score is carried forward until a "
          "newer policy source becomes available.")
    return docs


def t_examples(sentences):
    picks = [
        ("Supportive", "Inventors are encouraged to consult with the OTT"),
        ("Neutral", "Special procedures relevant to these processes"),
        ("Restrictive", "As a condition of affiliation with the University, members"),
        ("Supportive", "The University intends that the obligation for disclosure shall not unduly"),
        ("Restrictive", "Duty to Disclose Inventions to Univers ity Each Invention must be disclosed"),
    ]
    body, provenance = [], []
    for i, (cls, start) in enumerate(picks):
        row = sentences[sentences["Sentence_Cleaned"].str.startswith(start, na=False)].iloc[0]
        provenance.append({"Row": i + 1, "Class": cls, "Institution": row["Institution"], "Year": int(row["Year"]),
                           "File": row["File"], "Sentence_Number": row["Sentence_Number"],
                           "Sentence": row["Sentence_Cleaned"]})
        text = row["Sentence_Cleaned"]
        if start.startswith("Duty to Disclose"):
            text = "\\ldots\\ " + text.split("Univers ity ", 1)[1] + "."
        p = row[f"P_{cls.lower()}_calib"]
        if i == 3:
            body.append("\\midrule")
            body.append("\\multicolumn{3}{l}{\\textit{Two disclosure provisions, different framing}} \\\\")
        text = text.replace("%", "\\%").replace("&", "\\&")
        body.append(f"{cls} & ``{text}'' & {p:.3f} \\\\")
    pd.DataFrame(provenance).to_csv(REV / "table_examples_provenance.csv", index=False)
    table("tab_examples", "Illustrative Sentence-Level Communicative Stance", "tab:examples",
          "p{0.13\\textwidth}p{0.70\\textwidth}c",
          ["Class & Illustrative sentence & $\\hat p_{\\text{class}}$ \\\\"], body,
          "$\\hat p_{\\text{class}}$ is the calibrated model probability assigned to the displayed class. Source "
          "institutions and years are recorded in the replication package. The last two rows both concern invention "
          "disclosure but do not impose identical legal terms; they illustrate how provisions serving the same policy "
          "function can differ in stance. The examples illustrate the annotation construct; they are not an "
          "independent accuracy test.")


def t_distribution():
    d = pd.read_csv(STRONG / "distribution_samples.csv").set_index("sample")
    a, o = d.loc["policy_in_force"], d.loc["observed_only"]
    rows = [
        ("Mean", a["mean"], o["mean"]),
        ("Median", a["median"], o["median"]),
        ("Standard deviation", a["sd"], o["sd"]),
        ("Minimum", a["minimum"], o["minimum"]),
        ("Maximum", a["maximum"], o["maximum"]),
    ]
    q = pd.read_csv(DERIVED / "policy_level_indices_primary_1944_2025.csv")["Mean_Tone_Score"].quantile([0.25, 0.75])
    obs = pd.read_csv(DERIVED / "policy_level_indices_primary_1944_2025.csv")
    oq = obs.loc[obs["Is_Carried_Forward"] == 0, "Mean_Tone_Score"].quantile([0.25, 0.75])
    body = [f"{r} & {f3(x)} & {f3(y)} \\\\" for r, x, y in rows[:3]]
    body += [f"25th percentile & {f3(q[0.25])} & {f3(oq[0.25])} \\\\", f"75th percentile & {f3(q[0.75])} & {f3(oq[0.75])} \\\\"]
    body += [f"{r} & {f3(x)} & {f3(y)} \\\\" for r, x, y in rows[3:]]
    body += [f"Share below 0.50 & {a['share_below_050']*100:.1f}\\% & {o['share_below_050']*100:.1f}\\% \\\\",
             "\\midrule", f"Observations & {int(a['N']):,} & {int(o['N'])} \\\\"]
    table("tab_distribution", "Distribution of PCSI", "tab:dist", "lcc",
          [" & Policy-in-force panel & Years with source text \\\\"], body,
          "The policy-in-force panel carries each university's most recent score forward until a newer policy source "
          "appears; the second column uses only university-years for which policy text is directly available. "
          "Analysis window 1944--2025; 150 universities.")
    return a, o


def t_linguistic(panel):
    comp = panel.dropna(subset=["Mean_Tone_Score", "Tone_Index", "Clarity_Index", "Legal_Load_Index"])
    fits = [smf.ols(f, comp).fit(cov_type="cluster", cov_kwds={"groups": comp["Institution"]}) for f in [
        "Mean_Tone_Score ~ Tone_Index",
        "Mean_Tone_Score ~ Tone_Index + Clarity_Index",
        "Mean_Tone_Score ~ Tone_Index + Clarity_Index + Legal_Load_Index"]]
    body = []
    for var, lab in [("Tone_Index", "Tone Index"), ("Clarity_Index", "Clarity Index"), ("Legal_Load_Index", "Legal Load Index")]:
        body.append(lab + " & " + " & ".join(coef(f.params[var], f.pvalues[var]) if var in f.params else "" for f in fits) + " \\\\")
        body.append(" & " + " & ".join(se(f.bse[var]) if var in f.params else "" for f in fits) + " \\\\")
    sh = pd.read_csv(REV / "table6_shapley_r2.csv")
    p = sh[sh["Sample"].str.startswith("Policy")].set_index("Feature")["Share of explained"]
    o = sh[sh["Sample"].str.startswith("Source")].set_index("Feature")["Share of explained"]
    body += ["\\midrule",
             "$R^2$ & " + " & ".join(f3(f.rsquared) for f in fits) + " \\\\",
             f"Shapley share of $R^2$: Tone / Clarity / Legal Load & & & {p['Tone']:.2f} / {p['Clarity']:.2f} / {p['Legal Load']:.2f} \\\\",
             f"\\quad Same, years with source text & & & {o['Tone']:.2f} / {o['Clarity']:.2f} / {o['Legal Load']:.2f} \\\\",
             "Observations & " + " & ".join(f"{int(f.nobs):,}" for f in fits) + " \\\\"]
    table("tab_linguistic", "Interpreting PCSI with Transparent Linguistic Features", "tab:linguistic", "lccc",
          [" & (1) & (2) & (3) \\\\"], body,
          "Dependent variable is policy-in-force PCSI; 150 universities. Standard errors clustered by university are in "
          "parentheses. Shapley shares average each index's marginal contribution to $R^2$ over all orders of entry "
          "and sum to one. Each diagnostic combines standardized components, so one unit of a composite should not be "
          "read as one standard deviation of that composite. *** $p<0.01$; ** $p<0.05$; * $p<0.10$.")
    return fits


def institution_frame(panel):
    return panel.groupby("Institution").agg(
        pcsi=("Mean_Tone_Score", "mean"), Type=("Type", "first"), LG=("Land-Grant Institution", "first"),
        Med=("MEDSCHOOL", "first"), Urb=("Urb", "first")).reset_index()


def t_institution_means(panel):
    inst = institution_frame(panel)
    body = []

    def block(title, groups):
        body.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for lab, g in groups:
            body.append(f"\\quad {lab} & {len(g)} & {f3(g.mean())} & {f3(g.std())} & [{f3(g.min())}, {f3(g.max())}] \\\\")

    block("Carnegie classification and control", [(t, inst.loc[inst.Type == t, "pcsi"]) for t in TYPE_ORDER])
    block("Mission and program", [("Land-grant", inst.loc[inst.LG == 1, "pcsi"]), ("Non-land-grant", inst.loc[inst.LG == 0, "pcsi"]),
                                 ("Medical school", inst.loc[inst.Med == 1, "pcsi"]), ("No medical school", inst.loc[inst.Med == 0, "pcsi"])])
    block("Urbanicity", [("Rural", inst.loc[inst.Urb == "Rural", "pcsi"]), ("Semi-urban", inst.loc[inst.Urb == "Semi-Urban", "pcsi"]),
                         ("City", inst.loc[inst.Urb == "Urban", "pcsi"])])
    table("tab_inst_means", "University-Level PCSI by Institutional Characteristics", "tab:inst-means", "lcccc",
          ["Group & $N$ & Mean & SD & Range \\\\"], body,
          "Unit of analysis is the university; PCSI is averaged across each university's policy-in-force years before "
          "group statistics are computed. Five universities lack urbanicity data. Rank-based tests are reported in "
          "Appendix~\\ref{app:inst}.")

    # Rank tests + Holm (Appendix D).
    kw_type = kruskal(*[inst.loc[inst.Type == t, "pcsi"] for t in TYPE_ORDER]).pvalue
    lg = mannwhitneyu(inst.loc[inst.LG == 1, "pcsi"], inst.loc[inst.LG == 0, "pcsi"], alternative="two-sided").pvalue
    urb = kruskal(*[inst.loc[inst.Urb == u, "pcsi"] for u in ["Rural", "Semi-Urban", "Urban"]]).pvalue
    med = mannwhitneyu(inst.loc[inst.Med == 1, "pcsi"], inst.loc[inst.Med == 0, "pcsi"], alternative="two-sided").pvalue
    raw = [kw_type, lg, urb, med]
    holm = multipletests(raw, method="holm")[1]
    labels = [("Carnegie classification and control", "Kruskal--Wallis"), ("Land-grant status", "Wilcoxon rank-sum"),
              ("Urbanicity", "Kruskal--Wallis"), ("Medical-school status", "Wilcoxon rank-sum")]
    body = [f"{a} & {b} & {f3(r)} & {f3(h)} \\\\" for (a, b), r, h in zip(labels, raw, holm)]
    table("tab_holm", "Institutional Comparisons with Multiple-Testing Adjustment", "tab:holm", "llcc",
          ["Characteristic & Test & Raw $p$ & Holm $p$ \\\\"], body,
          "University-level mean PCSI ($N=150$; 145 for urbanicity). The Holm adjustment is applied to this family of "
          "four tests.")
    return dict(zip(["type", "lg", "urb", "med"], zip(raw, holm)))


def t_institution_reg(panel):
    rob = pd.read_csv(REV / "table5_institutional_robustness.csv")
    cols = [("Published (HC3)", "(1) Type only"), ("Published (HC3)", "(2) + LG, Med, Urbanicity"),
            ("Published (HC3)", "(3) + Region FE"), ("Source-text years + year FE (clustered)", "(3) + Region FE")]
    terms = ["Private R1", "Private R2", "Public R2", "Land-grant", "Medical school"]
    body = []
    for term in terms:
        cs, ss = [], []
        for sample, spec in cols:
            r = rob[(rob["Sample / estimator"] == sample) & (rob["Specification"] == spec) & (rob["Term"] == term)]
            if len(r):
                cs.append(coef(r["Coef"].iloc[0], r["p"].iloc[0]))
                ss.append(se(r["SE"].iloc[0]))
            else:
                cs.append("")
                ss.append("")
        body.append(f"{term} & " + " & ".join(cs) + " \\\\")
        body.append(" & " + " & ".join(ss) + " \\\\")
    meta = []
    for sample, spec in cols:
        r = rob[(rob["Sample / estimator"] == sample) & (rob["Specification"] == spec)].iloc[0]
        meta.append(r)
    body += ["\\midrule",
             "Urbanicity controls & No & Yes & Yes & Yes \\\\",
             "Region fixed effects & No & No & Yes & Yes \\\\",
             "Source-year fixed effects & No & No & No & Yes \\\\",
             "$R^2$ & " + " & ".join(f3(m["R2"]) for m in meta) + " \\\\",
             "Observations & " + " & ".join(str(int(m["N"])) for m in meta) + " \\\\",
             "Universities & 150 & 145 & 144 & 144 \\\\"]
    table("tab_inst_reg", "Institutional Correlates of PCSI", "tab:inst-reg", "lcccc",
          [" & \\multicolumn{3}{c}{University means} & Source-text years \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}",
           " & (1) & (2) & (3) & (4) \\\\"], body,
          "Dependent variable is PCSI. Columns (1)--(3) use one observation per university (mean policy-in-force PCSI) "
          "with HC3 robust standard errors; Columns (2)--(3) omit universities with missing covariates. Column (4) uses "
          "university-years with directly available policy text, compares policies written in the same year through "
          "source-year fixed effects, and clusters standard errors by university. Public R1, rural, and Midwest are the "
          "omitted categories. *** $p<0.01$; ** $p<0.05$; * $p<0.10$.")


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------
def t_classifier():
    cfg = json.loads((MODEL / "config.json").read_text())
    met = json.loads((MODEL / "metrics.json").read_text())
    temp = json.loads((MODEL / "temperature_scaling.json").read_text())["temperature"]
    tok = json.loads((MODEL / "tokenizer_config.json").read_text())
    maxlen = tok.get("model_max_length", 256)
    maxlen = 256 if maxlen > 10000 else maxlen
    rows = [("Base model", "\\texttt{bert-base-uncased}"), ("Transformer layers", cfg["num_hidden_layers"]),
            ("Attention heads", cfg["num_attention_heads"]), ("Hidden size", cfg["hidden_size"]),
            ("Intermediate size", f"{cfg['intermediate_size']:,}"), ("Maximum input length", "256 tokens"),
            ("Output classes", len(cfg["id2label"])), ("Training sentences", f"{met['n_train']:,}"),
            ("Validation sentences", met["n_val"]), ("Total labeled sentences", f"{met['n_train'] + met['n_val']:,}"),
            ("Temperature parameter", f"$T = {temp:.4f}$"), ("Validation accuracy", f"{met['accuracy']:.3f}"),
            ("Validation macro-F1", f"{met['f1_macro']:.3f}")]
    table("tab_classifier", "Preserved Classifier Configuration", "tab:classifier", "lr",
          ["Parameter & Value \\\\"], [f"{a} & {b} \\\\" for a, b in rows],
          "Configuration and metrics are read from the preserved model artifacts in the replication package.")


def t_aggregation():
    ag = pd.read_csv(STRONG / "pcsi_aggregation_agreement_4277.csv")
    body = ["Baseline (unweighted mean) & 0.437 & 1.000 & 1.000 & 1.000 \\\\"]
    for _, r in ag.iterrows():
        body.append(f"{r['Alternative'].replace('%', '\\%')} & {f3(r['Mean_alternative'])} & {f3(r['Pearson_with_mean'])} & "
                    f"{f3(r['Spearman_with_mean'])} & {f3(r['Below_0.50_agreement'])} \\\\")
    table("tab_aggregation", "Sensitivity to Alternative Sentence Aggregation Rules", "tab:aggregation", "lcccc",
          ["Aggregation & Mean & Pearson $r$ & Spearman $\\rho$ & Midpoint agreement \\\\"], body,
          "$N = 4{,}277$ policy-in-force university-years. Correlations are with baseline PCSI. Midpoint agreement is "
          "the share of observations that the baseline and the alternative place on the same side of 0.50.")
    vd = pd.read_csv(STRONG / "variance_decomposition.csv").set_index("specification")
    labels = {"baseline": "Baseline", "median": "Median", "trimmed": "Trimmed mean", "weighted": "Length-weighted mean"}
    body = []
    for sample, title in [("panel", "Policy-in-force panel"), ("observed", "Years with source text")]:
        body.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{{title}}}}} \\\\")
        for k, lab in labels.items():
            r = vd.loc[f"{k}_{sample}"]
            body.append(f"\\quad {lab} & {f3(r['between_share'])} & {f3(r['within_share'])} \\\\")
    table("tab_variance", "Between- and Within-University Variation", "tab:variance", "lcc",
          ["Specification & Between share & Within share \\\\"], body,
          "Shares partition the total sum of squared deviations of PCSI into between- and within-university "
          "components.")


def t_length():
    d = pd.read_csv(STRONG / "linguistic_length_sensitivity.csv")
    models = ["base", "plus_log_words", "plus_log_sentences"]
    body = []
    for var, lab in [("Tone_Index", "Tone Index"), ("Clarity_Index", "Clarity Index"), ("Legal_Load_Index", "Legal Load Index")]:
        rs = [d[(d.model == m) & (d.term == var)].iloc[0] for m in models]
        body.append(f"{lab} & " + " & ".join(coef(r.coef, r.p) for r in rs) + " \\\\")
        body.append(" & " + " & ".join(se(r.se) for r in rs) + " \\\\")
    r2 = [d[d.model == m].r2.iloc[0] for m in models]
    body += ["\\midrule", "Policy-length control & No & Log words & Log sentences \\\\",
             "$R^2$ & " + " & ".join(f3(x) for x in r2) + " \\\\", "Observations & 4,277 & 4,277 & 4,277 \\\\"]
    table("tab_length", "Linguistic Decomposition with Policy-Length Controls", "tab:length", "lccc",
          [" & Baseline & + Log words & + Log sentences \\\\"], body,
          "University-clustered standard errors in parentheses. *** $p<0.01$; ** $p<0.05$; * $p<0.10$.")


def t_temporal():
    c = pd.read_csv(STRONG / "temporal_common_cohorts.csv")
    body = [f"{int(r.cohort_start)} & {int(r.institutions)} & {f3(r.mean_start)} & {f3(r.mean_2025)} & "
            f"{signed(r.fe_linear_slope, 5)} & {f3(r.p)} \\\\" for r in c.itertuples()]
    table("tab_cohorts", "PCSI Trends Within Fixed University Cohorts", "tab:cohorts", "cccccc",
          ["Cohort start & Universities & Start mean & 2025 mean & FE trend per year & $p$ \\\\"], body,
          "Each cohort contains the universities already represented in the policy-in-force panel by the start year. "
          "Trends come from regressions of PCSI on year with university fixed effects; standard errors are clustered "
          "by university.")
    o = pd.read_csv(STRONG / "temporal_observed_only_trends.csv")
    body = [f"{int(r.start)} & {int(r.N)} & {signed(r.pooled_slope, 5)} & {signed(r.institutionFE_slope, 5)} & "
            f"{se(r.institutionFE_se, 5)} & {f3(r.institutionFE_p)} \\\\" for r in o.itertuples()]
    table("tab_observed_trends", "Temporal Trends Using Only Years with Source Policy Text", "tab:observed-trends", "cccccc",
          ["Start year & $N$ & Pooled trend & FE trend & SE & $p$ \\\\"], body,
          "University-years with directly available policy text from the start year onward. Standard errors are "
          "clustered by university.")


def t_inst_robust():
    rob = pd.read_csv(REV / "table5_institutional_robustness.csv")
    rows = [("Published (HC3)", "University means"), ("Common 144 sample (HC3)", "University means, common sample"),
            ("WLS: weight = source policy years (HC3)", "WLS, weight = source policy years"),
            ("WLS: weight = sentences (HC3)", "WLS, weight = sentences"),
            ("Era-adjusted means: year-demeaned panel (HC3)", "Era-adjusted university means"),
            ("Source-text years + year FE (clustered)", "Source-text years, year FE"),
            ("Source-text years 1980+ + year FE (clustered)", "Source-text years from 1980, year FE")]

    def get(sample, spec, term):
        return rob[(rob["Sample / estimator"] == sample) & (rob["Specification"] == spec) & (rob["Term"] == term)].iloc[0]

    body = []
    for sample, lab in rows:
        a = get(sample, "(1) Type only", "Private R1")
        b = get(sample, "(3) + Region FE", "Private R1")
        c = get(sample, "(3) + Region FE", "Land-grant")
        body.append(f"{lab} & {coef(a.Coef, a.p)} & {coef(b.Coef, b.p)} & {coef(c.Coef, c.p)} & {int(b.N)} \\\\")
        body.append(f" & {se(a.SE)} & {se(b.SE)} & {se(c.SE)} & \\\\")
    table("tab_inst_robust", "Private-R1 and Land-Grant Coefficients Across Samples and Estimators", "tab:inst-robust",
          "lcccc", [" & Type only & \\multicolumn{2}{c}{Full controls} & \\\\", "\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}",
                    " & Private R1 & Private R1 & Land-grant & $N$ \\\\"], body,
          "Dependent variable is PCSI; public R1 is the omitted type. ``Full controls'' adds land-grant, medical-school, "
          "and urbanicity indicators and region fixed effects. University-mean rows use HC3 standard errors; the "
          "published row uses 150 universities for the type-only column and 144 with full controls. Era-adjusted means "
          "subtract each calendar year's cross-sectional mean from the policy-in-force panel before averaging within "
          "university. Source-text rows use university-years with directly available policy text, include source-year "
          "fixed effects, and cluster standard errors by university. *** $p<0.01$; ** $p<0.05$; * $p<0.10$.")


def t_topics():
    tr = pd.read_csv(REV / "topic_within_trends.csv")
    cov = pd.read_csv(REV / "topic_coverage.csv").set_index("Topic")
    body = []
    for r in tr.itertuples():
        topic = r.Topic.replace("&", "\\&")
        if r.Topic.startswith("All"):
            body.append("\\midrule")
            share, mean = "1.000", ""
        else:
            share = f3(cov.loc[r.Topic, "Share of sentences (multi-label)"])
            mean = f3(cov.loc[r.Topic, "Mean sentence score"])
        body.append(f"{topic} & {share} & {mean} & {signed(r[2], 5)} & {se(r.SE, 5)} & {f3(r.p)} \\\\")
    table("tab_topics", "Within-University Trends in Sentence Stance by Clause Topic, 1980--2025", "tab:topics",
          "lccccc", ["Topic & Share & Mean score & FE trend per year & SE & $p$ \\\\"], body,
          "Each row regresses the university-year mean sentence score within a topic on year and university fixed "
          "effects, using source-text university-years from 1980 and universities observed in at least two such years; "
          "standard errors are clustered by university. Topics are assigned by keyword rules and can overlap; share is "
          "the fraction of 1980--2025 sentences matching the topic. Within-topic trends allow overlapping keyword "
          "matches. The exact shift-share calculation instead uses mutually exclusive primary topics and is reported "
          "separately in Table~\\ref{tab:decomp}.")


def t_rule_form():
    cov = pd.read_csv(REV / "deontic_coverage.csv")
    summ = pd.read_csv(REV / "deontic_summary.csv").set_index("Statistic")["Value"]
    body = []
    for r in cov.itertuples():
        lab = r[1]
        if lab.startswith("Sanction"):
            body.append("\\midrule")
        body.append(f"{lab} & {f3(r[2])} & {f3(r[3])} & {f3(r[5])} & {f3(r[6])} \\\\")
    table("tab_rule_form", "Communicative Stance by Rule Form", "tab:rule-form", "lcccc",
          ["Rule form & Share & Mean score & Restrictive & Supportive \\\\"], body,
          "Deontic operators follow the Institutional Grammar of \\citet{crawford1995grammar} and are assigned by "
          "keyword rules with priority prohibition $>$ obligation $>$ permission (Appendix~\\ref{app:topics}). "
          "Sanction language (the grammar's ``or else'' component) can co-occur with any form. Restrictive and "
          "supportive shares use the classifier's most likely label. Rule form explains "
          f"{summ['Sentence-level R2 of score on deontic x sanction cells']*100:.1f} percent of sentence-level variance "
          "in stance. The exact 63-university within-cell and composition changes are reported in "
          "Table~\\ref{tab:decomp}.")


def t_decomposition():
    """Exact shift-share table (Appendix H), from generate_revision_analyses.py."""
    cols = {}
    for key, path, idx in [("topic", "topic_shift_share.csv", "Primary topic"),
                           ("form", "deontic_shift_share.csv", "Deontic x sanction")]:
        d = pd.read_csv(REV / path).set_index(idx)["Within contribution"]
        tot = pd.read_csv(REV / path).set_index(idx).loc["Total"]
        cols[key] = {"early": d["Early mean (university average)"], "late": d["Late mean (university average)"],
                     "change": d["Observed change"], "within": tot["Within contribution"],
                     "comp": tot["Composition contribution"]}
    note = (REV / "topic_shift_share_note.txt").read_text()
    n_uni = int(note.split(":")[1].split()[0])
    early_n = int(note.split("early=")[1].split(",")[0])
    late_n = int(note.split("late=")[1].split()[0])
    a, b = cols["topic"], cols["form"]
    six = lambda x: f"${x:.6f}$" if x < 0 else f"{x:.6f}"
    pct = lambda x, tot: f"{100 * x / tot:.1f}\\%"
    body = [
        f"Early period: mean score (1980--1999) & {a['early']:.6f} & {b['early']:.6f} \\\\",
        f"Late period: mean score (2010--2025) & {a['late']:.6f} & {b['late']:.6f} \\\\",
        f"\\textbf{{Observed change}} & \\textbf{{{six(a['change'])}}} & \\textbf{{{six(b['change'])}}} \\\\",
        f"Within-category contribution & {six(a['within'])} & {six(b['within'])} \\\\",
        f"Composition contribution & {six(a['comp'])} & {six(b['comp'])} \\\\",
        f"Within-category share of decline & {pct(a['within'], a['change'])} & {pct(b['within'], b['change'])} \\\\",
        f"Composition share of decline & {pct(a['comp'], a['change'])} & {pct(b['comp'], b['change'])} \\\\",
        f"Universities with policies in both periods & {n_uni} & {n_uni} \\\\",
        f"Early and late sentences & {early_n:,}; {late_n:,} & {early_n:,}; {late_n:,} \\\\",
    ]
    table("tab_decomposition", "Exact Shift--Share Decompositions of the Modern PCSI Decline", "tab:decomp", "lrr",
          [" & Exclusive primary topic & Deontic $\\times$ sanction \\\\"], body,
          "The unit of comparison is a university appearing in both periods; universities are weighted equally. "
          "Each sentence is assigned to one primary topic by fixed keyword priority, or to one of eight "
          "deontic-by-sanction cells. Category changes are decomposed within each university using midpoint shares "
          "and midpoint conditional means, then averaged. If a category is absent in a period, its conditional mean "
          "is imputed from the observed period, assigning category entry/exit to composition. The university-level "
          "accounting identity holds to machine precision. These alternative decompositions are not additive and are "
          "sensitive to the coding rules and missing-category convention. The numbers are descriptive, not causal "
          "estimates.")


def t_alternatives():
    d = pd.read_csv(REV / "alternative_constructions.csv")
    body = []
    for _, r in d.iterrows():
        corr = "" if pd.isna(r["Correlation with baseline"]) else f3(r["Correlation with baseline"])
        short = {"Baseline PCSI (calibrated probabilities)": "Baseline PCSI",
                 "High-confidence sentences only (max p >= 0.9)": "High-confidence sentences"}
        body.append(f"{short.get(r['Construction'], r['Construction'])} & {corr} & {f3(r['Mean'])} & {f3(r['Share below 0.50'])} & "
                    f"{f3(r['Between-university share'])} & {coef(r['FE trend 1980+'], r['FE trend p'], 5)} & "
                    f"{coef(r['Private R1 (year FE, full controls)'], r['Private R1 p'])} \\\\")
        body.append(f" & & & & & {se(r['FE trend SE'], 5)} & {se(r['Private R1 SE'])} \\\\")
    table("tab_alternatives", "Headline Results Under Alternative Constructions of PCSI", "tab:alternatives",
          "lcccccc",
          [" & $r$ with & & Share & Between & FE trend, & \\\\",
           "Construction & baseline & Mean & $<0.50$ & share & 1980+ & Private R1 \\\\"],
          body,
          "All columns use the 480 university-years with directly available policy text. Hard labels score each "
          "sentence 0, 0.5, or 1 by its most likely class. High-confidence uses only sentences whose largest calibrated "
          "class probability is at least 0.9 (about two-thirds of sentences). Framing net of rule form subtracts the "
          "mean score of each sentence's deontic-by-sanction cell before aggregation. The trend is the coefficient on "
          "year with university fixed effects, using universities with at least two source-text years from 1980; the private-R1 coefficient is from the Column (4) specification of "
          "Table~\\ref{tab:inst-reg}. Standard errors clustered by university. *** $p<0.01$; ** $p<0.05$; * $p<0.10$.",
          size="\\footnotesize")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    panel, sentences = load()
    t_sample(panel, sentences)
    t_examples(sentences)
    t_distribution()
    t_linguistic(panel)
    holm = t_institution_means(panel)
    t_institution_reg(panel)
    t_classifier()
    t_aggregation()
    t_length()
    t_temporal()
    t_inst_robust()
    t_topics()
    t_rule_form()
    t_decomposition()
    t_alternatives()
    figdir = Path(__file__).resolve().parent / "figures"
    figdir.mkdir(exist_ok=True)
    for name in ["fig1_pci_histogram.pdf", "fig3_temporal_trend_pcsi.pdf", "fig_topic_trends.pdf"]:
        shutil.copy2(PKG / "paper_outputs" / "figures" / name, figdir / name)
    print("Holm:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in holm.items()})
    print(f"Tables written to {OUT}")


if __name__ == "__main__":
    main()
