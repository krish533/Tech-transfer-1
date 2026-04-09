from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, pearsonr


PAL = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "red": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "grey": "#999999",
}
NMID = 0.50
TYPE_ORDER = ["Private R1", "Public R1", "Private R2", "Public R2"]
URBAN_ORDER = ["Rural", "Semi-Rural", "City"]
REGION_ORDER = ["West Coast", "Mid-Atlantic", "South", "Midwest", "Southwest", "Northeast"]

REGIONS = {
    "ME": "Northeast",
    "NH": "Northeast",
    "VT": "Northeast",
    "MA": "Northeast",
    "RI": "Northeast",
    "CT": "Northeast",
    "NY": "Mid-Atlantic",
    "NJ": "Mid-Atlantic",
    "PA": "Mid-Atlantic",
    "DE": "Mid-Atlantic",
    "MD": "Mid-Atlantic",
    "DC": "Mid-Atlantic",
    "OH": "Midwest",
    "MI": "Midwest",
    "IN": "Midwest",
    "IL": "Midwest",
    "WI": "Midwest",
    "MN": "Midwest",
    "IA": "Midwest",
    "MO": "Midwest",
    "ND": "Midwest",
    "SD": "Midwest",
    "NE": "Midwest",
    "KS": "Midwest",
    "VA": "South",
    "WV": "South",
    "KY": "South",
    "TN": "South",
    "NC": "South",
    "SC": "South",
    "GA": "South",
    "FL": "South",
    "AL": "South",
    "MS": "South",
    "AR": "South",
    "LA": "South",
    "OK": "South",
    "TX": "South",
    "MT": "Southwest",
    "ID": "Southwest",
    "WY": "Southwest",
    "CO": "Southwest",
    "NM": "Southwest",
    "AZ": "Southwest",
    "UT": "Southwest",
    "NV": "Southwest",
    "WA": "West Coast",
    "OR": "West Coast",
    "CA": "West Coast",
}

TYPE_COLOR_MAP = {
    "Private R1": "#D55E00",
    "Public R1": "#0072B2",
    "Private R2": "#CC79A7",
    "Public R2": "#009E73",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.35,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def latex_escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def save_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    latex_df = df.copy()
    tex = latex_df.to_latex(index=False, escape=False, na_rep="", float_format=lambda x: f"{x:.3f}")
    tex_path.write_text(tex, encoding="utf-8")


def savefig(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def institution_type(private_value, r1_value) -> str:
    private = "Private" if pd.notna(private_value) and int(float(private_value)) == 1 else "Public"
    research = "R1" if pd.notna(r1_value) and int(float(r1_value)) == 1 else "R2"
    return f"{private} {research}"


def yn(value) -> str:
    if pd.isna(value):
        return "--"
    return "Y" if int(float(value)) == 1 else "N"


def build_ranking_tex(ranked: pd.DataFrame, appendix_dir: Path) -> None:
    appendix_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        "",
        r"\begin{longtable}{@{}p{4.6cm}>{\centering\arraybackslash}p{0.7cm}>{\centering\arraybackslash}p{1.45cm}>{\centering\arraybackslash}p{0.9cm}>{\centering\arraybackslash}p{0.5cm}>{\centering\arraybackslash}p{1.0cm}>{\centering\arraybackslash}p{1.0cm}>{\centering\arraybackslash}p{0.75cm}>{\centering\arraybackslash}p{0.55cm}>{\centering\arraybackslash}p{0.85cm}>{\centering\arraybackslash}p{0.95cm}>{\centering\arraybackslash}p{0.85cm}@{}}",
        r"\caption{Institution-Level PCI Scores: Full Reference Table}",
        r"\label{tab:inst_pci} \\",
        r"\toprule",
        r"\textbf{Institution} & \textbf{St.} & \textbf{Type} & \textbf{Medical} & \textbf{LG} & \textbf{Mean PCI} & \textbf{Latest PCI} & \textbf{Yr} & \textbf{\textit{N}} & \textbf{Tone} & \textbf{Clarity} & \textbf{Legal} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{12}{l}{\textit{Table \ref{tab:inst_pci} continued from previous page}} \\",
        r"\toprule",
        r"\textbf{Institution} & \textbf{St.} & \textbf{Type} & \textbf{Medical} & \textbf{LG} & \textbf{Mean PCI} & \textbf{Latest PCI} & \textbf{Yr} & \textbf{\textit{N}} & \textbf{Tone} & \textbf{Clarity} & \textbf{Legal} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{12}{r}{\textit{Continued on next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in ranked.itertuples(index=False):
        cells = [
            latex_escape(row.Institution),
            latex_escape(row.STATE),
            latex_escape(row.Type),
            latex_escape(row.Med),
            latex_escape(row.LG),
            f"{row.Mean_PCI:.3f}",
            f"{row.Latest_PCI:.3f}",
            f"{int(row.Latest_Year)}",
            f"{int(row.N)}",
            f"{row.Tone:.2f}",
            f"{row.Clarity:.2f}",
            f"{row.Legal:.2f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    (appendix_dir / "ranking_fragment.tex").write_text("\n".join(lines), encoding="utf-8")

    standalone = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.6in]{geometry}",
        r"\usepackage{booktabs,longtable,array,pdflscape}",
        r"\begin{document}",
        r"\begin{landscape}",
        r"\input{ranking_fragment}",
        r"\end{landscape}",
        r"\end{document}",
    ]
    (appendix_dir / "Ranking.tex").write_text("\n".join(standalone), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices-file")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    data_dir = package_root / "data" / "derived"
    outputs_dir = package_root / "paper_outputs"
    fig_dir = outputs_dir / "figures"
    table_dir = outputs_dir / "tables"
    appendix_dir = outputs_dir / "appendix"
    log_path = outputs_dir / "paper_outputs_summary.txt"

    indices_path = Path(args.indices_file) if args.indices_file else data_dir / "policy_level_indices_institution_year.csv"
    df = pd.read_csv(indices_path)
    required_meta = {"STATE", "Private", "Carnegie R1", "MEDSCHOOL", "Urbanicity (cat)", "Land-Grant Institution"}
    if not required_meta.issubset(df.columns):
        missing = sorted(required_meta.difference(df.columns))
        raise ValueError(f"Indices file is missing required institution columns: {missing}")

    df["Urbanicity"] = df["Urbanicity (cat)"].replace({"Urban": "City", "Semi-Urban": "Semi-Rural"})
    df["Region"] = df["STATE"].map(REGIONS)
    df["Type"] = df.apply(lambda r: institution_type(r["Private"], r["Carnegie R1"]), axis=1)

    inst = (
        df.sort_values(["Institution", "Year"])
        .groupby("Institution", as_index=False)
        .agg(
            Mean_PCI=("Mean_Tone_Score", "mean"),
            Tone=("Tone_Index", "mean"),
            Clarity=("Clarity_Index", "mean"),
            Legal=("Legal_Load_Index", "mean"),
            N=("Year", "size"),
            Latest_Year=("Year", "max"),
            Latest_PCI=("Mean_Tone_Score", "last"),
            STATE=("STATE", "first"),
            Private=("Private", "first"),
            Carnegie_R1=("Carnegie R1", "first"),
            MEDSCHOOL=("MEDSCHOOL", "first"),
            Urbanicity=("Urbanicity", "first"),
            Land_Grant=("Land-Grant Institution", "first"),
            Region=("Region", "first"),
        )
    )
    inst["Type"] = inst.apply(lambda r: institution_type(r["Private"], r["Carnegie_R1"]), axis=1)
    inst["Med"] = inst["MEDSCHOOL"].apply(yn)
    inst["LG"] = inst["Land_Grant"].apply(yn)

    summary_lines = []

    # Table 5
    pci = df["Mean_Tone_Score"].dropna()
    t5 = pd.DataFrame(
        {
            "Statistic": [
                "Mean",
                "Median",
                "Standard Deviation",
                "Minimum",
                "Maximum",
                "Neutral midpoint",
                "Percent below midpoint",
                "Percent above midpoint",
                "Theoretical range",
                "Institution-year observations",
                "Institutions",
            ],
            "Value": [
                round(float(pci.mean()), 3),
                round(float(pci.median()), 3),
                round(float(pci.std()), 3),
                round(float(pci.min()), 3),
                round(float(pci.max()), 3),
                0.500,
                f"{(pci < 0.5).mean() * 100:.1f}%",
                f"{(pci >= 0.5).mean() * 100:.1f}%",
                "[0, 1]",
                int(len(pci)),
                int(df['Institution'].nunique()),
            ],
        }
    )
    save_table(t5, table_dir, "table5_pci_stats", "Descriptive Statistics for Policy-Level PCI Scores")
    summary_lines.append(f"Table 5 mean={pci.mean():.3f} median={pci.median():.3f} N={len(pci)} institutions={df['Institution'].nunique()}")

    # Table 6
    rows = []
    type_groups = [inst.loc[inst["Type"] == t, "Mean_PCI"].dropna() for t in TYPE_ORDER]
    kw_type = kruskal(*[g.values for g in type_groups if len(g) > 0])
    for i, (label, group) in enumerate(zip(TYPE_ORDER, type_groups)):
        rows.append(
            {
                "Group": label,
                "N": int(len(group)),
                "Mean": round(float(group.mean()), 3),
                "SD": round(float(group.std()), 3),
                "Range": f"[{group.min():.3f}, {group.max():.3f}]",
                "Test": "K-W" if i == 0 else "",
                "p-value": f"{kw_type.pvalue:.3f}{stars(kw_type.pvalue)}" if i == 0 else "",
            }
        )
    rows.append({"Group": "Urbanicity", "N": "", "Mean": "", "SD": "", "Range": "", "Test": "", "p-value": ""})
    urban_names = [u for u in URBAN_ORDER if u in set(inst["Urbanicity"].dropna())]
    urban_groups = [inst.loc[inst["Urbanicity"] == u, "Mean_PCI"].dropna() for u in urban_names]
    kw_urban = kruskal(*[g.values for g in urban_groups if len(g) > 0])
    for i, (label, group) in enumerate(zip(urban_names, urban_groups)):
        rows.append(
            {
                "Group": label,
                "N": int(len(group)),
                "Mean": round(float(group.mean()), 3),
                "SD": round(float(group.std()), 3),
                "Range": f"[{group.min():.3f}, {group.max():.3f}]",
                "Test": "K-W" if i == 0 else "",
                "p-value": f"{kw_urban.pvalue:.3f}{stars(kw_urban.pvalue)}" if i == 0 else "",
            }
        )
    rows.append({"Group": "Land-Grant Institution", "N": "", "Mean": "", "SD": "", "Range": "", "Test": "", "p-value": ""})
    lg_yes = inst.loc[inst["Land_Grant"] == 1, "Mean_PCI"].dropna()
    lg_no = inst.loc[inst["Land_Grant"] == 0, "Mean_PCI"].dropna()
    lg_test = mannwhitneyu(lg_yes, lg_no, alternative="two-sided")
    for i, (label, group) in enumerate([("Yes", lg_yes), ("No", lg_no)]):
        rows.append(
            {
                "Group": label,
                "N": int(len(group)),
                "Mean": round(float(group.mean()), 3),
                "SD": round(float(group.std()), 3),
                "Range": f"[{group.min():.3f}, {group.max():.3f}]",
                "Test": "Rank-sum" if i == 0 else "",
                "p-value": f"{lg_test.pvalue:.3f}{stars(lg_test.pvalue)}" if i == 0 else "",
            }
        )
    rows.append({"Group": "Medical School", "N": "", "Mean": "", "SD": "", "Range": "", "Test": "", "p-value": ""})
    med_yes = inst.loc[inst["MEDSCHOOL"] == 1, "Mean_PCI"].dropna()
    med_no = inst.loc[inst["MEDSCHOOL"] == 0, "Mean_PCI"].dropna()
    med_test = mannwhitneyu(med_yes, med_no, alternative="two-sided")
    for i, (label, group) in enumerate([("Yes", med_yes), ("No", med_no)]):
        rows.append(
            {
                "Group": label,
                "N": int(len(group)),
                "Mean": round(float(group.mean()), 3),
                "SD": round(float(group.std()), 3),
                "Range": f"[{group.min():.3f}, {group.max():.3f}]",
                "Test": "Rank-sum" if i == 0 else "",
                "p-value": f"{med_test.pvalue:.3f}{stars(med_test.pvalue)}" if i == 0 else "",
            }
        )
    t6 = pd.DataFrame(rows)
    save_table(t6, table_dir, "table6_crosssectional", "Cross-Sectional Variation in PCI Scores (Institution-Level)")
    summary_lines.append(f"Table 6 cross-sectional sample={len(inst)} institutions")

    # Table 8
    t8 = []
    for var, label in [
        ("Tone_Index", "Tone Index"),
        ("Clarity_Index", "Clarity Index"),
        ("Legal_Load_Index", "Legal Load Index"),
    ]:
        s = df[var].dropna()
        t8.append(
            {
                "Sub-Index": label,
                "Mean": round(float(s.mean()), 3),
                "SD": round(float(s.std()), 3),
                "Min": round(float(s.min()), 3),
                "Max": round(float(s.max()), 3),
                "N": int(len(s)),
            }
        )
    save_table(pd.DataFrame(t8), table_dir, "table8_subindices", "Descriptive Statistics for Linguistic Sub-Indices")

    # Table 9
    comp = df[["Mean_Tone_Score", "Tone_Index", "Clarity_Index", "Legal_Load_Index"]].dropna().rename(
        columns={"Mean_Tone_Score": "pci"}
    )
    m1 = smf.ols("pci ~ Tone_Index", data=comp).fit(cov_type="HC3")
    m2 = smf.ols("pci ~ Tone_Index + Clarity_Index", data=comp).fit(cov_type="HC3")
    m3 = smf.ols("pci ~ Tone_Index + Clarity_Index + Legal_Load_Index", data=comp).fit(cov_type="HC3")
    t9_rows = []
    for var, label in [
        ("Tone_Index", "Tone Index"),
        ("Clarity_Index", "Clarity Index"),
        ("Legal_Load_Index", "Legal Load Index"),
    ]:
        row = {"Variable": label}
        for model_name, model in [("1", m1), ("2", m2), ("3", m3)]:
            if var in model.params.index:
                row[f"Model {model_name}"] = f"{model.params[var]:.3f}{stars(model.pvalues[var])}"
                row[f"SE {model_name}"] = f"({model.bse[var]:.3f})"
            else:
                row[f"Model {model_name}"] = ""
                row[f"SE {model_name}"] = ""
        t9_rows.append(row)
    t9_rows.append(
        {
            "Variable": "R^2",
            "Model 1": f"{m1.rsquared:.3f}",
            "SE 1": "",
            "Model 2": f"{m2.rsquared:.3f}",
            "SE 2": "",
            "Model 3": f"{m3.rsquared:.3f}",
            "SE 3": "",
        }
    )
    t9_rows.append(
        {
            "Variable": "Observations",
            "Model 1": str(int(m1.nobs)),
            "SE 1": "",
            "Model 2": str(int(m2.nobs)),
            "SE 2": "",
            "Model 3": str(int(m3.nobs)),
            "SE 3": "",
        }
    )
    save_table(pd.DataFrame(t9_rows), table_dir, "table9_decomposition", "Variance Decomposition of the Policy Communication Index")
    summary_lines.append(f"Table 9 R2={m3.rsquared:.3f} N={int(m3.nobs)}")

    # Table 10
    comp["PCI Quintile"] = pd.qcut(comp["pci"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    q_prof = (
        comp.groupby("PCI Quintile", observed=True)
        .agg(
            Mean_PCI=("pci", "mean"),
            Tone_Index=("Tone_Index", "mean"),
            Clarity_Index=("Clarity_Index", "mean"),
            Legal_Load_Index=("Legal_Load_Index", "mean"),
            N=("pci", "size"),
        )
        .reset_index()
        .round(3)
    )
    save_table(q_prof, table_dir, "table10_quintile", "Mean Sub-Index Values by PCI Quintile")

    # Table 11
    reg = (
        inst.dropna(subset=["Region"])
        .groupby("Region", as_index=False)
        .agg(Mean_PCI=("Mean_PCI", "mean"), SD=("Mean_PCI", "std"), N=("Institution", "size"))
    )
    reg["Region"] = pd.Categorical(reg["Region"], categories=REGION_ORDER, ordered=True)
    reg = reg.sort_values("Mean_PCI", ascending=False).round(3)
    save_table(reg, table_dir, "table11_regional", "Institution-Level PCI by U.S. Region")

    # Figures
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(pci, bins=40, color=PAL["orange"], edgecolor="white", lw=0.4)
    ax.axvline(NMID, color="dimgrey", ls="--", lw=1.6, label="Neutral midpoint (0.50)")
    ax.axvline(pci.mean(), color=PAL["blue"], ls="-", lw=1.8, label=f"Mean ({pci.mean():.3f})")
    ax.axvline(pci.median(), color=PAL["green"], ls=":", lw=1.5, label=f"Median ({pci.median():.3f})")
    ax.set_xlabel("Policy Communication Index (PCI)")
    ax.set_ylabel("Number of institution-year observations")
    ax.set_title(f"Distribution of Policy-Level PCI Scores\nN = {len(pci):,} institution-year observations")
    ax.legend(framealpha=0.9, fontsize=9)
    ax.set_xlim(max(0.0, float(pci.min()) - 0.02), min(1.0, float(pci.max()) + 0.02))
    fig.tight_layout()
    savefig(fig, fig_dir, "fig1_pci_histogram")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    ax = axes[0]
    for i, t in enumerate(TYPE_ORDER):
        g = inst.loc[inst["Type"] == t, "Mean_PCI"].dropna().values
        if len(g) < 2:
            continue
        vp = ax.violinplot(g, [i], widths=0.65, showmedians=True)
        vp["bodies"][0].set_facecolor(TYPE_COLOR_MAP[t])
        vp["bodies"][0].set_alpha(0.72)
        vp["cmedians"].set_color("white")
        vp["cmedians"].set_lw(2)
    ax.axhline(NMID, color="dimgrey", ls="--", lw=1.2)
    ax.set_xticks(range(len(TYPE_ORDER)))
    ax.set_xticklabels(TYPE_ORDER, rotation=14, ha="right", fontsize=8)
    ax.set_ylabel("Institution-Level Mean PCI")
    ax.set_title(f"A. Institutional Type\n(K-W p={kw_type.pvalue:.3f})", fontweight="bold")

    ax = axes[1]
    vp = ax.violinplot([med_yes.values, med_no.values], [0, 1], widths=0.6, showmedians=True)
    for body, color in zip(vp["bodies"], [PAL["blue"], PAL["orange"]]):
        body.set_facecolor(color)
        body.set_alpha(0.72)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_lw(2)
    ax.axhline(NMID, color="dimgrey", ls="--", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Medical\n(N={len(med_yes)})", f"No medical\n(N={len(med_no)})"], fontsize=9)
    ax.set_title(f"B. Medical School\n(p={med_test.pvalue:.3f})", fontweight="bold")

    ax = axes[2]
    vp = ax.violinplot([lg_yes.values, lg_no.values], [0, 1], widths=0.6, showmedians=True)
    for body, color in zip(vp["bodies"], [PAL["red"], PAL["green"]]):
        body.set_facecolor(color)
        body.set_alpha(0.72)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_lw(2)
    ax.axhline(NMID, color="dimgrey", ls="--", lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Land-Grant\n(N={len(lg_yes)})", f"Non-Land-Grant\n(N={len(lg_no)})"], fontsize=9)
    ax.set_title(f"C. Land-Grant Status\n(p={lg_test.pvalue:.3f})", fontweight="bold")
    fig.suptitle("PCI Distribution by Institutional Characteristics", fontsize=11, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir, "fig2_violin_crosssectional")

    annual = (
        df[df["Year"].between(1991, 2023)]
        .groupby("Year")
        .agg(mean=("Mean_Tone_Score", "mean"), sem=("Mean_Tone_Score", lambda x: x.sem()), N=("Institution", "size"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(annual["Year"], annual["mean"] - 1.96 * annual["sem"], annual["mean"] + 1.96 * annual["sem"], alpha=0.15, color=PAL["blue"])
    ax.plot(annual["Year"], annual["mean"], color=PAL["blue"], lw=2.2, zorder=3, label="Annual mean PCI")
    ax.scatter(annual["Year"], annual["mean"], s=annual["N"] * 8, color=PAL["blue"], alpha=0.6, zorder=4, label="Dot size proportional to N")
    rolling = annual.set_index("Year")["mean"].rolling(3, min_periods=2).mean()
    ax.plot(rolling.index, rolling.values, color=PAL["orange"], lw=1.6, ls="-.", label="3-year rolling mean")
    ax.axhline(NMID, color="dimgrey", ls="--", lw=1.2, label="Neutral midpoint")
    ax.axvline(2011, color=PAL["red"], ls=":", lw=1.6, alpha=0.9)
    ax.text(2011.3, annual["mean"].min() + 0.01, "Stanford v. Roche\n(2011)", fontsize=8, color=PAL["red"], ha="left", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mean PCI")
    ax.set_title("Temporal Evolution of the Policy Communication Index")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(1990.5, 2023.5)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    fig.tight_layout()
    savefig(fig, fig_dir, "fig3_temporal_trend")

    fig, ax = plt.subplots(figsize=(10, 5))
    for t, color in [("Private R1", PAL["red"]), ("Public R1", PAL["blue"])]:
        ann_t = (
            df[df["Type"] == t]
            .groupby("Year")
            .agg(mean=("Mean_Tone_Score", "mean"), sem=("Mean_Tone_Score", lambda x: x.sem()), N=("Institution", "size"))
            .reset_index()
        )
        ann_t = ann_t[ann_t["N"] >= 2]
        ax.fill_between(ann_t["Year"], ann_t["mean"] - 1.96 * ann_t["sem"], ann_t["mean"] + 1.96 * ann_t["sem"], alpha=0.12, color=color)
        ax.plot(ann_t["Year"], ann_t["mean"], lw=2, color=color, label=t)
    ax.axhline(NMID, color="dimgrey", ls="--", lw=1.2)
    ax.axvline(2011, color=PAL["pink"], ls=":", lw=1.6, alpha=0.9, label="Stanford v. Roche (2011)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mean PCI")
    ax.set_title("PCI Trend by Institutional Type")
    ax.legend(fontsize=9)
    ax.set_xlim(1990, 2024)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    fig.tight_layout()
    savefig(fig, fig_dir, "fig4_trend_by_type")

    reg_plot = reg.sort_values("Mean_PCI")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(reg_plot["Region"], reg_plot["Mean_PCI"], color=PAL["blue"], alpha=0.82)
    for bar, row in zip(bars, reg_plot.itertuples(index=False)):
        ax.text(row.Mean_PCI + 0.002, bar.get_y() + bar.get_height() / 2, f"{row.Mean_PCI:.3f}  (N={int(row.N)})", va="center", fontsize=9, color="dimgrey")
    ax.axvline(NMID, color="dimgrey", ls="--", lw=1.3, label="Neutral midpoint (0.50)")
    ax.axvline(inst["Mean_PCI"].mean(), color=PAL["orange"], ls="-.", lw=1.3, label=f"Overall mean ({inst['Mean_PCI'].mean():.3f})")
    ax.set_xlabel("Mean Institution-Level PCI")
    ax.set_title("Institution-Level Mean PCI by U.S. Region")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig5_regional_bar")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, var, lbl, color in zip(
        axes,
        ["Tone_Index", "Clarity_Index", "Legal_Load_Index"],
        ["Tone Index", "Clarity Index", "Legal Load Index"],
        [PAL["blue"], PAL["green"], PAL["red"]],
    ):
        coef_v = m3.params.get(var, 0.0)
        partial_y = m3.resid + coef_v * comp[var]
        partial_x = comp[var]
        ax.scatter(partial_x, partial_y, alpha=0.20, s=10, color=color)
        slope, intercept = np.polyfit(partial_x, partial_y, 1)
        xr = np.linspace(partial_x.min(), partial_x.max(), 100)
        ax.plot(xr, slope * xr + intercept, color="black", lw=2)
        ax.set_xlabel(lbl)
        ax.set_ylabel("PCI (partial residual)")
        ax.set_title(f"PCI ~ {lbl}\ncoef={coef_v:.3f}{stars(m3.pvalues.get(var, 1.0))}", fontsize=10)
    fig.suptitle("Partial Regression Plots: Linguistic Dimensions and PCI", fontsize=11)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig6_partial_regression")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(q_prof))
    width = 0.22
    ax.bar(x - width, q_prof["Tone_Index"], width, label="Tone Index", color=PAL["blue"], alpha=0.85)
    ax.bar(x, q_prof["Clarity_Index"], width, label="Clarity Index", color=PAL["green"], alpha=0.85)
    ax.bar(x + width, q_prof["Legal_Load_Index"], width, label="Legal Load Index", color=PAL["red"], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n(PCI={v:.3f})" for q, v in zip(q_prof["PCI Quintile"], q_prof["Mean_PCI"])], fontsize=8.5)
    ax.set_ylabel("Mean Sub-Index (z-score)")
    ax.set_title("Linguistic Profile by PCI Quintile")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig7_quintile_profile")

    ranked = inst.sort_values("Mean_PCI").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, max(14, len(ranked) * 0.125)))
    colors = ranked["Type"].map(TYPE_COLOR_MAP).fillna(PAL["grey"])
    ax.scatter(ranked["Mean_PCI"], range(len(ranked)), color=colors, s=28, zorder=3, alpha=0.9)
    ax.axvline(NMID, color="dimgrey", ls="--", lw=1.1)
    ax.axvline(ranked["Mean_PCI"].mean(), color="black", ls="-.", lw=1.1)
    ax.set_yticks(range(len(ranked)))
    ax.set_yticklabels(ranked["Institution"].str[:42], fontsize=5.6)
    ax.set_xlabel("Mean PCI (institution-level average across institution-years)")
    ax.set_title("Institution-Level PCI Scores\nSorted by mean PCI", fontsize=10)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig8_ranked_institutions")

    annual_legal = (
        df[df["Year"].between(2000, 2023)]
        .groupby("Year")
        .agg(mean=("Legal_Load_Index", "mean"), sem=("Legal_Load_Index", lambda x: x.sem()), N=("Institution", "size"))
        .reset_index()
    )
    annual_legal = annual_legal[annual_legal["N"] >= 3]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.fill_between(annual_legal["Year"], annual_legal["mean"] - 1.96 * annual_legal["sem"], annual_legal["mean"] + 1.96 * annual_legal["sem"], alpha=0.15, color=PAL["red"])
    ax.plot(annual_legal["Year"], annual_legal["mean"], color=PAL["red"], lw=2.2, label="Annual mean Legal Load Index")
    ax.axhline(0, color="dimgrey", ls="--", lw=1.0, alpha=0.8, label="Corpus mean")
    ax.axvline(2011, color=PAL["blue"], ls=":", lw=1.6, alpha=0.9, label="Stanford v. Roche (2011)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Legal Load Index (z-score)")
    ax.set_title("Legal-Technical Density Over Time")
    ax.legend(fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir, "fig9_legal_load_time")

    # Ranking outputs
    ranking_cols = [
        "Institution",
        "STATE",
        "Type",
        "Med",
        "LG",
        "Mean_PCI",
        "Latest_PCI",
        "Latest_Year",
        "N",
        "Tone",
        "Clarity",
        "Legal",
    ]
    ranked[ranking_cols].to_csv(table_dir / "table_reference_ranking.csv", index=False)
    build_ranking_tex(ranked[ranking_cols], appendix_dir)

    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Generated manuscript-facing outputs under {outputs_dir}")


if __name__ == "__main__":
    main()
