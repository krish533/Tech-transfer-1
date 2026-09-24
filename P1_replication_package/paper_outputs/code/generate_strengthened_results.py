from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import kruskal, mannwhitneyu, trim_mean

PRIMARY_START = 1944
PRIMARY_END = 2025
COHORT_STARTS = (1990, 1995, 2000, 2005)
OBSERVED_TREND_STARTS = (1980, 1990, 2000)


def word_count(text: object) -> int:
    text = "" if pd.isna(text) else str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return len(text.split())


def save(df: pd.DataFrame, outdir: Path, name: str, index: bool = False) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / name, index=index)


def cluster_fit(formula: str, data: pd.DataFrame):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["Institution"]}
    )


def variance_shares(df: pd.DataFrame, value: str) -> tuple[float, float]:
    x = df[["Institution", value]].dropna().copy()
    grand = x[value].mean()
    grouped = x.groupby("Institution")[value]
    means = grouped.mean()
    counts = grouped.size()
    ss_between = float(((means - grand) ** 2 * counts).sum())
    centered = x[value] - x["Institution"].map(means)
    ss_within = float((centered**2).sum())
    total = ss_between + ss_within
    return ss_between / total, ss_within / total


def build_alternative_aggregations(sentences: pd.DataFrame, panel: pd.DataFrame):
    s = sentences.copy()
    s["sentence_length"] = s["Sentence_Cleaned"].map(word_count)
    groups = s.groupby(["Institution", "Year"], sort=False)

    obs = groups.agg(
        mean_pcsi=("ToneScore_0_1", "mean"),
        median_pcsi=("ToneScore_0_1", "median"),
    ).reset_index()

    trimmed = (
        groups["ToneScore_0_1"]
        .apply(lambda x: trim_mean(x.dropna().to_numpy(), 0.05))
        .rename("trim_pcsi")
        .reset_index()
    )

    def weighted(g: pd.DataFrame) -> float:
        values = g["ToneScore_0_1"].astype(float).to_numpy()
        weights = g["sentence_length"].astype(float).to_numpy()
        if weights.sum() == 0:
            return float(np.mean(values))
        return float(np.average(values, weights=weights))

    weighted_rows = []
    for (inst, year), g in groups:
        weighted_rows.append((inst, year, weighted(g)))
    weighted_df = pd.DataFrame(weighted_rows, columns=["Institution", "Year", "weighted_pcsi"])

    obs = obs.merge(trimmed, on=["Institution", "Year"], validate="one_to_one")
    obs = obs.merge(weighted_df, on=["Institution", "Year"], validate="one_to_one")
    obs = obs[(obs["Year"] >= PRIMARY_START) & (obs["Year"] <= PRIMARY_END)].copy()

    alt_panel = panel.merge(
        obs,
        left_on=["Institution", "Source_Year"],
        right_on=["Institution", "Year"],
        how="left",
        suffixes=("", "_observed"),
        validate="many_to_one",
    )
    alt_panel = alt_panel[(alt_panel["Year"] >= PRIMARY_START) & (alt_panel["Year"] <= PRIMARY_END)].copy()

    assert len(alt_panel) == 4277
    assert len(obs) == 480
    assert np.nanmax(np.abs(alt_panel["Mean_Tone_Score"] - alt_panel["mean_pcsi"])) < 1e-10
    assert np.nanmax(np.abs(alt_panel["Median_Tone_Score"] - alt_panel["median_pcsi"])) < 1e-10
    return obs, alt_panel


def distribution_output(panel: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in [("policy_in_force", panel), ("observed_only", observed)]:
        x = frame["Mean_Tone_Score"].dropna()
        rows.append(
            {
                "sample": label,
                "N": len(x),
                "mean": x.mean(),
                "median": x.median(),
                "sd": x.std(ddof=1),
                "minimum": x.min(),
                "maximum": x.max(),
                "share_below_050": (x < 0.50).mean(),
            }
        )
    return pd.DataFrame(rows)


def aggregation_outputs(obs_alt: pd.DataFrame, panel_alt: pd.DataFrame, outdir: Path) -> None:
    cols_panel = ["Mean_Tone_Score", "median_pcsi", "trim_pcsi", "weighted_pcsi"]
    labels = {
        "Mean_Tone_Score": "Mean PCSI",
        "median_pcsi": "Median PCSI",
        "trim_pcsi": "5% trimmed mean",
        "weighted_pcsi": "Length-weighted mean",
    }
    pearson = panel_alt[cols_panel].corr("pearson").rename(index=labels, columns=labels)
    spearman = panel_alt[cols_panel].corr("spearman").rename(index=labels, columns=labels)
    save(pearson, outdir, "pcsi_aggregation_pearson_4277.csv", index=True)
    save(spearman, outdir, "pcsi_aggregation_spearman_4277.csv", index=True)

    direct_cols = ["mean_pcsi", "median_pcsi", "trim_pcsi", "weighted_pcsi"]
    save(obs_alt[direct_cols].corr("pearson"), outdir, "pcsi_aggregation_pearson_direct480.csv", index=True)
    save(obs_alt[direct_cols].corr("spearman"), outdir, "pcsi_aggregation_spearman_direct480.csv", index=True)

    summary = []
    for col, name in labels.items():
        x = panel_alt[col]
        summary.append(
            {
                "Aggregation": name,
                "Mean": x.mean(),
                "Median": x.median(),
                "SD": x.std(ddof=1),
                "Min": x.min(),
                "Max": x.max(),
            }
        )
    save(pd.DataFrame(summary), outdir, "pcsi_aggregation_summary_4277.csv")

    baseline = panel_alt["Mean_Tone_Score"]
    baseline_side = baseline < 0.50
    agreement_rows = []
    for col, name in [("median_pcsi", "Median PCSI"), ("trim_pcsi", "5% trimmed mean"), ("weighted_pcsi", "Length-weighted mean")]:
        x = panel_alt[col]
        agreement_rows.append(
            {
                "Alternative": name,
                "Pearson_with_mean": baseline.corr(x, method="pearson"),
                "Spearman_with_mean": baseline.corr(x, method="spearman"),
                "Mean_alternative": x.mean(),
                "Mean_difference_vs_mean": (x - baseline).mean(),
                "Below_0.50_agreement": ((x < 0.50) == baseline_side).mean(),
            }
        )
    save(pd.DataFrame(agreement_rows), outdir, "pcsi_aggregation_agreement_4277.csv")


def institutional_summary(frame: pd.DataFrame, value_col: str) -> dict[str, float]:
    inst = frame.groupby("Institution", as_index=False).agg(
        value=(value_col, "mean"),
        Type=("Type", "first"),
        land=("Land-Grant Institution", "first"),
        med=("MEDSCHOOL", "first"),
    )
    type_groups = [inst.loc[inst["Type"] == t, "value"].dropna() for t in ["Private R1", "Public R1", "Private R2", "Public R2"]]
    priv = type_groups[0]
    pub = type_groups[1]
    land = inst.loc[inst["land"] == 1, "value"].dropna()
    nonland = inst.loc[inst["land"] == 0, "value"].dropna()
    med = inst.loc[inst["med"] == 1, "value"].dropna()
    nomed = inst.loc[inst["med"] == 0, "value"].dropna()
    return {
        "private_R1_mean": priv.mean(),
        "public_R1_mean": pub.mean(),
        "privateR1_minus_publicR1": priv.mean() - pub.mean(),
        "type_kw_p": kruskal(*[g.to_numpy() for g in type_groups]).pvalue,
        "landgrant_mean": land.mean(),
        "nonland_mean": nonland.mean(),
        "landgrant_minus_nonland": land.mean() - nonland.mean(),
        "landgrant_rank_p": mannwhitneyu(land, nonland, alternative="two-sided").pvalue,
        "medical_minus_no_med": med.mean() - nomed.mean(),
        "medical_rank_p": mannwhitneyu(med, nomed, alternative="two-sided").pvalue,
    }


def institutional_outputs(panel_alt: pd.DataFrame, observed_panel: pd.DataFrame, outdir: Path) -> None:
    specs = [
        ("pcsi", "Mean_Tone_Score"),
        ("median_alt", "median_pcsi"),
        ("trim_alt", "trim_pcsi"),
        ("weighted_alt", "weighted_pcsi"),
    ]
    rows = []
    for name, col in specs:
        r = {"measure": name}
        r.update(institutional_summary(panel_alt, col))
        rows.append(r)
    save(pd.DataFrame(rows), outdir, "institutional_aggregation_robustness.csv")

    inst = observed_panel.groupby("Institution", as_index=False).agg(
        value=("Mean_Tone_Score", "mean"),
        Type=("Type", "first"),
        land=("Land-Grant Institution", "first"),
        med=("MEDSCHOOL", "first"),
    )
    priv = inst.loc[inst["Type"] == "Private R1", "value"]
    pub = inst.loc[inst["Type"] == "Public R1", "value"]
    type_groups = [inst.loc[inst["Type"] == t, "value"].dropna() for t in ["Private R1", "Public R1", "Private R2", "Public R2"]]
    land = inst.loc[inst["land"] == 1, "value"]
    nonland = inst.loc[inst["land"] == 0, "value"]
    med = inst.loc[inst["med"] == 1, "value"]
    nomed = inst.loc[inst["med"] == 0, "value"]
    out = pd.DataFrame([{
        "privateR1_minus_publicR1": priv.mean() - pub.mean(),
        "privateR1_rank_p": mannwhitneyu(priv, pub, alternative="two-sided").pvalue,
        "landgrant_minus_nonland": land.mean() - nonland.mean(),
        "landgrant_rank_p": mannwhitneyu(land, nonland, alternative="two-sided").pvalue,
        "medical_minus_no_med": med.mean() - nomed.mean(),
        "medical_rank_p": mannwhitneyu(med, nomed, alternative="two-sided").pvalue,
        "type_kw_p": kruskal(*[g.to_numpy() for g in type_groups]).pvalue,
    }])
    save(out, outdir, "institutional_observed_only.csv")


def linguistic_outputs(panel: pd.DataFrame, outdir: Path) -> None:
    df = panel.copy()
    df["log_words"] = np.log(df["n_words"].clip(lower=1))
    df["log_sents"] = np.log(df["n_sentences"].clip(lower=1))
    formulas = {
        "base": "Mean_Tone_Score ~ Tone_Index + Clarity_Index + Legal_Load_Index",
        "plus_log_words": "Mean_Tone_Score ~ Tone_Index + Clarity_Index + Legal_Load_Index + log_words",
        "plus_log_sentences": "Mean_Tone_Score ~ Tone_Index + Clarity_Index + Legal_Load_Index + log_sents",
    }
    rows = []
    for model_name, formula in formulas.items():
        fit = cluster_fit(formula, df)
        for term in fit.params.index:
            if term == "Intercept":
                continue
            rows.append({
                "model": model_name,
                "term": term,
                "coef": fit.params[term],
                "se": fit.bse[term],
                "p": fit.pvalues[term],
                "r2": fit.rsquared,
                "N": int(fit.nobs),
            })
    save(pd.DataFrame(rows), outdir, "linguistic_length_sensitivity.csv")

    corr = panel[["Tone_Index", "Clarity_Index", "Legal_Load_Index"]].corr("pearson")
    save(corr, outdir, "transparent_index_correlations.csv", index=True)


def revision_outputs(observed: pd.DataFrame, outdir: Path) -> None:
    o = observed.sort_values(["Institution", "Year"]).copy()
    o["previous_year"] = o.groupby("Institution")["Year"].shift(1)
    o["previous_pcsi"] = o.groupby("Institution")["Mean_Tone_Score"].shift(1)
    r = o[o["previous_year"].notna()].copy()
    r["delta"] = r["Mean_Tone_Score"] - r["previous_pcsi"]
    r["gap_years"] = r["Year"] - r["previous_year"]
    cols = ["Institution", "previous_year", "Year", "previous_pcsi", "Mean_Tone_Score", "delta", "gap_years"]
    save(r[cols], outdir, "revision_changes.csv")
    summary = pd.DataFrame([{
        "transitions": len(r),
        "institutions_with_multiple_observed_policies": r["Institution"].nunique(),
        "mean_change": r["delta"].mean(),
        "median_change": r["delta"].median(),
        "mean_abs_change": r["delta"].abs().mean(),
        "median_abs_change": r["delta"].abs().median(),
        "share_negative": (r["delta"] < 0).mean(),
        "share_positive": (r["delta"] > 0).mean(),
        "share_abs_ge_002": (r["delta"].abs() >= 0.02).mean(),
        "share_abs_ge_005": (r["delta"].abs() >= 0.05).mean(),
        "median_year_gap": r["gap_years"].median(),
        "mean_year_gap": r["gap_years"].mean(),
    }])
    save(summary, outdir, "revision_change_summary.csv")


def temporal_outputs(panel_alt: pd.DataFrame, observed: pd.DataFrame, outdir: Path) -> None:
    cohort_rows = []
    for start in COHORT_STARTS:
        institutions = panel_alt.loc[panel_alt["Year"] == start, "Institution"].unique()
        sub = panel_alt[(panel_alt["Institution"].isin(institutions)) & (panel_alt["Year"].between(start, PRIMARY_END))].copy()
        fit = cluster_fit("Mean_Tone_Score ~ Year + C(Institution)", sub)
        cohort_rows.append({
            "cohort_start": start,
            "institutions": len(institutions),
            "mean_start": panel_alt[(panel_alt["Year"] == start) & (panel_alt["Institution"].isin(institutions))]["Mean_Tone_Score"].mean(),
            "mean_2025": panel_alt[(panel_alt["Year"] == PRIMARY_END) & (panel_alt["Institution"].isin(institutions))]["Mean_Tone_Score"].mean(),
            "change": panel_alt[(panel_alt["Year"] == PRIMARY_END) & (panel_alt["Institution"].isin(institutions))]["Mean_Tone_Score"].mean() - panel_alt[(panel_alt["Year"] == start) & (panel_alt["Institution"].isin(institutions))]["Mean_Tone_Score"].mean(),
            "fe_linear_slope": fit.params["Year"],
            "se": fit.bse["Year"],
            "p": fit.pvalues["Year"],
        })
    save(pd.DataFrame(cohort_rows), outdir, "temporal_common_cohorts.csv")

    observed_rows = []
    for start in OBSERVED_TREND_STARTS:
        sub = observed[observed["Year"] >= start].copy()
        pooled = cluster_fit("Mean_Tone_Score ~ Year", sub)
        fe = cluster_fit("Mean_Tone_Score ~ Year + C(Institution)", sub)
        observed_rows.append({
            "start": start,
            "N": len(sub),
            "pooled_slope": pooled.params["Year"],
            "pooled_se": pooled.bse["Year"],
            "pooled_p": pooled.pvalues["Year"],
            "institutionFE_slope": fe.params["Year"],
            "institutionFE_se": fe.bse["Year"],
            "institutionFE_p": fe.pvalues["Year"],
        })
    save(pd.DataFrame(observed_rows), outdir, "temporal_observed_only_trends.csv")

    dec = observed.copy()
    dec["decade"] = (dec["Year"] // 10 * 10).astype(int)
    decade = dec.groupby("decade")["Mean_Tone_Score"].agg(N="size", mean_pcsi="mean", median_pcsi="median").reset_index()
    save(decade, outdir, "observed_policy_decade_means.csv")

    institutions = panel_alt.loc[panel_alt["Year"] == 2000, "Institution"].unique()
    sub = panel_alt[(panel_alt["Institution"].isin(institutions)) & (panel_alt["Year"].between(2000, PRIMARY_END))].copy()
    agg_rows = []
    for col in ["Mean_Tone_Score", "median_pcsi", "trim_pcsi", "weighted_pcsi"]:
        fit = cluster_fit(f"{col} ~ Year + C(Institution)", sub)
        m0 = sub.loc[sub["Year"] == 2000, col].mean()
        m1 = sub.loc[sub["Year"] == PRIMARY_END, col].mean()
        agg_rows.append({
            "measure": col,
            "institutions": len(institutions),
            "mean_2000": m0,
            "mean_2025": m1,
            "change": m1 - m0,
            "fe_slope": fit.params["Year"],
            "se": fit.bse["Year"],
            "p": fit.pvalues["Year"],
        })
    save(pd.DataFrame(agg_rows), outdir, "temporal_aggregation_robustness.csv")

    y2025 = panel_alt[panel_alt["Year"] == PRIMARY_END].copy()
    y2025["policy_age"] = PRIMARY_END - y2025["Source_Year"]
    fresh_rows = []
    for age in [5, 10, 15]:
        x = y2025[y2025["policy_age"] <= age]["Mean_Tone_Score"]
        fresh_rows.append({"max_policy_age": age, "N": len(x), "mean": x.mean(), "median": x.median(), "share_below_050": (x < 0.50).mean()})
    x = y2025["Mean_Tone_Score"]
    fresh_rows.append({"max_policy_age": "all", "N": len(x), "mean": x.mean(), "median": x.median(), "share_below_050": (x < 0.50).mean()})
    save(pd.DataFrame(fresh_rows), outdir, "year2025_freshness.csv")


def variance_output(panel_alt: pd.DataFrame, obs_alt: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for name, frame, col in [
        ("baseline_panel", panel_alt, "Mean_Tone_Score"),
        ("median_panel", panel_alt, "median_pcsi"),
        ("trimmed_panel", panel_alt, "trim_pcsi"),
        ("weighted_panel", panel_alt, "weighted_pcsi"),
        ("baseline_observed", obs_alt, "mean_pcsi"),
        ("median_observed", obs_alt, "median_pcsi"),
        ("trimmed_observed", obs_alt, "trim_pcsi"),
        ("weighted_observed", obs_alt, "weighted_pcsi"),
    ]:
        b, w = variance_shares(frame, col)
        rows.append({"specification": name, "between_share": b, "within_share": w})
    save(pd.DataFrame(rows), outdir, "variance_decomposition.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate strengthened PCSI result and robustness outputs used in the final manuscript.")
    parser.add_argument("--panel-file")
    parser.add_argument("--sentences-file")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    panel_path = Path(args.panel_file) if args.panel_file else package_root / "data" / "derived" / "policy_level_indices_institution_year.csv"
    sentence_path = Path(args.sentences_file) if args.sentences_file else package_root / "data" / "derived" / "sentence_scores_canonical.csv"
    outdir = Path(args.outdir) if args.outdir else package_root / "paper_outputs" / "tables" / "strengthened_results"

    panel = pd.read_csv(panel_path, low_memory=False)
    sentences = pd.read_csv(sentence_path, low_memory=False)
    panel = panel[(panel["Year"] >= PRIMARY_START) & (panel["Year"] <= PRIMARY_END)].copy()
    observed = panel[panel["Is_Carried_Forward"] == 0].copy()

    if len(panel) != 4277 or panel["Institution"].nunique() != 150:
        raise ValueError(f"Expected primary panel N=4,277 and 150 institutions; found N={len(panel)}, institutions={panel['Institution'].nunique()}")
    if len(observed) != 480:
        raise ValueError(f"Expected 480 directly observed institution-year records; found {len(observed)}")

    obs_alt, panel_alt = build_alternative_aggregations(sentences, panel)

    save(distribution_output(panel, observed), outdir, "distribution_samples.csv")
    aggregation_outputs(obs_alt, panel_alt, outdir)
    institutional_outputs(panel_alt, observed, outdir)
    linguistic_outputs(panel, outdir)
    revision_outputs(observed, outdir)
    temporal_outputs(panel_alt, observed, outdir)
    variance_output(panel_alt, obs_alt, outdir)

    print(f"Strengthened manuscript outputs written to {outdir}")
    print("Primary panel:", len(panel), "rows; observed policies:", len(observed))


if __name__ == "__main__":
    main()
