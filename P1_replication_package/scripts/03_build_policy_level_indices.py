import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTIVE_VERBS = {
    "support", "assist", "help", "encourage", "facilitate", "guide", "advise", "work with",
    "partner", "collaborate", "welcome", "provide", "offer", "enable", "promote",
}
RESTRICTIVE_TERMS = {
    "shall", "must", "required", "require", "prohibit", "prohibited", "forbid", "forbidden",
    "not allowed", "may not", "no longer", "subject to", "reserves the right", "compliance",
    "obligation", "obligations",
}
SANCTION_TERMS = {
    "disciplinary action", "termination", "forfeiture", "liability", "legal remedies",
    "penalty", "penalties", "sanction", "breach", "violation",
}
PERMISSION_MODALS = {"may", "can", "should", "could", "might"}
OBLIGATION_MODALS = {"shall", "must", "will", "required"}
LEGALESE_TERMS = {
    "herein", "hereby", "hereinafter", "aforesaid", "pursuant to", "notwithstanding",
    "indemnify", "thereunder", "thereof", "whereas",
}
IP_TECH_TERMS = {
    "assignment", "assign", "royalty", "royalties", "equity", "patent", "patents",
    "prosecution", "infringement", "license", "licensing", "licence", "disclosure",
    "invention", "inventor", "copyright", "trademark", "trade secret",
}
PROCEDURAL_CUES = {
    "submit", "disclose", "contact", "notify", "complete", "fill out", "file a disclosure",
    "report", "must report", "must disclose", "within", "no later than", "at least",
}
SECOND_PERSON = {"you", "your", "yours"}
INCLUSIVE_WE = {"we", "our", "us", "our office", "our team"}
MADEY_2002_TERMS = {
    "madey", "duke", "experimental use", "research exemption", "non-commercial research",
    "infringement", "liability", "third-party rights", "third party rights",
}
ROCHE_2011_TERMS = {
    "stanford", "roche", "bayh-dole", "bayh dole", "ownership", "title",
    "hereby assign", "agree to assign", "present assignment", "inventor owns",
}

KEY_COLS = ["Institution", "Year", "File", "Sentence_Number", "Sentence_Cleaned"]
YEAR_COLS = ["Institution", "Year"]


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def lexicon_count(text: str, lexicon: set[str]) -> int:
    count = 0
    for term in lexicon:
        if " " in term:
            count += text.count(term)
        else:
            count += len(re.findall(rf"\b{re.escape(term)}\b", text))
    return count


def safe_rate(num: float, denom: float) -> float:
    return num / denom if denom > 0 else np.nan


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def add_sentence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ToneScore_0_1"] = df["P_supportive_calib"].fillna(0) + 0.5 * df["P_neutral_calib"].fillna(0)
    df["text"] = df["Sentence_Cleaned"].apply(normalize_text)
    df["sentence_length"] = df["text"].str.split().str.len()
    df["long_sentence"] = (df["sentence_length"] > 30).astype(int)

    feature_map = {
        "supportive_total": SUPPORTIVE_VERBS,
        "restrictive_total": RESTRICTIVE_TERMS,
        "sanction_total": SANCTION_TERMS,
        "second_person_total": SECOND_PERSON,
        "inclusive_we_total": INCLUSIVE_WE,
        "permission_modal_total": PERMISSION_MODALS,
        "obligation_modal_total": OBLIGATION_MODALS,
        "legalese_total": LEGALESE_TERMS,
        "iptech_total": IP_TECH_TERMS,
        "madey2002_total": MADEY_2002_TERMS,
        "roche2011_total": ROCHE_2011_TERMS,
    }
    for col, lexicon in feature_map.items():
        df[col] = df["text"].apply(lambda text: lexicon_count(text, lexicon))

    df["procedural_share_num"] = df["text"].apply(lambda text: int(any(term in text for term in PROCEDURAL_CUES)))
    return df


def aggregate_institution_year(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(YEAR_COLS, dropna=False)
    panel = grouped.agg(
        n_sentences=("text", "size"),
        n_words=("sentence_length", "sum"),
        Mean_Tone_Score=("ToneScore_0_1", "mean"),
        Median_Tone_Score=("ToneScore_0_1", "median"),
        mean_sentence_length=("sentence_length", "mean"),
        long_sentence_share=("long_sentence", "mean"),
        procedural_share=("procedural_share_num", "mean"),
        supportive_total=("supportive_total", "sum"),
        restrictive_total=("restrictive_total", "sum"),
        sanction_total=("sanction_total", "sum"),
        second_person_total=("second_person_total", "sum"),
        inclusive_we_total=("inclusive_we_total", "sum"),
        permission_modal_total=("permission_modal_total", "sum"),
        obligation_modal_total=("obligation_modal_total", "sum"),
        legalese_total=("legalese_total", "sum"),
        iptech_total=("iptech_total", "sum"),
        madey2002_total=("madey2002_total", "sum"),
        roche2011_total=("roche2011_total", "sum"),
        supportive_share=("Pred_Label", lambda s: (s == "supportive").mean()),
        neutral_share=("Pred_Label", lambda s: (s == "neutral").mean()),
        restrictive_share=("Pred_Label", lambda s: (s == "restrictive").mean()),
    ).reset_index()

    rate_specs = [
        ("supportive_total", "supportive_per_1000w"),
        ("restrictive_total", "restrictive_per_1000w"),
        ("sanction_total", "sanction_per_1000w"),
        ("second_person_total", "second_person_per_1000w"),
        ("inclusive_we_total", "inclusive_we_per_1000w"),
        ("legalese_total", "legalese_per_1000w"),
        ("iptech_total", "iptech_per_1000w"),
        ("madey2002_total", "madey2002_per_1000w"),
        ("roche2011_total", "roche2011_per_1000w"),
    ]
    for total_col, out_col in rate_specs:
        panel[out_col] = panel.apply(lambda row: safe_rate(row[total_col], row["n_words"]) * 1000, axis=1)

    panel["obligation_modal_share"] = panel.apply(
        lambda row: safe_rate(row["obligation_modal_total"], row["obligation_modal_total"] + row["permission_modal_total"]),
        axis=1,
    )

    tone_vars = [
        "supportive_per_1000w",
        "second_person_per_1000w",
        "inclusive_we_per_1000w",
        "restrictive_per_1000w",
        "sanction_per_1000w",
        "obligation_modal_share",
    ]
    for column in tone_vars:
        panel[f"{column}_z"] = zscore(panel[column])
    panel["Tone_Index"] = (
        panel["supportive_per_1000w_z"]
        + panel["second_person_per_1000w_z"]
        + panel["inclusive_we_per_1000w_z"]
        - panel["restrictive_per_1000w_z"]
        - panel["sanction_per_1000w_z"]
        - panel["obligation_modal_share_z"]
    ) / 6

    clarity_vars = ["mean_sentence_length", "long_sentence_share", "procedural_share"]
    for column in clarity_vars:
        panel[f"{column}_z"] = zscore(panel[column])
    panel["Clarity_Index"] = (
        -panel["mean_sentence_length_z"]
        - panel["long_sentence_share_z"]
        + panel["procedural_share_z"]
    ) / 3

    legal_vars = ["legalese_per_1000w", "iptech_per_1000w"]
    for column in legal_vars:
        panel[f"{column}_z"] = zscore(panel[column])
    panel["Legal_Load_Index"] = (panel["legalese_per_1000w_z"] + panel["iptech_per_1000w_z"]) / 2

    panel["Institution_Year_Key"] = panel["Institution"].astype(str) + " || " + panel["Year"].astype(str)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun-file", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rerun = pd.read_csv(args.rerun_file, low_memory=False)
    dedup = rerun.drop_duplicates(subset=KEY_COLS, keep="first").copy()
    dedup = add_sentence_features(dedup)
    inst_year_panel = aggregate_institution_year(dedup)

    sentence_keep_cols = [
        "Institution",
        "Year",
        "File",
        "Sentence_Number",
        "Sentence_Cleaned",
        "Cleaned_Char_Count",
        "P_restrictive_calib",
        "P_neutral_calib",
        "P_supportive_calib",
        "SRN_score",
        "ToneScore_0_1",
        "Pred_Label",
    ]
    inst_year_keep_cols = [
        "Institution_Year_Key",
        "Institution",
        "Year",
        "n_sentences",
        "n_words",
        "Mean_Tone_Score",
        "Median_Tone_Score",
        "supportive_share",
        "neutral_share",
        "restrictive_share",
        "supportive_per_1000w",
        "restrictive_per_1000w",
        "sanction_per_1000w",
        "second_person_per_1000w",
        "inclusive_we_per_1000w",
        "obligation_modal_share",
        "mean_sentence_length",
        "long_sentence_share",
        "procedural_share",
        "legalese_per_1000w",
        "iptech_per_1000w",
        "madey2002_per_1000w",
        "roche2011_per_1000w",
        "Tone_Index",
        "Clarity_Index",
        "Legal_Load_Index",
    ]

    dedup[sentence_keep_cols].sort_values(["Institution", "Year", "File", "Sentence_Number"]).to_csv(
        out_dir / "sentence_scores_canonical.csv", index=False
    )
    inst_year_panel[inst_year_keep_cols].sort_values(["Institution", "Year"]).to_csv(
        out_dir / "policy_level_indices_institution_year.csv", index=False
    )

    summary = {
        "sentence_rows_raw": int(len(rerun)),
        "sentence_rows_canonical": int(len(dedup)),
        "institutions": int(dedup["Institution"].nunique()),
        "institution_years": int(dedup[["Institution", "Year"]].drop_duplicates().shape[0]),
        "policy_documents_observed": int(dedup[["Institution", "Year", "File"]].drop_duplicates().shape[0]),
    }
    print(summary)


if __name__ == "__main__":
    main()
