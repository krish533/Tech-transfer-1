"""Inter-rater reliability for the PCSI annotation construct.

The original coder-level pre-adjudication records are not preserved, so
reliability is assessed on a fresh random sample of corpus sentences.

Step 1 -- draw a coding sheet (sentences only, no model predictions shown):
    python paper_outputs/code/compute_interrater_reliability.py draw --n 400 --seed 20261019

    Writes data/annotation/irr_coding_sheet.csv with columns
    sentence_id, sentence, coder1, coder2, adjudicated.
    Two coders fill coder1 and coder2 independently with one of
    restrictive / neutral / supportive; disagreements are adjudicated in
    the adjudicated column (leave it blank where the coders agree).

Step 2 -- score the completed sheet:
    python paper_outputs/code/compute_interrater_reliability.py score

    Writes paper_outputs/tables/revision_analyses/interrater_reliability.csv
    and ../manuscript/irr_numbers.tex, which the manuscript reads
    automatically to fill in the reliability numbers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
SENTENCES = PKG / "data" / "derived" / "sentence_scores_canonical.csv"
SHEET = PKG / "data" / "annotation" / "irr_coding_sheet.csv"
OUT_CSV = PKG / "paper_outputs" / "tables" / "revision_analyses" / "interrater_reliability.csv"
OUT_TEX = PKG.parent / "manuscript" / "irr_numbers.tex"
CLASSES = ["restrictive", "neutral", "supportive"]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in CLASSES)
    return (po - pe) / (1 - pe)


def macro_f1(truth: pd.Series, pred: pd.Series) -> float:
    f1s = []
    for c in CLASSES:
        tp = ((pred == c) & (truth == c)).sum()
        fp = ((pred == c) & (truth != c)).sum()
        fn = ((pred != c) & (truth == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def draw(n: int, seed: int) -> None:
    s = pd.read_csv(SENTENCES, low_memory=False)
    s = s[s["Year"].between(1944, 2025)].reset_index().rename(columns={"index": "sentence_id"})
    sample = s.sample(n=n, random_state=seed)[["sentence_id", "Sentence_Cleaned"]].rename(columns={"Sentence_Cleaned": "sentence"})
    sample["coder1"] = ""
    sample["coder2"] = ""
    sample["adjudicated"] = ""
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SHEET, index=False)
    print(f"Wrote {len(sample)} sentences to {SHEET}")


def score() -> None:
    sheet = pd.read_csv(SHEET)
    for col in ["coder1", "coder2", "adjudicated"]:
        sheet[col] = sheet[col].fillna("").astype(str).str.strip().str.lower()
    coded = sheet[sheet["coder1"].isin(CLASSES) & sheet["coder2"].isin(CLASSES)].copy()
    if coded.empty:
        raise SystemExit("No rows with both coder labels; fill coder1 and coder2 first.")
    coded["final"] = np.where(coded["adjudicated"].isin(CLASSES), coded["adjudicated"],
                              np.where(coded["coder1"] == coded["coder2"], coded["coder1"], ""))
    unresolved = int((coded["final"] == "").sum())

    s = pd.read_csv(SENTENCES, low_memory=False)
    s = s[s["Year"].between(1944, 2025)].reset_index().rename(columns={"index": "sentence_id"})
    coded = coded.merge(s[["sentence_id", "Pred_Label"]], on="sentence_id", how="left")
    final = coded[coded["final"] != ""]

    rows = {
        "Sentences double-coded": len(coded),
        "Raw agreement": (coded["coder1"] == coded["coder2"]).mean(),
        "Cohen's kappa": cohen_kappa(coded["coder1"], coded["coder2"]),
        "Unresolved disagreements": unresolved,
        "Model agreement with final label": (final["Pred_Label"] == final["final"]).mean(),
        "Model macro-F1 against final label": macro_f1(final["final"], final["Pred_Label"]),
        "Model kappa against final label": cohen_kappa(final["final"], final["Pred_Label"]),
    }
    for c in CLASSES:
        both = coded[(coded["coder1"] == c) | (coded["coder2"] == c)]
        rows[f"Specific agreement, {c}"] = (2 * ((coded["coder1"] == c) & (coded["coder2"] == c)).sum()
                                            / ((coded["coder1"] == c).sum() + (coded["coder2"] == c).sum())) if len(both) else np.nan
    out = pd.DataFrame({"Statistic": list(rows), "Value": list(rows.values())})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    kappa = rows["Cohen's kappa"]
    pct = lambda x: f"{100 * x:.1f}"
    macros = {
        "IRRN": f"{len(coded):,}",
        "IRRAgree": pct(rows["Raw agreement"]),
        "IRRKappa": f"{kappa:.2f}",
        "IRRModelAgree": pct(rows["Model agreement with final label"]),
        "IRRModelFone": f"{rows['Model macro-F1 against final label']:.3f}",
        "IRRSpecR": pct(rows["Specific agreement, restrictive"]),
        "IRRSpecN": pct(rows["Specific agreement, neutral"]),
        "IRRSpecS": pct(rows["Specific agreement, supportive"]),
    }
    OUT_TEX.write_text("% Generated by compute_interrater_reliability.py -- do not edit by hand.\n"
                       + "".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(out.to_string(index=False))
    print(f"Wrote {OUT_CSV} and {OUT_TEX}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, default=400)
    d.add_argument("--seed", type=int, default=20261019)
    sub.add_parser("score")
    args = parser.parse_args()
    draw(args.n, args.seed) if args.cmd == "draw" else score()


if __name__ == "__main__":
    main()
