"""Check that the saved classifier reproduces the canonical sentence scores.

Rescores a random sample of corpus sentences with the preserved model and
temperature (pipeline/02_score_sentences.py) and compares the calibrated
probabilities and labels with data/derived/sentence_scores_canonical.csv.
A full rescore of all 87,160 sentences is done by pipeline/run_all.py; this
sample check takes a few minutes on a CPU.

    python paper_outputs/code/verify_inference_reproducibility.py --n 2000
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PKG = Path(__file__).resolve().parents[2]
KEYS = ["Institution", "Year", "File", "Sentence_Number", "Sentence_Cleaned"]
PROBS = ["P_restrictive_calib", "P_neutral_calib", "P_supportive_calib"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()

    tmp = PKG / "data" / "intermediate"
    tmp.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(PKG / "data" / "raw" / "policy_sentences_cleaned_combined.csv", low_memory=False)
    sample = raw.sample(n=args.n, random_state=args.seed)
    sample_path, scored_path = tmp / "repro_sample.csv", tmp / "repro_sample_scored.csv"
    sample.to_csv(sample_path, index=False)

    model = PKG / "model" / "srn_cls_model"
    subprocess.run([sys.executable, str(PKG / "pipeline" / "02_score_sentences.py"), "--model-dir", str(model),
                    "--temperature-file", str(model / "temperature_scaling.json"), "--score-file", str(sample_path),
                    "--score-text-col", "Sentence_Cleaned", "--out-file", str(scored_path), "--device", "auto",
                    "--batch-size", "32"], check=True)

    scored = pd.read_csv(scored_path, low_memory=False)
    canon = pd.read_csv(PKG / "data" / "derived" / "sentence_scores_canonical.csv", low_memory=False)
    m = scored.merge(canon[KEYS + PROBS + ["Pred_Label"]].drop_duplicates(KEYS), on=KEYS, suffixes=("", "_canonical"))
    diffs = np.abs(m[PROBS].to_numpy() - m[[p + "_canonical" for p in PROBS]].to_numpy())
    out = pd.DataFrame({
        "Statistic": ["Sentences rescored", "Matched to canonical scores", "Max absolute probability difference",
                      "Mean absolute probability difference", "Label agreement"],
        "Value": [len(scored), len(m), float(diffs.max()), float(diffs.mean()),
                  float((m["Pred_Label"] == m["Pred_Label_canonical"]).mean())],
    })
    dest = PKG / "paper_outputs" / "tables" / "revision_analyses" / "inference_reproducibility.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
