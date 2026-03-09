import argparse
import re
from pathlib import Path

import pandas as pd


def clean_name(value) -> str:
    value = "" if pd.isna(value) else str(value)
    return re.sub(r"\s+", " ", value).strip()


def clean_year(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return pd.NA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-file", required=True)
    parser.add_argument("--panel-file", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--institution-col", default="Institution")
    parser.add_argument("--year-col", default="Year")
    args = parser.parse_args()

    policy_path = Path(args.policy_file)
    panel_path = Path(args.panel_file)
    out_path = Path(args.out_file)

    policy = pd.read_csv(policy_path, low_memory=False)
    panel = pd.read_csv(panel_path, low_memory=False)

    policy["Institution"] = policy["Institution"].apply(clean_name)
    policy["Year"] = policy["Year"].apply(clean_year)
    panel[args.institution_col] = panel[args.institution_col].apply(clean_name)
    panel[args.year_col] = panel[args.year_col].apply(clean_year)

    policy = policy.dropna(subset=["Institution", "Year"]).copy()
    panel = panel.dropna(subset=[args.institution_col, args.year_col]).copy()
    policy["Year"] = policy["Year"].astype(int)
    panel[args.year_col] = panel[args.year_col].astype(int)

    merged = panel.merge(
        policy,
        how="left",
        left_on=[args.institution_col, args.year_col],
        right_on=["Institution", "Year"],
        suffixes=("", "_policy"),
    )

    policy_cols = [col for col in policy.columns if col not in {"Institution", "Year"}]
    merged = merged.sort_values([args.institution_col, args.year_col]).reset_index(drop=True)
    merged[policy_cols] = merged.groupby(args.institution_col, sort=False)[policy_cols].ffill()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
