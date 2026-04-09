from pathlib import Path

import pandas as pd


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    raw_file = package_root / "data" / "raw" / "policy_sentences_cleaned_combined.csv"
    intermediate_dir = package_root / "data" / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.read_csv(raw_file, low_memory=False)
    combined.to_csv(intermediate_dir / "policy_sentences_input_raw.csv", index=False)

    summary = {
        "raw_sentence_rows": int(len(combined)),
        "institutions": int(combined["Institution"].nunique()),
        "institution_years": int(combined[["Institution", "Year"]].drop_duplicates().shape[0]),
    }
    print(summary)


if __name__ == "__main__":
    main()
