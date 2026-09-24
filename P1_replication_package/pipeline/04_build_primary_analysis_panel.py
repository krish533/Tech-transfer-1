from pathlib import Path
import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the 1944--2025 primary PCSI analysis panel from the full canonical panel.")
    parser.add_argument("--input-file")
    parser.add_argument("--output-file")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input_file) if args.input_file else package_root / "data" / "derived" / "policy_level_indices_institution_year.csv"
    output_path = Path(args.output_file) if args.output_file else package_root / "data" / "derived" / "policy_level_indices_primary_1944_2025.csv"

    df = pd.read_csv(input_path, low_memory=False)
    primary = df[df["Year"].between(1944, 2025)].copy()

    if len(primary) != 4277 or primary["Institution"].nunique() != 150:
        raise ValueError(
            f"Expected 4,277 rows and 150 institutions in the primary panel; "
            f"found {len(primary)} rows and {primary['Institution'].nunique()} institutions."
        )
    observed = primary[primary["Is_Carried_Forward"] == 0]
    if len(observed) != 480:
        raise ValueError(f"Expected 480 directly observed policy records; found {len(observed)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary.to_csv(output_path, index=False)
    print(f"Wrote primary 1944--2025 analysis panel: {output_path}")


if __name__ == "__main__":
    main()
