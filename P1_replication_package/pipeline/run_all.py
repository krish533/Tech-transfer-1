import subprocess
import sys
from pathlib import Path


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    pipeline_dir = package_root / "pipeline"
    paper_code_dir = package_root / "paper_outputs" / "code"
    intermediate_dir = package_root / "data" / "intermediate"
    derived_dir = package_root / "data" / "derived"
    model_dir = package_root / "model" / "srn_cls_model"
    temperature_file = model_dir / "temperature_scaling.json"
    primary_panel = derived_dir / "policy_level_indices_primary_1944_2025.csv"

    intermediate_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([sys.executable, str(pipeline_dir / "01_build_input_corpus.py")], check=True)
    subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "02_score_sentences.py"),
            "--model-dir",
            str(model_dir),
            "--temperature-file",
            str(temperature_file),
            "--score-file",
            str(intermediate_dir / "policy_sentences_input_raw.csv"),
            "--score-text-col",
            "Sentence_Cleaned",
            "--out-file",
            str(intermediate_dir / "policy_sentences_scored_raw_rerun.csv"),
            "--device",
            "auto",
            "--batch-size",
            "32",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "03_build_policy_level_indices.py"),
            "--rerun-file",
            str(intermediate_dir / "policy_sentences_scored_raw_rerun.csv"),
            "--out-dir",
            str(derived_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(pipeline_dir / "04_build_primary_analysis_panel.py"),
            "--input-file",
            str(derived_dir / "policy_level_indices_institution_year.csv"),
            "--output-file",
            str(primary_panel),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(paper_code_dir / "generate_paper_outputs.py"),
            "--indices-file",
            str(primary_panel),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(paper_code_dir / "generate_strengthened_results.py"),
            "--panel-file",
            str(primary_panel),
            "--sentences-file",
            str(derived_dir / "sentence_scores_canonical.csv"),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(paper_code_dir / "generate_final_manuscript_figures.py"),
            "--panel-file",
            str(primary_panel),
            "--outdir",
            str(package_root / "paper_outputs" / "figures"),
        ],
        check=True,
    )

    print("Replication package run completed for the 1944--2025 manuscript sample, including strengthened robustness outputs and final manuscript figures.")


if __name__ == "__main__":
    main()
