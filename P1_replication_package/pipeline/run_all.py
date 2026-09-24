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

    subprocess.run([sys.executable, str(paper_code_dir / "generate_paper_outputs.py")], check=True)
    subprocess.run([sys.executable, str(paper_code_dir / "generate_strengthened_results.py")], check=True)

    print("Replication package run completed, including strengthened manuscript robustness outputs.")


if __name__ == "__main__":
    main()
