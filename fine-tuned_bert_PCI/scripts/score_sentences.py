import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_MAP = {"restrictive": 0, "neutral": 1, "supportive": 2}
ID2LABEL = {0: "restrictive", 1: "neutral", 2: "supportive"}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def learn_temperature(
    model,
    tokenizer,
    calibration_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    max_length: int,
    max_samples: int | None,
    device: str,
) -> float:
    df = calibration_df[[text_col, label_col]].dropna().copy()
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()
    df = df[df[label_col].isin(LABEL_MAP)]
    if max_samples is not None:
        df = df.iloc[:max_samples].copy()
    if df.empty:
        raise ValueError("No usable calibration rows after filtering labels.")

    labels = torch.tensor(df[label_col].map(LABEL_MAP).to_numpy(), dtype=torch.long, device=device)
    encoded = tokenizer(
        df[text_col].astype(str).tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    model.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(**encoded).logits

    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        scaled = logits / torch.clamp(temperature, min=1e-3)
        loss = loss_fn(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(temperature, min=1e-3).item())


def score_sentences(
    model,
    tokenizer,
    score_df: pd.DataFrame,
    text_col: str,
    max_length: int,
    batch_size: int,
    temperature: float,
    device: str,
) -> pd.DataFrame:
    texts = score_df[text_col].astype(str).tolist()
    out_frames = []

    model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits / temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            batch = score_df.iloc[start:start + batch_size].copy()
            batch["P_restrictive_calib"] = probs[:, 0]
            batch["P_neutral_calib"] = probs[:, 1]
            batch["P_supportive_calib"] = probs[:, 2]
            batch["SRN_score"] = batch["P_supportive_calib"] - batch["P_restrictive_calib"]
            batch["ToneScore_0_1"] = batch["P_supportive_calib"] + 0.5 * batch["P_neutral_calib"]
            batch["Pred_Label"] = np.argmax(probs, axis=1)
            batch["Pred_Label"] = batch["Pred_Label"].map(ID2LABEL)
            out_frames.append(batch)

    return pd.concat(out_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--calib-file", required=True)
    parser.add_argument("--calib-text-col", required=True)
    parser.add_argument("--calib-label-col", required=True)
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--score-text-col", required=True)
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--calib-max-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model_dir = Path(args.model_dir)
    calibration_path = Path(args.calib_file)
    score_path = Path(args.score_file)
    out_path = Path(args.out_file)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    calibration_df = read_table(calibration_path)
    score_df = read_table(score_path)

    temperature = learn_temperature(
        model=model,
        tokenizer=tokenizer,
        calibration_df=calibration_df,
        text_col=args.calib_text_col,
        label_col=args.calib_label_col,
        max_length=args.max_length,
        max_samples=args.calib_max_samples,
        device=device,
    )

    scored_df = score_sentences(
        model=model,
        tokenizer=tokenizer,
        score_df=score_df,
        text_col=args.score_text_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        temperature=temperature,
        device=device,
    )
    write_table(scored_df, out_path)
    print(f"Device: {device}")
    print(f"Temperature: {temperature:.4f}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
