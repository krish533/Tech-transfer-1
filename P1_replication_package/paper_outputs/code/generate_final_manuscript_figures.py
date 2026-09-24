from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the two figure files referenced by the final 1944--2025 manuscript."
    )
    parser.add_argument("--panel-file")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    package_root = Path(__file__).resolve().parents[2]
    panel_path = (
        Path(args.panel_file)
        if args.panel_file
        else package_root / "data" / "derived" / "policy_level_indices_primary_1944_2025.csv"
    )
    outdir = Path(args.outdir) if args.outdir else package_root / "paper_outputs" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path, low_memory=False)
    panel = panel[panel["Year"].between(1944, 2025)].copy()
    if len(panel) != 4277 or panel["Institution"].nunique() != 150:
        raise ValueError(
            f"Expected final manuscript panel with 4,277 rows and 150 institutions; "
            f"found {len(panel)} rows and {panel['Institution'].nunique()} institutions."
        )

    # Figure 1: PCSI distribution in the primary policy-in-force panel.
    x = panel["Mean_Tone_Score"].dropna()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(x, bins=30, alpha=0.75, edgecolor="white")
    ax.axvline(0.50, linestyle="--", linewidth=1.2, label="Scale midpoint (0.50)")
    ax.axvline(x.mean(), linestyle="-", linewidth=1.2, label=f"Mean ({x.mean():.3f})")
    ax.axvline(x.median(), linestyle=":", linewidth=1.4, label=f"Median ({x.median():.3f})")
    ax.set_xlabel("Policy Communication Stance Index (PCSI)")
    ax.set_ylabel("Policy-in-force institution-years")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_pci_histogram.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig1_pci_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: annual mean PCSI beginning in 1978, the first year with >=5 universities.
    annual = (
        panel.groupby("Year")["Mean_Tone_Score"]
        .agg(mean="mean", sd="std", n="size")
        .reset_index()
    )
    annual = annual[(annual["Year"] >= 1978) & (annual["n"] >= 5)].copy()
    annual["se"] = annual["sd"] / np.sqrt(annual["n"])
    annual["ci_low"] = annual["mean"] - 1.96 * annual["se"]
    annual["ci_high"] = annual["mean"] + 1.96 * annual["se"]
    annual["rolling3"] = annual["mean"].rolling(3, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    denom = max(1, annual["n"].max() - annual["n"].min())
    sizes = 16 + 44 * (annual["n"] - annual["n"].min()) / denom
    ax.fill_between(
        annual["Year"].to_numpy(),
        annual["ci_low"].to_numpy(),
        annual["ci_high"].to_numpy(),
        alpha=0.18,
    )
    ax.scatter(annual["Year"], annual["mean"], s=sizes, alpha=0.65, label="Annual mean")
    ax.plot(
        annual["Year"],
        annual["rolling3"],
        linestyle="--",
        linewidth=1.6,
        label="3-year rolling mean",
    )
    ax.axhline(0.50, linestyle=":", linewidth=1.2, label="Scale midpoint (0.50)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean PCSI")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_temporal_trend_pcsi.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig3_temporal_trend_pcsi.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Final manuscript figures written to {outdir}")


if __name__ == "__main__":
    main()
