from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PKG: Path
RED = "#cf4a5d"
BLUE = "#2f80bd"
TEAL = "#218f83"
GOLD = "#d9b66f"
INK = "#263242"
GRID = "#ccd6e1"
PDF_METADATA = {
    "Creator": "PhysioCAT release renderer",
    "CreationDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.4,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
    }
)


def panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    ax.text(-0.18, 1.055, letter, transform=ax.transAxes, fontsize=12.0, fontweight="bold", color=INK)
    ax.set_title(title, loc="left", pad=6, fontweight="bold", color="black")


def polish(ax: plt.Axes, *, xgrid: bool = False, ygrid: bool = False) -> None:
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.75)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.75)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(color=INK, width=0.75, length=3)
    ax.set_axisbelow(True)


def metric(frame: pd.DataFrame, model: str, outcome: str) -> float:
    row = frame[(frame.model == model) & (frame.outcome == outcome)]
    if len(row) != 1:
        raise ValueError(f"Expected one metric row for {model}/{outcome}; found {len(row)}")
    return float(row.mae.iloc[0])


def paired_mae_panel(ax: plt.Axes, mechanism: pd.DataFrame, models: list[str], labels: list[str]) -> None:
    sbp = np.asarray([metric(mechanism, model, "SBP") for model in models])
    dbp = np.asarray([metric(mechanism, model, "DBP") for model in models])
    y = np.arange(len(models))[::-1]
    for row, left, right in zip(y, dbp, sbp, strict=True):
        ax.hlines(row, left, right, color=GRID, lw=2.2, zorder=1)
    ax.scatter(sbp, y, marker="o", s=34, color=RED, edgecolor="white", linewidth=0.4, zorder=3, label="SBP")
    ax.scatter(dbp, y, marker="s", s=32, color=BLUE, edgecolor="white", linewidth=0.4, zorder=3, label="DBP")
    ax.axvline(metric(mechanism, "physiocat", "SBP"), color=RED, ls=(0, (4, 2)), lw=0.8, alpha=0.65)
    ax.axvline(metric(mechanism, "physiocat", "DBP"), color=BLUE, ls=(0, (4, 2)), lw=0.8, alpha=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    lower = min(float(sbp.min()), float(dbp.min())) - 0.16
    upper = max(float(sbp.max()), float(dbp.max())) + 0.22
    ax.set_xlim(lower, upper)
    ax.set_xlabel("MAE (mmHg)")
    polish(ax, xgrid=True)


def render(output: Path) -> None:
    main = pd.read_csv(PKG / "artifacts/metrics/main/main_window_metrics.csv")
    random_split = pd.read_csv(PKG / "artifacts/metrics/protocol/random_split_window_metrics.csv")
    mechanism = pd.read_csv(PKG / "artifacts/metrics/mechanism/mechanism_window_metrics.csv")
    negative = pd.read_csv(PKG / "artifacts/metrics/secondary/negative_controls.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.55, 5.95))
    plt.subplots_adjust(left=0.115, right=0.985, top=0.955, bottom=0.105, wspace=0.36, hspace=0.52)

    ax = axes[0, 0]
    outcomes = ["SBP", "DBP"]
    exact = np.asarray([metric(main, "physiocat", outcome) for outcome in outcomes])
    overlap = np.asarray([metric(random_split, "random_split_physiocat", outcome) for outcome in outcomes])
    x = np.arange(2)
    width = 0.34
    colors = [RED, BLUE]
    ax.bar(x - width / 2, exact, width, color=colors, alpha=0.98, label="Subject-disjoint")
    ax.bar(x + width / 2, overlap, width, facecolor="white", edgecolor=colors, hatch="///", linewidth=1.0, label="Random split")
    for xi, high, low, color in zip(x, exact, overlap, colors, strict=True):
        delta = low - high
        ax.annotate("", xy=(xi + 0.17, low + 0.10), xytext=(xi, high - 0.06), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": color})
        ax.text(xi, high + 0.28, f"{delta:.2f}", color=color, ha="center", va="bottom", fontweight="bold", fontsize=8.3)
    ax.text(0.49, 0.64, "apparent error deflation", ha="center", fontsize=7.3, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel("MAE (mmHg)")
    ax.set_ylim(0, 4.85)
    ax.legend(frameon=False, loc="upper right", fontsize=6.6, handlelength=1.2)
    polish(ax, ygrid=True)
    panel_title(ax, "A", "Protocol sensitivity")

    ax = axes[0, 1]
    paired_mae_panel(
        ax,
        mechanism,
        ["physiocat", "ppg_leading_mirror", "direction_agnostic_local", "shifted_offsets_9_12", "attention_edge_ablation", "matched_no_delay"],
        ["PhysioCAT", "PPG-leading\nmirror", "Local\nagnostic", "Shifted\nband", "Degree-preserving\nrewiring", "No-delay"],
    )
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
    panel_title(ax, "B", "Structured mask controls")

    ax = axes[1, 0]
    paired_mae_panel(
        ax,
        mechanism,
        ["physiocat", "ppg_leading_mirror", "unidirectional_delay", "matched_no_delay", "gated_fusion", "early_concat"],
        ["PhysioCAT", "PPG-leading\nmirror", "Unidirectional", "No-delay", "Gated", "Early\nconcat"],
    )
    panel_title(ax, "C", "Fusion-direction controls")

    ax = axes[1, 1]
    perturbations = [
        "Token-aligned circular PPG shift",
        "Cross-subject ECG-PPG pairing",
        "PPG time reversal",
        "Implausible delay mask",
    ]
    short = ["Token-aligned\nshift", "Cross-\nsubject", "PPG\nreversal", "Implausible\nmask"]
    phys = []
    no_delay = []
    for perturbation in perturbations:
        phys.append(float(negative[(negative.model == "PhysioCAT") & (negative.perturbation == perturbation)].sbp_degradation_mmhg.iloc[0]))
        no_delay.append(float(negative[(negative.model == "No-delay-band cross-attention") & (negative.perturbation == perturbation)].sbp_degradation_mmhg.iloc[0]))
    y = np.arange(len(perturbations))[::-1]
    height = 0.32
    ax.barh(y + height / 2, phys, height, color=TEAL, label="PhysioCAT")
    ax.barh(y - height / 2, no_delay, height, color=GOLD, label="No-delay")
    ax.set_yticks(y)
    ax.set_yticklabels(short)
    ax.set_xlim(0, max(4.25, max(phys) + 0.20))
    ax.set_xlabel("SBP MAE increase (mmHg)")
    ax.legend(frameon=False, loc="lower right", ncol=2, fontsize=6.7, handlelength=1.3, columnspacing=0.8)
    polish(ax, xgrid=True)
    panel_title(ax, "D", "Timing perturbations")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=300, metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Figure 5 from the released protocol and mechanism tables")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global PKG
    PKG = args.package_root.resolve()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
