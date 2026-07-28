from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PKG: Path
INK = "#172033"
GRID = "#D8E0EA"
SBP = "#C84E55"
DBP = "#2F7EBB"
PHYSIO = "#177E72"
BASE = "#C56A33"
GREY = "#9AA7B3"
PDF_METADATA = {
    "Creator": "PhysioCAT release renderer",
    "CreationDate": datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 28, 0, 0, tzinfo=timezone.utc),
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 6.9,
        "axes.titlesize": 7.8,
        "axes.labelsize": 7.1,
        "axes.linewidth": 0.72,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "legend.fontsize": 6.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 500,
        "figure.dpi": 180,
        "axes.unicode_minus": False,
    }
)


DISPLAY = {
    "pat_ridge": "PAT ridge",
    "random_forest": "Engineered RF",
    "cnn_bilstm": "CNN-BiLSTM",
    "bp_net": "BP-Net",
    "te_sagru": "TE-SAGRU",
    "mufubp_net": "MuFuBP-Net",
    "matched_no_delay": "No-delay X-attn",
    "physiocat": "PhysioCAT",
}


def clean(ax: plt.Axes, grid: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid, color=GRID, lw=0.5, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.4, width=0.7, color=INK)
    for spine in ax.spines.values():
        spine.set_color(INK)


def panel(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.07) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=10.2, fontweight="bold", color=INK)


def metric(metrics: pd.DataFrame, model: str, outcome: str) -> pd.Series:
    row = metrics[(metrics.model == model) & (metrics.outcome == outcome)]
    if len(row) != 1:
        raise KeyError(f"Expected one metric row for {model}/{outcome}")
    return row.iloc[0]


def density_agreement(ax: plt.Axes, ref: np.ndarray, pred: np.ndarray, lim: tuple[float, float], color: str, name: str, letter: str, cmap: str, row: pd.Series) -> None:
    hb = ax.hexbin(ref, pred, gridsize=58, mincnt=1, bins="log", cmap=cmap, linewidths=0, rasterized=True)
    ax.plot(lim, lim, color=INK, lw=0.85)
    fit = np.polyfit(ref, pred, 1)
    ax.plot(lim, np.polyval(fit, lim), color=color, lw=1.05, ls=(0, (4, 2)))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"Reference {name} (mmHg)")
    ax.set_ylabel(f"Estimated {name} (mmHg)")
    ax.set_title(f"{name} agreement", loc="left", fontweight="bold")
    clean(ax, "both")
    ax.text(
        0.04,
        0.94,
        f"MAE {float(row.mae):.2f}  ME {float(row.me):+.2f}\nr = {float(row.pearson_r):.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.1,
        color=INK,
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.22", alpha=0.88),
    )
    colorbar = plt.colorbar(hb, ax=ax, fraction=0.041, pad=0.012)
    colorbar.ax.tick_params(labelsize=5.1, length=2)
    panel(ax, letter)


def bland_altman(ax: plt.Axes, ref: np.ndarray, pred: np.ndarray, name: str, color: str, letter: str, cmap: str, row: pd.Series) -> None:
    mean = (ref + pred) / 2.0
    diff = pred - ref
    bias = float(row.me)
    sde = float(row.sde)
    loa_low = bias - 1.96 * sde
    loa_high = bias + 1.96 * sde
    hb = ax.hexbin(mean, diff, gridsize=58, mincnt=1, bins="log", cmap=cmap, linewidths=0, rasterized=True)
    ax.axhline(bias, color=INK, lw=1.05, zorder=4)
    ax.axhline(loa_high, color=color, lw=1.15, ls=(0, (4, 2)), zorder=4)
    ax.axhline(loa_low, color=color, lw=1.15, ls=(0, (4, 2)), zorder=4)
    bins = np.linspace(np.quantile(mean, 0.01), np.quantile(mean, 0.99), 28)
    centers = 0.5 * (bins[:-1] + bins[1:])
    medians = [np.median(diff[(mean >= low) & (mean < high)]) for low, high in zip(bins[:-1], bins[1:], strict=True)]
    ax.plot(centers, medians, color="#B88400", lw=1.35, zorder=5)
    xlim = (60, 225) if name == "SBP" else (35, 135)
    tail = float(np.quantile(np.abs(diff - bias), 0.999))
    ymax = max(35.0 if name == "SBP" else 24.0, np.ceil((tail + 2.0) / 2.0) * 2.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(f"Mean reference/estimate {name} (mmHg)")
    ax.set_ylabel("Prediction - reference (mmHg)")
    ax.set_title(f"{name} Bland--Altman", loc="left", fontweight="bold")
    clean(ax, "both")
    ax.text(
        0.04,
        0.94,
        f"bias {bias:+.2f}\n95% LoA {loa_low:+.1f} to {loa_high:+.1f}",
        transform=ax.transAxes,
        va="top",
        fontsize=5.9,
        color=INK,
        bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.20", alpha=0.88),
    )
    colorbar = plt.colorbar(hb, ax=ax, fraction=0.041, pad=0.012)
    colorbar.ax.tick_params(labelsize=5.1, length=2)
    x0, x1 = ax.get_xlim()
    ax.text(x1 - 0.02 * (x1 - x0), loa_high, "upper LoA", ha="right", va="bottom", fontsize=5.1, color=color)
    ax.text(x1 - 0.02 * (x1 - x0), loa_low, "lower LoA", ha="right", va="top", fontsize=5.1, color=color)
    panel(ax, letter)


def render(output: Path) -> None:
    metrics = pd.read_csv(PKG / "artifacts/metrics/main/main_window_metrics.csv")
    predictions = pd.read_csv(PKG / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz", compression="gzip", low_memory=False)

    fig = plt.figure(figsize=(7.22, 6.35))
    grid = fig.add_gridspec(3, 2, left=0.125, right=0.985, top=0.965, bottom=0.075, wspace=0.31, hspace=0.55)

    ax = fig.add_subplot(grid[0, 0])
    order = ["pat_ridge", "random_forest", "cnn_bilstm", "bp_net", "te_sagru", "mufubp_net", "matched_no_delay", "physiocat"]
    y = np.arange(len(order))[::-1]
    for index, model in enumerate(order):
        sbp_mae = float(metric(metrics, model, "SBP").mae)
        dbp_mae = float(metric(metrics, model, "DBP").mae)
        if model == "physiocat":
            ax.axhspan(y[index] - 0.39, y[index] + 0.39, color="#EEF7F4", zorder=0)
        ax.plot([dbp_mae, sbp_mae], [y[index], y[index]], color="#C6D0DA", lw=1.15, zorder=1)
        ax.plot(sbp_mae, y[index] + 0.10, marker="o", ms=4.6, color=SBP, mec="white", mew=0.45, zorder=3)
        ax.plot(dbp_mae, y[index] - 0.10, marker="s", ms=4.2, color=DBP, mec="white", mew=0.45, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([DISPLAY[name] for name in order])
    ax.set_xlim(2.55, 10.0)
    ax.set_xlabel("MAE (mmHg), lower is better")
    ax.set_title("Five-fold subject-grouped model ranking", loc="left", fontweight="bold")
    clean(ax, "x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SBP, markeredgecolor="white", label="SBP"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor=DBP, markeredgecolor="white", label="DBP"),
        ],
        frameon=False,
        loc="lower right",
        borderpad=0,
    )
    panel(ax, "A", x=-0.20)

    ax = fig.add_subplot(grid[0, 1])
    models = ["physiocat", "mufubp_net", "matched_no_delay"]
    distributions = []
    for model in models:
        absolute = (predictions[f"pred_{model}_sbp"] - predictions.sbp).abs()
        distributions.append(absolute.groupby(predictions.subject_id, sort=False).mean().to_numpy())
    colors = [PHYSIO, BASE, GREY]
    violins = ax.violinplot(distributions, positions=[0, 1, 2], widths=0.78, showmeans=False, showextrema=False)
    for body, color in zip(violins["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_alpha(0.33)
        body.set_edgecolor("none")
    boxes = ax.boxplot(distributions, positions=[0, 1, 2], widths=0.22, patch_artist=True, showfliers=False)
    for box, color in zip(boxes["boxes"], colors, strict=True):
        box.set(facecolor="white", edgecolor=color, lw=1.0)
    for key in ("whiskers", "caps"):
        for item in boxes[key]:
            item.set(color=INK, lw=0.75)
    for median in boxes["medians"]:
        median.set(color=INK, lw=1.1)
    rng = np.random.default_rng(6171)
    for position, values in enumerate(distributions):
        points = rng.choice(values, min(120, len(values)), replace=False)
        ax.scatter(np.full_like(points, position) + rng.normal(0, 0.045, len(points)), points, s=4.2, color=INK, alpha=0.13, rasterized=True)
    ax.axhline(5.0, color=GRID, lw=0.8, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["PhysioCAT", "MuFuBP-Net", "No-delay\nX-attn"])
    ax.set_ylabel("Subject-level SBP MAE (mmHg)")
    ax.set_title("Subject-level dispersion", loc="left", fontweight="bold")
    ax.set_ylim(1.5, max(12.2, float(np.quantile(np.concatenate(distributions), 0.992))))
    clean(ax, "y")
    ax.text(0.03, 0.94, f"median / 90th percentile\n{np.median(distributions[0]):.2f} / {np.quantile(distributions[0], 0.90):.2f} mmHg", transform=ax.transAxes, va="top", fontsize=5.9, color=PHYSIO)
    panel(ax, "B", x=-0.18)

    sbp_ref = predictions.sbp.to_numpy(float)
    dbp_ref = predictions.dbp.to_numpy(float)
    sbp_pred = predictions.pred_physiocat_sbp.to_numpy(float)
    dbp_pred = predictions.pred_physiocat_dbp.to_numpy(float)
    sbp_row = metric(metrics, "physiocat", "SBP")
    dbp_row = metric(metrics, "physiocat", "DBP")
    density_agreement(fig.add_subplot(grid[1, 0]), sbp_ref, sbp_pred, (60, 225), SBP, "SBP", "C", "magma", sbp_row)
    density_agreement(fig.add_subplot(grid[1, 1]), dbp_ref, dbp_pred, (35, 135), DBP, "DBP", "D", "Blues", dbp_row)
    bland_altman(fig.add_subplot(grid[2, 0]), sbp_ref, sbp_pred, "SBP", SBP, "E", "viridis", sbp_row)
    bland_altman(fig.add_subplot(grid[2, 1]), dbp_ref, dbp_pred, "DBP", DBP, "F", "YlGnBu", dbp_row)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.012, metadata=PDF_METADATA)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the data-driven primary results figure")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global PKG
    PKG = args.package_root.resolve()
    render(args.output.resolve())


if __name__ == "__main__":
    main()
