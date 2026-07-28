from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse


PKG: Path
OUT: Path

TEAL = "#138a7e"
ORANGE = "#d7743f"
PURPLE = "#7360b6"
BLUE = "#2f73b7"
INK = "#111827"
GRID = "#d7e1ef"
SPINE = "#263242"
PDF_METADATA = {
    "Creator": "PhysioCAT release renderer",
    "CreationDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
}


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.7,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "standard",
    }
)


def panel_title(ax: plt.Axes, letter: str, title: str, *, title_x: float = 0.035) -> None:
    ax.text(
        -0.11,
        1.075,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.4,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )
    ax.text(
        title_x,
        1.085,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        fontweight="bold",
        color=INK,
        clip_on=False,
    )


def polish(ax: plt.Axes, *, xgrid: bool = False) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.75)
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.75)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.85)
    ax.tick_params(width=0.75, length=3.0, color=SPINE)


def read_primary_metrics() -> pd.DataFrame:
    return pd.read_csv(PKG / "artifacts" / "metrics" / "main" / "main_window_metrics.csv")


def read_external_metrics() -> pd.DataFrame:
    return pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "external_zero_shot_summary.csv")


def read_source_validation_metrics() -> pd.DataFrame:
    return pd.read_csv(PKG / "artifacts" / "metrics" / "external" / "source_model_internal_validation.csv")


def plot_zero_shot(ax: plt.Axes) -> None:
    ext = read_external_metrics()
    source_validation = read_source_validation_metrics()
    cohorts = ["Vital source\nvalidation", "PulseDB-\nMIMIC", "MIMIC-BP"]
    x = np.arange(len(cohorts))

    def value(model: str, dataset: str, endpoint: str) -> float:
        if dataset == "PulseDB-Vital":
            slug = {"PhysioCAT": "physiocat", "MuFuBP-Net": "mufubp_net"}[model]
            row = source_validation.loc[
                (source_validation.source_model == slug)
                & (source_validation.outcome == endpoint.upper())
            ].iloc[0]
            return float(row.mae)
        else:
            row = ext.loc[(ext["dataset"] == dataset) & (ext["model"] == model)].iloc[0]
            return float(row[f"{endpoint}_mae"])

    series = {
        ("PhysioCAT", "sbp"): [value("PhysioCAT", d, "sbp") for d in ["PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"]],
        ("PhysioCAT", "dbp"): [value("PhysioCAT", d, "dbp") for d in ["PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"]],
        ("MuFuBP-Net", "sbp"): [value("MuFuBP-Net", d, "sbp") for d in ["PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"]],
        ("MuFuBP-Net", "dbp"): [value("MuFuBP-Net", d, "dbp") for d in ["PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"]],
    }

    ax.plot(x, series[("MuFuBP-Net", "sbp")], color=ORANGE, marker="o", linewidth=1.8, markersize=4.8)
    ax.plot(x, series[("MuFuBP-Net", "dbp")], color=ORANGE, marker="s", linestyle="--", linewidth=1.8, markersize=4.6)
    ax.plot(x, series[("PhysioCAT", "sbp")], color=TEAL, marker="o", linewidth=1.8, markersize=4.8)
    ax.plot(x, series[("PhysioCAT", "dbp")], color=TEAL, marker="s", linestyle="--", linewidth=1.8, markersize=4.6)
    for i in (1, 2):
        ax.annotate(
            "",
            xy=(x[i], series[("PhysioCAT", "sbp")][i] + 0.04),
            xytext=(x[i], series[("MuFuBP-Net", "sbp")][i] - 0.04),
            arrowprops=dict(arrowstyle="-|>", color="#7a8797", lw=0.85, shrinkA=0, shrinkB=0),
        )
    ax.text(2.04, series[("MuFuBP-Net", "sbp")][-1], "MuFuBP-Net SBP", color=ORANGE, va="center", fontsize=6.0)
    ax.text(2.04, series[("PhysioCAT", "sbp")][-1] - 0.04, "PhysioCAT SBP", color=TEAL, va="center", fontsize=6.0)
    ax.text(2.04, series[("MuFuBP-Net", "dbp")][-1] + 0.01, "MuFuBP-Net DBP", color=ORANGE, va="center", fontsize=6.0)
    ax.text(2.04, series[("PhysioCAT", "dbp")][-1] - 0.02, "PhysioCAT DBP", color=TEAL, va="center", fontsize=6.0)
    ax.set_xlim(-0.1, 2.48)
    all_values = np.concatenate([np.asarray(values, dtype=float) for values in series.values()])
    ax.set_ylim(2.9, float(all_values.max()) + 0.30)
    ax.set_xticks(x, cohorts)
    ax.set_ylabel("MAE (mmHg)")
    panel_title(ax, "A", "Frozen source-model transfer")
    polish(ax, xgrid=True)


def plot_few_shot(ax: plt.Axes) -> None:
    df = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "few_shot_calibration.csv")
    x = df["calibration_windows"].to_numpy(float)
    lines = [
        ("physiocat_sbp", TEAL, "o", "-", "PhysioCAT SBP"),
        ("baseline_sbp", ORANGE, "o", "-", "Baseline SBP"),
        ("physiocat_dbp", TEAL, "s", "--", "PhysioCAT DBP"),
        ("baseline_dbp", ORANGE, "s", "--", "Baseline DBP"),
    ]
    for prefix, color, marker, linestyle, label in lines:
        y = df[f"{prefix}_mae"].to_numpy(float)
        lo = df[f"{prefix}_resampling_low"].to_numpy(float)
        hi = df[f"{prefix}_resampling_high"].to_numpy(float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, linewidth=1.8, markersize=4.3, label=label)
    ax.set_xlim(-0.35, 10.55)
    ax.set_ylim(2.95, 5.75)
    ax.set_xticks([0, 1, 2, 5, 10])
    ax.set_xlabel("Calibration windows per subject")
    ax.set_ylabel("MIMIC-BP MAE (mmHg)")
    ax.legend(loc="upper right", fontsize=5.8, handlelength=1.6, borderaxespad=0.25, labelspacing=0.25)
    ax.text(0.02, 0.03, "shading: 2.5--97.5% across calibration-window draws", transform=ax.transAxes, fontsize=5.1, color="#596579")
    panel_title(ax, "B", "Few-shot residual-offset calibration")
    polish(ax, xgrid=True)


def plot_retention_sensitivity(ax: plt.Axes) -> None:
    df = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "retention_sensitivity.csv").iloc[:4].copy()
    labels = ["No SQI\nthreshold", "Relaxed", "Default", "Stricter"]
    x = np.arange(len(df))
    ax.plot(x, df.sbp_mae, color=TEAL, marker="o", linewidth=1.9, markersize=4.8, label="SBP")
    ax.plot(x, df.dbp_mae, color=BLUE, marker="s", linestyle="--", linewidth=1.9, markersize=4.5, label="DBP")
    for xi, row in zip(x, df.itertuples(index=False), strict=True):
        ax.text(xi, float(row.sbp_mae) + 0.08, f"{int(row.windows):,}", ha="center", va="bottom", fontsize=5.4, color="#596579")
    ax.set_xlim(-0.25, len(df) - 0.75)
    ax.set_ylim(2.9, max(5.0, float(df.sbp_mae.max()) + 0.35))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Frozen-model MAE (mmHg)")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.5, borderaxespad=0.2)
    ax.text(0.02, 0.03, "labels show evaluated windows", transform=ax.transAxes, fontsize=5.2, color="#596579")
    panel_title(ax, "C", "Input-quality sensitivity")
    polish(ax, xgrid=True)


def plot_residuals(ax: plt.Axes) -> None:
    path = PKG / "artifacts" / "predictions" / "pulsedb_vital_subject_grouped_oof_predictions.csv.gz"
    with gzip.open(path, "rt") as handle:
        df = pd.read_csv(handle, usecols=["sbp", "pred_physiocat_sbp"])
    x = df["sbp"].to_numpy(float)
    y = df["pred_physiocat_sbp"].to_numpy(float) - x
    keep = (x >= 70) & (x <= 205) & (y >= -32) & (y <= 32)
    # Use a compact continuous-density display while keeping every bin tied
    # to the released prediction rows.
    density = ax.hexbin(
        x[keep],
        y[keep],
        gridsize=62,
        extent=(70, 205, -32, 32),
        mincnt=1,
        cmap="RdYlBu_r",
        norm=LogNorm(),
        linewidths=0,
        alpha=0.95,
    )
    density.set_rasterized(True)
    ax.axhline(0, color=SPINE, linewidth=0.75, linestyle=(0, (3, 2)))
    coef = np.polyfit(x[keep], y[keep], deg=1)
    xx = np.linspace(70, 205, 120)
    ax.plot(xx, coef[0] * xx + coef[1], color=ORANGE, linewidth=1.05)
    ax.set_xlim(70, 205)
    ax.set_ylim(-32, 32)
    ax.set_xlabel("Reference SBP (mmHg)")
    ax.set_ylabel("Residual error (mmHg)")
    panel_title(ax, "D", "Residual structure under BP extremes")
    polish(ax, xgrid=True)
    cb = plt.colorbar(density, ax=ax, fraction=0.046, pad=0.018)
    cb.ax.tick_params(labelsize=5.8, length=2.2, width=0.55)


def plot_source_shift(ax: plt.Axes) -> None:
    source = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "source_shift_projection.csv")
    colors = {"PulseDB-Vital": TEAL, "PulseDB-MIMIC": BLUE, "MIMIC-BP": ORANGE}
    labels = {"PulseDB-Vital": "Vital", "PulseDB-MIMIC": "PulseDB-MIMIC", "MIMIC-BP": "MIMIC-BP"}
    for cohort in ("PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"):
        subset = source[source.cohort == cohort]
        pts = subset[["projection_1", "projection_2"]].to_numpy(float)
        mean = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        color = colors[cohort]
        label = labels[cohort]
        ax.scatter(pts[:, 0], pts[:, 1], s=6.0, color=color, alpha=0.28, edgecolors="none", label=label)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse(mean, width=2.0 * np.sqrt(vals[0]), height=2.0 * np.sqrt(vals[1]), angle=angle, fill=False, lw=1.15, ec=color)
        ax.add_patch(ell)
        ax.scatter([mean[0]], [mean[1]], s=18, color=color, edgecolor="white", linewidth=0.55, zorder=4)
    all_points = source[["projection_1", "projection_2"]].to_numpy(float)
    x_margin = 0.08 * np.ptp(all_points[:, 0])
    y_margin = 0.08 * np.ptp(all_points[:, 1])
    ax.set_xlim(all_points[:, 0].min() - x_margin, all_points[:, 0].max() + x_margin)
    ax.set_ylim(all_points[:, 1].min() - y_margin, all_points[:, 1].max() + y_margin)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#d6dee9")
        ax.spines[side].set_linewidth(0.8)
    ax.legend(loc="lower left", fontsize=6.2, markerscale=1.05, borderaxespad=0.55)
    panel_title(ax, "E", "Input-summary source-shift map")


def plot_mimic_bridge(ax: plt.Axes) -> None:
    df = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "mimic_bp_protocol_audit.csv")
    names = ["PhysioCAT", "Matched no-delay", "MuFuBP-Net"]
    y = np.arange(len(names))[::-1]
    lookup = df.set_index("model")
    sbp = np.array([lookup.loc[n, "sbp_mae"] for n in names], dtype=float)
    dbp = np.array([lookup.loc[n, "dbp_mae"] for n in names], dtype=float)
    for yi, s, d in zip(y, sbp, dbp):
        ax.plot([d, s], [yi, yi], color="#d5dce6", linewidth=4.0, solid_capstyle="round", zorder=1)
    ax.scatter(sbp, y, s=37, color=TEAL, zorder=3, label="SBP")
    ax.scatter(dbp, y, s=31, color=BLUE, marker="s", zorder=3, label="DBP")
    for yi, s, d in zip(y, sbp, dbp):
        ax.text(s + 0.03, yi, f"{s:.2f}", va="center", fontsize=6.8, color=INK)
        ax.text(d + 0.03, yi - 0.13, f"{d:.2f}", va="center", fontsize=6.8, color=INK)
    ax.set_yticks(y, names)
    lower = min(float(dbp.min()), float(sbp.min())) - 0.28
    upper = max(float(dbp.max()), float(sbp.max())) + 0.48
    ax.set_xlim(lower, upper)
    ax.set_ylim(-0.25, 2.55)
    ax.set_xlabel("MAE (mmHg)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        fontsize=6.2,
        handlelength=1.0,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    panel_title(ax, "F", "MIMIC-BP comparison")
    polish(ax, xgrid=True)


def main() -> None:
    global PKG, OUT
    parser = argparse.ArgumentParser(description="Render submitted Figure 6 from released sources")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    PKG = args.package_root
    OUT = args.output
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(6.93, 5.37),
        gridspec_kw={"wspace": 0.58, "hspace": 0.62},
        constrained_layout=False,
    )
    plot_zero_shot(axes[0, 0])
    plot_few_shot(axes[0, 1])
    plot_retention_sensitivity(axes[0, 2])
    plot_residuals(axes[1, 0])
    plot_source_shift(axes[1, 1])
    plot_mimic_bridge(axes[1, 2])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.085, top=0.915, wspace=0.58, hspace=0.62)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, metadata=PDF_METADATA)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
