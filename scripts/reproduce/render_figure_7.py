from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle


PKG: Path
PDF: Path
PNG: Path | None

BLUE = "#2F70C8"
TEAL = "#0B8F87"
RED = "#D84A52"
PURPLE = "#6F57B5"
INK = "#172033"
GRID = "#D8E1EC"
GREY = "#9CA8B3"
PDF_METADATA = {
    "Creator": "PhysioCAT release renderer",
    "CreationDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 20, 0, 0, tzinfo=timezone.utc),
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.11, 1.08, label, transform=ax.transAxes, fontsize=16, weight="bold", color=INK)


def style_ax(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.tick_params(labelsize=8)


def display_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    low, high = np.percentile(values, [2, 98])
    return np.clip((values - low) / max(high - low, 1e-8), -0.08, 1.08)


def draw_wave_panel(ax, kind: str) -> None:
    rows = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "representative_windows.csv")
    role = "easier" if kind == "easy" else "harder"
    row = rows[rows.role == role].iloc[0]
    asset = PKG / str(row.waveform_asset)
    if hashlib.sha256(asset.read_bytes()).hexdigest() != str(row.waveform_sha256):
        raise AssertionError("Representative waveform asset hash differs from its source row")
    archive = np.load(asset, allow_pickle=False)
    waveform_row = int(row.waveform_row)
    if (
        str(archive["role"][waveform_row]) != role
        or str(archive["window_id"][waveform_row]) != str(row.window_id)
        or str(archive["subject_id"][waveform_row]) != str(row.subject_id)
    ):
        raise AssertionError("Representative waveform arrays are not bound to the selected result row")
    ecg = display_scale(archive["ecg"][waveform_row])
    ppg = display_scale(archive["ppg"][waveform_row])
    sqi = archive["sqi_tokens"][waveform_row].astype(float)
    t = np.arange(len(ecg), dtype=float) / 250.0
    if kind == "hard":
        low_quality = np.min(sqi, axis=0) < 0.55
        padded = np.r_[False, low_quality, False].astype(int)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1)
        token_seconds = 8.0 / sqi.shape[-1]
        for start, stop in zip(starts, stops, strict=True):
            ax.axvspan(start * token_seconds, stop * token_seconds, color="#D9D9D9", alpha=0.8, lw=0)
        title = "Representative harder window"
        note = f"Hard case: low SQI + rhythm irregularity\nSBP/DBP MAE {row.sbp_mae:.1f} / {row.dbp_mae:.1f} mmHg"
        note_color = RED
    else:
        title = "Representative easier window"
        note = f"Easy case: clean regular coupling\nSBP/DBP MAE {row.sbp_mae:.1f} / {row.dbp_mae:.1f} mmHg"
        note_color = TEAL
    ax.plot(t, ecg + 1.08, color="black", lw=0.95)
    ax.plot(t, ppg, color=TEAL, lw=1.0)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.15, 2.25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#B8C6D6")
        spine.set_linewidth(0.8)
    ax.set_title(title, loc="left", fontsize=10.5, weight="bold", pad=5)
    ax.text(0.03, 0.78, note, transform=ax.transAxes, color=note_color, fontsize=8.6, weight="bold")


def subgroup_panel(ax) -> None:
    source = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "figure_7_subgroup_source.csv")
    order = [
        "Age 18-40",
        "Age 41-65",
        "Age 66+",
        "Male",
        "Female",
        "Lower-BP windows",
        "Elevated-BP windows",
        "Normal rhythm",
        "Irregular rhythm",
        "Lowest SQI",
    ]
    rows = source.set_index("display_label").loc[order].reset_index()
    values = rows[["sbp_mae", "dbp_mae"]].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("phys", ["#EAF5E6", "#69B5C0", "#174A8B"])
    vmin = math.floor((float(values.min()) - 0.08) * 10) / 10
    vmax = math.ceil((float(values.max()) + 0.08) * 10) / 10
    norm = Normalize(vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, facecolor=cmap(norm(values[i, j])), edgecolor="none"))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks([0, 1], ["SBP", "DBP"])
    ax.set_yticks(np.arange(len(rows)), rows["display_label"])
    ax.tick_params(length=0, labelsize=8.2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            color = "white" if values[i, j] > 4.65 else INK
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8.5, weight="bold", color=color)
    ax.set_title("Descriptive subgroup error map", loc="left", fontsize=11, weight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("MAE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    panel_label(ax, "A")


def footprint_panel(ax) -> None:
    summary = pd.read_csv(PKG / "artifacts" / "metrics" / "main" / "main_window_metrics.csv")
    p = summary[summary.model == "physiocat"].set_index("outcome")
    u = summary[summary.model == "matched_no_delay"].set_index("outcome")
    sbp_ratio = 100.0 * p.loc["SBP", "mae"] / u.loc["SBP", "mae"]
    dbp_ratio = 100.0 * p.loc["DBP", "mae"] / u.loc["DBP", "mae"]
    labels = ["SBP MAE", "DBP MAE", "Params", "FLOPs", "Latency"]
    vals = [sbp_ratio, dbp_ratio, 100.0, 100.0, 100.0 * 1.86 / 1.84]
    y = np.arange(len(labels))[::-1]
    ax.axvspan(78, 100, color="#E8F6F2", zorder=0)
    ax.barh(y[:2], vals[:2], height=0.55, color=TEAL, alpha=0.24, edgecolor="none")
    ax.scatter(vals, y, s=[34, 34, 24, 24, 28], color=[TEAL, TEAL, "#52627A", "#52627A", "#52627A"], zorder=3)
    ax.axvline(100, color="#52627A", lw=0.9, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlim(78, 103)
    ax.set_xlabel("PhysioCAT / no-delay control (%)")
    ax.set_title("Matched footprint contrast", loc="left", fontsize=10.5, weight="bold")
    ax.text(vals[0] + 0.75, y[0], f"{vals[0]-100:+.1f}%", va="center", color=TEAL, fontsize=8, weight="bold")
    ax.text(vals[1] + 0.75, y[1], f"{vals[1]-100:+.1f}%", va="center", color=TEAL, fontsize=8, weight="bold")
    ax.text(100.6, y[2], "matched", va="center", color="#52627A", fontsize=8, weight="bold")
    ax.text(100.6, y[3], "matched", va="center", color="#52627A", fontsize=8, weight="bold")
    ax.text(99.5, y[4], "+0.02 ms", va="center", ha="right", color="#52627A", fontsize=8, weight="bold")
    style_ax(ax)
    panel_label(ax, "D")


def deployment_panel(ax) -> None:
    dep = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "deployment_profile.csv")
    platforms = ["H800", "Xeon", "Orin FP16", "Orin INT8"]
    lat = dep["end_to_end_latency_ms"].to_numpy(float)
    y = np.arange(len(platforms))[::-1]
    colors = [TEAL, GREY, BLUE, PURPLE]
    ax.barh(y, lat, color=colors, height=0.55)
    ax.axvline(8000, color="#D97A3A", ls="--", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(3, 10000)
    ax.set_yticks(y, platforms)
    ax.set_xlabel("Processing latency after window acquisition (ms)")
    ax.set_title("Post-acquisition processing", loc="left", fontsize=10.5, weight="bold")
    for y0, v in zip(y, lat):
        ax.text(v * 1.08, y0, f"{v:g} ms", va="center", fontsize=8, color=INK)
    style_ax(ax)
    panel_label(ax, "E")


def failure_panel(ax) -> None:
    fail = pd.read_csv(PKG / "artifacts" / "metrics" / "secondary" / "error_tail_composition.csv")
    order = ["Low SQI", "Elevated-BP windows", "Irregular rhythm", "BP extreme", "Other"]
    colors = {
        "Low SQI": TEAL,
        "Elevated-BP windows": RED,
        "Irregular rhythm": PURPLE,
        "BP extreme": BLUE,
        "Other": GREY,
    }
    labels = {"Low SQI": "Low SQI", "Elevated-BP windows": "Elev. BP", "Irregular rhythm": "Rhythm", "BP extreme": "BP ext.", "Other": "Other"}
    ax.set_title("Descriptive error-tail composition", loc="left", fontsize=9.5, weight="bold", pad=31)
    bar_positions = [0.62, -0.30]
    for yi, endpoint in zip(bar_positions, ["SBP", "DBP"]):
        subset = fail.loc[fail["endpoint"] == endpoint].set_index("composition_category").loc[order]
        left = 0.0
        for category, row in subset.iterrows():
            width = float(row["tail_share_pct"])
            ax.barh(yi, width, left=left, color=colors[category], height=0.55, edgecolor="white", linewidth=0.8)
            if width >= 8:
                ax.text(left + width / 2, yi, labels[category], ha="center", va="center", color="white", fontsize=7.8)
            left += width
    handles = [Patch(facecolor=colors[name], edgecolor="none", label=labels[name]) for name in order]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        fontsize=5.7,
        handlelength=0.9,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 1.22)
    ax.set_yticks(bar_positions, ["SBP tail", "DBP tail"])
    ax.set_xlabel("Share of 90th-percentile error tail (%)")
    style_ax(ax)
    ax.text(-0.11, 1.24, "F", transform=ax.transAxes, fontsize=16, weight="bold", color=INK)


def main() -> None:
    global PKG, PDF, PNG
    parser = argparse.ArgumentParser(description="Render submitted Figure 7 from released sources")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None)
    args = parser.parse_args()
    PKG = args.package_root
    PDF = args.output
    PNG = args.png
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(11.2, 7.3), dpi=220)
    gs = fig.add_gridspec(
        3,
        3,
        width_ratios=[1.05, 1.18, 1.05],
        height_ratios=[0.92, 0.92, 0.86],
        wspace=0.44,
        hspace=0.52,
    )

    ax_a = fig.add_subplot(gs[0:2, 0])
    subgroup_panel(ax_a)

    ax_b = fig.add_subplot(gs[0, 1:])
    draw_wave_panel(ax_b, "easy")
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 1:])
    draw_wave_panel(ax_c, "hard")
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[2, 0])
    footprint_panel(ax_d)
    ax_e = fig.add_subplot(gs[2, 1])
    deployment_panel(ax_e)
    ax_f = fig.add_subplot(gs[2, 2])
    failure_panel(ax_f)

    PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, metadata=PDF_METADATA)
    if PNG is not None:
        PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG)
    plt.close(fig)
    print(f"Wrote {PDF}")
    if PNG is not None:
        print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
