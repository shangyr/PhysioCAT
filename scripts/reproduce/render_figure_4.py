from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Public defaults are package-relative.  The teacher-side build passes its
# source, package root, and output explicitly, so no historical or local
# workspace path is embedded in the distributed renderer.
DEFAULT_PACKAGE = Path.cwd()
DEFAULT_TARGET = DEFAULT_PACKAGE / "paper" / "figures" / "Figure_4.pdf"
PANEL_METADATA = {
    "Creator": "PhysioCAT release renderer",
    "CreationDate": None,
    "ModDate": None,
}


def no_delay_heatmap(package: Path) -> bytes:
    matrices = pd.read_csv(package / "artifacts/attention/figure4_example_attention_matrices.csv.gz")
    free = matrices.pivot(
        index="ecg_token", columns="ppg_token", values="matched_no_delay_attention"
    ).to_numpy(float)
    if free.shape != (125, 125) or not np.allclose(free[:-3].sum(axis=1), 1.0, atol=2e-8):
        raise AssertionError("Figure 4C requires the frozen row-normalized 125-token no-delay export")
    vmax = float(np.quantile(free[:-3], 0.997))
    fig = plt.figure(figsize=(3.2, 2.65), dpi=300, facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.imshow(
        free,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_axis_off()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, facecolor="white", bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return buffer.getvalue()


def attention_panel(package: Path) -> tuple[bytes, float]:
    summary = pd.read_csv(package / "artifacts/attention/attention_alignment_summary.csv")
    row = summary[
        (summary.analysis == "Observed sparse attention")
        & summary.metric.str.contains("PAT vs expected attention delay", regex=False)
    ]
    if len(row) != 1:
        raise AssertionError("Expected one observed attention/PAT correlation")
    value = float(row.iloc[0].estimate)
    attention = pd.read_csv(package / "artifacts/attention/window_level_attention_summary.csv.gz")
    if len(attention) != 1024 or set(attention.evaluation_role.astype(str)) != {"outer_test"}:
        raise AssertionError("Figure 4 attention panel requires the frozen subject-grouped outer-test export")
    display = attention.sample(n=360, random_state=244_951).sort_values("pat_ms")
    slope, intercept, _, _, _ = stats.linregress(
        attention.pat_ms.to_numpy(float), attention.attention_expected_delay_ms.to_numpy(float)
    )

    fig, ax = plt.subplots(figsize=(2.75, 2.28), dpi=220)
    ax.scatter(display.pat_ms, display.attention_expected_delay_ms, s=11, color="#8fb7d8", alpha=0.42, linewidths=0)
    central = display.pat_ms.between(190, 360)
    ax.scatter(
        display.loc[central, "pat_ms"],
        display.loc[central, "attention_expected_delay_ms"],
        s=14,
        color="#168b87",
        alpha=0.72,
        linewidths=0,
    )
    xx = np.linspace(120, 450, 160)
    ax.plot(xx, intercept + slope * xx, color="#7360b6", linewidth=1.35)
    ax.set_xlim(110, 460)
    ax.set_ylim(185, 392)
    ax.set_xticks([200, 300, 400])
    ax.set_yticks([200, 250, 300, 350])
    ax.grid(True, color="#d7e1ef", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#263242")
    ax.tick_params(labelsize=6.4, width=0.6, length=2.5)
    ax.set_xlabel("Apparent PAT estimate (ms)", fontsize=7.1)
    ax.set_ylabel("Expected attention lag (ms)", fontsize=7.1)
    ax.set_title("Attention lag vs apparent PAT", loc="left", fontsize=8.0, fontweight="bold", pad=5)
    ax.text(0.055, 0.86, f"r = {value:.2f}", transform=ax.transAxes, fontsize=6.5, color="#172033")
    fig.text(0.015, 0.955, "D", ha="left", va="top", fontsize=12.5, fontweight="bold", color="#172033")
    fig.subplots_adjust(left=0.27, right=0.985, bottom=0.24, top=0.83)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", facecolor="white", metadata=PANEL_METADATA)
    plt.close(fig)
    return buffer.getvalue(), value


def delay_panel(package: Path) -> bytes:
    table = pd.read_csv(package / "artifacts/metrics/secondary/delay_band_sweep.csv")
    expected = ["120-300 ms", "120-350 ms", "120-450 ms", "80-550 ms", "Unconstrained"]
    if table.delay_band.astype(str).tolist() != expected:
        raise AssertionError("Delay-band source order differs from the frozen Figure 4 contract")
    x = np.arange(len(table))
    labels = ["120–300", "120–350", "120–450", "80–550", "free"]
    sbp = table.sbp_mae.to_numpy(float)
    dbp = table.dbp_mae.to_numpy(float)

    fig, ax = plt.subplots(figsize=(2.85, 2.65), dpi=220)
    ax.axvspan(1.65, 2.35, color="#dcefeb", alpha=0.72, zorder=0)
    ax.plot(x, sbp, color="#ca4c55", marker="o", linewidth=1.55, markersize=4.8, label="SBP")
    ax.plot(x, dbp, color="#347fb8", marker="s", linewidth=1.55, markersize=4.6, label="DBP")
    lower = np.floor((min(sbp.min(), dbp.min()) - 0.18) * 2.0) / 2.0
    upper = np.ceil((max(sbp.max(), dbp.max()) + 0.18) * 2.0) / 2.0
    ax.set_ylim(lower, upper)
    ax.set_xlim(-0.2, len(x) - 0.8)
    ax.set_xticks(x, labels, rotation=17, ha="right")
    ax.set_ylabel("MAE (mmHg)", fontsize=7.2)
    ax.set_xlabel("Allowed ECG-to-PPG delay band (ms)", fontsize=7.0, labelpad=2)
    ax.set_title("Delay-band sensitivity plateau", loc="left", fontsize=8.0, fontweight="bold", pad=5)
    ax.grid(axis="y", color="#d7e1ef", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=6.25, width=0.6, length=2.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.01), frameon=False, ncol=2, fontsize=6.4, handlelength=1.4, columnspacing=0.9)
    fig.text(0.012, 0.955, "E", ha="left", va="top", fontsize=12.5, fontweight="bold", color="#172033")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.27, top=0.82)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", facecolor="white", metadata=PANEL_METADATA)
    plt.close(fig)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the publication-facing Figure 4 and refresh its data-driven D/E panels")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    package = args.package_root.resolve()
    source = (args.source or package / "paper" / "figure_sources" / "diagrams" / "Figure_4_base.pdf").resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    panel_d_bytes, attention_r = attention_panel(package)
    panel_e_bytes = delay_panel(package)
    panel_c_bytes = no_delay_heatmap(package)
    shutil.copy2(source, output)
    document = fitz.open(output)
    page = document[0]
    pdf_panels = [
        (fitz.Rect(344.0, 127.0, 501.4, 266.0), panel_d_bytes),
        (fitz.Rect(0.0, 278.0, 174.0, 428.6), panel_e_bytes),
    ]
    panel_c_rect = fitz.Rect(198.8487, 144.0434, 326.0442, 248.9365)
    for rect, _ in pdf_panels + [(panel_c_rect, b"")]:
        page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_image(panel_c_rect, stream=panel_c_bytes, keep_proportion=False, overlay=True)
    for rect, payload in pdf_panels:
        panel_document = fitz.open("pdf", payload)
        page.show_pdf_page(rect, panel_document, 0, keep_proportion=False, overlay=True)
        panel_document.close()
    document.set_metadata(
        {
            "title": "PhysioCAT Figure 4",
            "author": "PhysioCAT Authors",
            "subject": "Publication-facing mechanism analysis with data-linked aggregate panels",
            "keywords": "ECG PPG attention delay",
            "creator": "PhysioCAT release renderer",
            "producer": "PyMuPDF",
        }
    )
    temporary = output.with_name(output.stem + "_numeric_sync.pdf")
    document.save(temporary, garbage=4, deflate=True, clean=True, no_new_id=True)
    document.close()
    temporary.replace(output)
    print(f"Materialized publication-facing Figure 4 and regenerated Panels C/D/E from released sources (attention r = {attention_r:.2f})")


if __name__ == "__main__":
    main()
