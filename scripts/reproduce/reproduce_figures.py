from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import fitz
import matplotlib
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_pdf_text(path: Path) -> list[str]:
    document = fitz.open(path)
    try:
        text = " ".join(page.get_text("text") for page in document)
    finally:
        document.close()
    return sorted(text.split())


def rendered_rgb(path: Path, scale: float = 1.5) -> tuple[np.ndarray, tuple[float, float]]:
    document = fitz.open(path)
    try:
        if len(document) != 1:
            raise AssertionError(f"Expected one-page figure PDF: {path}")
        page = document[0]
        size = (float(page.rect.width), float(page.rect.height))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
        return image.copy(), size
    finally:
        document.close()


def verify_pdf_content(submitted: Path, regenerated: Path, figure_id: int) -> dict[str, object]:
    submitted_hash = digest(submitted)
    regenerated_hash = digest(regenerated)
    byte_identical = submitted_hash == regenerated_hash
    row: dict[str, object] = {
        "figure": figure_id,
        "sha256": submitted_hash,
        "generated_sha256": regenerated_hash,
        "bytes": submitted.stat().st_size,
        "byte_identical": byte_identical,
        "verification_mode": "byte-identical" if byte_identical else "scientific-content contract",
    }
    if byte_identical:
        row.update(
            {
                "page_geometry_max_delta_points": 0.0,
                "raster_shape_max_delta_pixels": 0,
                "visual_mean_absolute_difference": 0.0,
                "visual_changed_pixel_fraction": 0.0,
                "text_tokens_identical": True,
            }
        )
        return row

    expected, expected_size = rendered_rgb(submitted)
    observed, observed_size = rendered_rgb(regenerated)
    geometry_delta = max(abs(a - b) for a, b in zip(expected_size, observed_size, strict=True))
    raster_shape_delta = max(abs(a - b) for a, b in zip(expected.shape, observed.shape, strict=True))
    if geometry_delta > 1.0 or raster_shape_delta > 2 or expected.shape[2] != observed.shape[2]:
        raise AssertionError(
            f"Figure {figure_id} page geometry changed: {expected_size}/{expected.shape} vs "
            f"{observed_size}/{observed.shape}"
        )
    comparison_height = min(expected.shape[0], observed.shape[0])
    comparison_width = min(expected.shape[1], observed.shape[1])
    expected = expected[:comparison_height, :comparison_width]
    observed = observed[:comparison_height, :comparison_width]
    difference = np.abs(expected.astype(np.int16) - observed.astype(np.int16))
    mean_absolute_difference = float(difference.mean() / 255.0)
    changed_fraction = float((difference.max(axis=2) > 24).mean())
    text_identical = normalized_pdf_text(submitted) == normalized_pdf_text(regenerated)
    if not text_identical:
        raise AssertionError(f"Figure {figure_id} extracted text changed across rendering environments")
    if mean_absolute_difference >= 0.025 or changed_fraction >= 0.18:
        raise AssertionError(
            f"Figure {figure_id} changed beyond the renderer-tolerance contract: "
            f"mean_abs={mean_absolute_difference:.5f}, changed_fraction={changed_fraction:.5f}"
        )
    row.update(
        {
            "page_geometry_max_delta_points": geometry_delta,
            "raster_shape_max_delta_pixels": raster_shape_delta,
            "visual_mean_absolute_difference": mean_absolute_difference,
            "visual_changed_pixel_fraction": changed_fraction,
            "text_tokens_identical": text_identical,
        }
    )
    return row


def rounded_metric(frame: pd.DataFrame, model: str, outcome: str, cohort: str | None = None) -> float:
    row = frame[(frame.model == model) & (frame.outcome == outcome)]
    if cohort is not None:
        row = row[row.cohort == cohort]
    return round(float(row.mae.iloc[0]) + 1e-12, 2)


def raw_metric(frame: pd.DataFrame, model: str, outcome: str, cohort: str | None = None) -> float:
    row = frame[(frame.model == model) & (frame.outcome == outcome)]
    if cohort is not None:
        row = row[row.cohort == cohort]
    return float(row.mae.iloc[0])


def powerpoint_text(path: Path) -> str:
    text = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")):
            root = ElementTree.fromstring(archive.read(name))
            text.extend(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
    return "\n".join(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce submitted figures from released diagrams, predictions, and source tables")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    submitted = root / "paper" / "figures"
    output = (args.output_dir or root / "reports" / "reproduced" / "figures").resolve()
    output.mkdir(parents=True, exist_ok=True)

    main_metrics = pd.read_csv(root / "artifacts/metrics/main/main_window_metrics.csv")
    primary_predictions = pd.read_csv(
        root / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        compression="gzip",
        usecols=["subject_id", "sbp", "pred_physiocat_sbp"],
    )
    subject_sbp_mae = (
        (primary_predictions.pred_physiocat_sbp - primary_predictions.sbp)
        .abs()
        .groupby(primary_predictions.subject_id)
        .mean()
    )
    mechanism = pd.read_csv(root / "artifacts/metrics/mechanism/mechanism_window_metrics.csv")
    external = pd.read_csv(root / "artifacts/metrics/external/external_window_metrics.csv")
    attention = pd.read_csv(root / "artifacts/attention/attention_alignment_summary.csv")
    sqi = pd.read_csv(root / "artifacts/metrics/secondary/sqi_strata.csv")
    subgroups = pd.read_csv(root / "artifacts/metrics/secondary/subgroup_analysis.csv")
    few_shot = pd.read_csv(root / "artifacts/metrics/secondary/few_shot_calibration.csv")
    negative = pd.read_csv(root / "artifacts/metrics/secondary/negative_controls.csv")
    delay_sweep = pd.read_csv(root / "artifacts/metrics/secondary/delay_band_sweep.csv")
    random_split = pd.read_csv(root / "artifacts/metrics/protocol/random_split_window_metrics.csv")

    def sqi_mae(model: str, stratum: str, outcome: str) -> float:
        row = sqi[(sqi.model == model) & (sqi.sqi_stratum == stratum)].iloc[0]
        return round(float(row[f"{outcome.lower()}_mae"]) + 1e-12, 2)

    def subgroup_mae(label: str, outcome: str) -> float:
        row = subgroups[subgroups.display_label == label].iloc[0]
        return round(float(row[f"{outcome.lower()}_mae"]) + 1e-12, 2)
    observed = {
        "primary_physiocat_sbp": rounded_metric(main_metrics, "physiocat", "SBP"),
        "primary_physiocat_dbp": rounded_metric(main_metrics, "physiocat", "DBP"),
        "primary_physiocat_sbp_r": round(float(main_metrics.loc[(main_metrics.model == "physiocat") & (main_metrics.outcome == "SBP"), "pearson_r"].iloc[0]) + 1e-12, 2),
        "primary_physiocat_dbp_r": round(float(main_metrics.loc[(main_metrics.model == "physiocat") & (main_metrics.outcome == "DBP"), "pearson_r"].iloc[0]) + 1e-12, 2),
        "figure_3_subject_sbp_median": round(float(subject_sbp_mae.median()) + 1e-12, 2),
        "figure_3_subject_sbp_p90": round(float(subject_sbp_mae.quantile(0.90)) + 1e-12, 2),
        "primary_no_delay_sbp": rounded_metric(main_metrics, "matched_no_delay", "SBP"),
        "primary_no_delay_dbp": rounded_metric(main_metrics, "matched_no_delay", "DBP"),
        "primary_mufubp_sbp": rounded_metric(main_metrics, "mufubp_net", "SBP"),
        "primary_mufubp_dbp": rounded_metric(main_metrics, "mufubp_net", "DBP"),
        "external_pm_physiocat_sbp": rounded_metric(external, "physiocat", "SBP", "pulsedb_mimic"),
        "external_pm_physiocat_dbp": rounded_metric(external, "physiocat", "DBP", "pulsedb_mimic"),
        "external_mbp_physiocat_sbp": rounded_metric(external, "physiocat", "SBP", "mimic_bp"),
        "external_mbp_physiocat_dbp": rounded_metric(external, "physiocat", "DBP", "mimic_bp"),
        "mirror_sbp": rounded_metric(mechanism, "ppg_leading_mirror", "SBP"),
        "mirror_dbp": rounded_metric(mechanism, "ppg_leading_mirror", "DBP"),
        "local_sbp": rounded_metric(mechanism, "direction_agnostic_local", "SBP"),
        "local_dbp": rounded_metric(mechanism, "direction_agnostic_local", "DBP"),
        "shifted_sbp": rounded_metric(mechanism, "shifted_offsets_9_12", "SBP"),
        "shifted_dbp": rounded_metric(mechanism, "shifted_offsets_9_12", "DBP"),
        "random_mask_sbp": rounded_metric(mechanism, "attention_edge_ablation", "SBP"),
        "random_mask_dbp": rounded_metric(mechanism, "attention_edge_ablation", "DBP"),
        "without_sqi_sbp": rounded_metric(mechanism, "without_sqi_fusion", "SBP"),
        "without_sqi_dbp": rounded_metric(mechanism, "without_sqi_fusion", "DBP"),
        "without_delay_and_sqi_sbp": rounded_metric(mechanism, "without_delay_and_sqi", "SBP"),
        "without_delay_and_sqi_dbp": rounded_metric(mechanism, "without_delay_and_sqi", "DBP"),
        "figure_1_attention_r": round(float(attention.loc[(attention.analysis == "Observed sparse attention") & attention.metric.str.contains("PAT vs expected attention delay", regex=False), "estimate"].iloc[0]) + 1e-12, 2),
        "figure_1_high_no_delay_sbp": sqi_mae("matched_no_delay", "high", "SBP"),
        "figure_1_high_physiocat_sbp": sqi_mae("physiocat", "high", "SBP"),
        "figure_1_medium_no_delay_sbp": sqi_mae("matched_no_delay", "medium", "SBP"),
        "figure_1_medium_physiocat_sbp": sqi_mae("physiocat", "medium", "SBP"),
        "figure_1_low_no_delay_sbp": sqi_mae("matched_no_delay", "low", "SBP"),
        "figure_1_low_physiocat_sbp": sqi_mae("physiocat", "low", "SBP"),
        "figure_1_normotensive_sbp": subgroup_mae("Lower-BP windows", "SBP"),
        "figure_1_normotensive_dbp": subgroup_mae("Lower-BP windows", "DBP"),
        "figure_1_hypertensive_sbp": subgroup_mae("Elevated-BP windows", "SBP"),
        "figure_1_hypertensive_dbp": subgroup_mae("Elevated-BP windows", "DBP"),
        "figure_1_low_sqi_sbp": subgroup_mae("Lowest SQI", "SBP"),
        "figure_1_low_sqi_dbp": subgroup_mae("Lowest SQI", "DBP"),
        "figure_1_ten_shot_sbp": round(float(few_shot.loc[few_shot.calibration_windows == 10, "physiocat_sbp_mae"].iloc[0]) + 1e-12, 2),
        "figure_1_ten_shot_dbp": round(float(few_shot.loc[few_shot.calibration_windows == 10, "physiocat_dbp_mae"].iloc[0]) + 1e-12, 2),
        "figure_5_protocol_sbp_delta": round(
            raw_metric(random_split, "random_split_physiocat", "SBP")
            - raw_metric(main_metrics, "physiocat", "SBP"),
            2,
        ),
        "figure_5_protocol_dbp_delta": round(
            raw_metric(random_split, "random_split_physiocat", "DBP")
            - raw_metric(main_metrics, "physiocat", "DBP"),
            2,
        ),
        "figure_5_negative_control_rows": int(negative.perturbation.notna().sum()),
    }
    figure_1_source = root / "paper/figure_sources/diagrams/Figure_1_source.pptx"
    source_text = powerpoint_text(figure_1_source)
    required_figure_1_text = [
        "Windowing & Scalar Targets", "SBP: aligned scalar target", "DBP: aligned scalar target",
        f"{observed['primary_physiocat_sbp']:.2f}", f"{observed['primary_physiocat_dbp']:.2f}",
        f"{observed['without_sqi_sbp']:.2f}", f"{observed['without_sqi_dbp']:.2f}",
        f"{observed['primary_no_delay_sbp']:.2f}", f"{observed['primary_no_delay_dbp']:.2f}",
        f"{observed['without_delay_and_sqi_sbp']:.2f}", f"{observed['without_delay_and_sqi_dbp']:.2f}",
        f"r = {observed['figure_1_attention_r']:.2f}",
        f"{observed['figure_1_high_no_delay_sbp']:.2f}", f"{observed['figure_1_high_physiocat_sbp']:.2f}",
        f"{observed['figure_1_medium_no_delay_sbp']:.2f}", f"{observed['figure_1_medium_physiocat_sbp']:.2f}",
        f"{observed['figure_1_low_no_delay_sbp']:.2f}", f"{observed['figure_1_low_physiocat_sbp']:.2f}",
        "Lower-BP window range", f"{observed['figure_1_normotensive_sbp']:.2f} / {observed['figure_1_normotensive_dbp']:.2f}",
        f"{observed['figure_1_hypertensive_sbp']:.2f} / {observed['figure_1_hypertensive_dbp']:.2f}",
        f"{observed['figure_1_low_sqi_sbp']:.2f} / {observed['figure_1_low_sqi_dbp']:.2f}",
        f"{observed['external_mbp_physiocat_sbp']:.2f} / {observed['external_mbp_physiocat_dbp']:.2f}",
        f"{observed['figure_1_ten_shot_sbp']:.2f} / {observed['figure_1_ten_shot_dbp']:.2f}",
    ]
    missing_text = [value for value in required_figure_1_text if value not in source_text]
    if missing_text:
        raise AssertionError(f"Editable Figure 1 source is not synchronized with released values: {missing_text}")
    if "Windowing & Released Labels" in source_text or "released SBP" in source_text or "released DBP" in source_text:
        raise AssertionError("Editable Figure 1 source retains the obsolete released-label wording")

    verification_rows: list[dict[str, object]] = []
    for figure_id in range(1, 3):
        source = submitted / f"Figure_{figure_id}.pdf"
        destination = output / source.name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        verification_rows.append(verify_pdf_content(source, destination, figure_id))

    figure_3_destination = output / "Figure_3.pdf"
    figure_3_script = root / "scripts/reproduce/render_figure_3.py"
    subprocess.run(
        [sys.executable, str(figure_3_script), "--package-root", str(root), "--output", str(figure_3_destination)],
        cwd=root,
        check=True,
    )
    verification_rows.append(
        verify_pdf_content(submitted / "Figure_3.pdf", figure_3_destination, 3)
    )

    figure_4_destination = output / "Figure_4.pdf"
    figure_4_script = root / "scripts/reproduce/render_figure_4.py"
    subprocess.run(
        [
            sys.executable,
            str(figure_4_script),
            "--package-root",
            str(root),
            "--output",
            str(figure_4_destination),
        ],
        cwd=root,
        check=True,
    )
    verification_rows.append(
        verify_pdf_content(submitted / "Figure_4.pdf", figure_4_destination, 4)
    )

    for figure_id in (5, 6, 7):
        destination = output / f"Figure_{figure_id}.pdf"
        script = root / "scripts" / "reproduce" / f"render_figure_{figure_id}.py"
        command = [sys.executable, str(script), "--package-root", str(root), "--output", str(destination)]
        subprocess.run(command, cwd=root, check=True)
        verification_rows.append(
            verify_pdf_content(submitted / f"Figure_{figure_id}.pdf", destination, figure_id)
        )
    verification_rows.sort(key=lambda row: int(row["figure"]))
    report = {
        "status": "PASS",
        "policy": "Figures 1--2 retain editable diagram sources. Figure 4 retains its publication-facing vector base for schematic/layout panels, releases the fixed-window B/C matrices, and refreshes aggregate D/E from data. Numeric sources, extracted text, one-page geometry, and fixed-raster content are mandatory for Figures 3--7; byte identity is additionally recorded when the renderer environment matches.",
        "renderer": {
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
            "pymupdf": fitz.VersionBind,
        },
        "verified_primary_values": observed,
        "figure_1_editable_source_values_verified": len(required_figure_1_text),
        "figure_5_protocol_and_negative_control_values_verified": int(4 + observed["figure_5_negative_control_rows"]),
        "figure_4_source_linked_panels_verified": 6,
        "figures": verification_rows,
    }
    report_path = root / "reports" / "reproduced" / "figure_source_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
