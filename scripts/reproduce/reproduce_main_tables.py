from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from physiocat.metrics import MAIN_MODELS, EXTERNAL_MODELS, read_predictions, summarize, frame_difference


def main():
    parser = argparse.ArgumentParser(description="Recompute main, mechanism, and external window-level tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = pd.read_csv(ROOT / "artifacts/provenance/baseline_implementation_provenance.csv")
    displays = dict(zip(provenance.model, provenance.model.map(lambda x: x)))
    reference_main = pd.read_csv(ROOT / "artifacts/metrics/main/main_window_metrics.csv")
    displays.update(dict(zip(reference_main.model, reference_main.display_name)))
    primary = read_predictions(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz")
    main_table = summarize(primary, MAIN_MODELS, "pulsedb_vital", displays)
    delta = frame_difference(reference_main, main_table)
    if delta.select_dtypes("number").to_numpy().max() > 7e-5:
        raise AssertionError("Main table differs from released reference")
    main_table.to_csv(args.output_dir / "main_window_metrics.csv", index=False, float_format="%.6f")

    reference_external = pd.read_csv(ROOT / "artifacts/metrics/external/external_window_metrics.csv")
    displays.update(dict(zip(reference_external.model, reference_external.display_name)))
    pm = read_predictions(ROOT / "artifacts/predictions/pulsedb_mimic_zero_shot_predictions.csv.gz")
    mbp = read_predictions(ROOT / "artifacts/predictions/mimic_bp_zero_shot_predictions.csv.gz")
    external = pd.concat([
        summarize(pm, EXTERNAL_MODELS, "pulsedb_mimic", displays),
        summarize(mbp, EXTERNAL_MODELS, "mimic_bp", displays),
    ], ignore_index=True)
    delta_ext = frame_difference(reference_external, external)
    if delta_ext.select_dtypes("number").to_numpy().max() > 7e-5:
        raise AssertionError("External table differs from released reference")
    external.to_csv(args.output_dir / "external_window_metrics.csv", index=False, float_format="%.6f")
    reference_protocol = pd.read_csv(ROOT / "artifacts/metrics/protocol/random_split_window_metrics.csv")
    protocol_models = list(dict.fromkeys(reference_protocol.model.tolist()))
    displays.update(dict(zip(reference_protocol.model, reference_protocol.display_name)))
    protocol_frame = read_predictions(ROOT / "artifacts/predictions/protocol_random_split_predictions.csv.gz")
    protocol = summarize(protocol_frame, protocol_models, "pulsedb_vital_random_segment_split", displays)
    delta_protocol = frame_difference(reference_protocol, protocol)
    if delta_protocol.select_dtypes("number").to_numpy().max() > 7e-5:
        raise AssertionError("Protocol-sensitivity table differs from released reference")
    protocol.to_csv(args.output_dir / "random_split_window_metrics.csv", index=False, float_format="%.6f")
    reference_mechanism = pd.read_csv(ROOT / "artifacts/metrics/mechanism/mechanism_window_metrics.csv")
    displays.update(dict(zip(reference_mechanism.model, reference_mechanism.display_name)))
    factorized_models = ["without_sqi_fusion", "without_delay_and_sqi"]
    mechanism_models = [model for model in dict.fromkeys(reference_mechanism.model.tolist()) if model not in factorized_models]
    complete_frame = read_predictions(ROOT / "artifacts/predictions/factorized_ablation_predictions.csv.gz")
    mechanism_frame = read_predictions(ROOT / "artifacts/predictions/mechanism_control_predictions.csv.gz")
    factorized = summarize(complete_frame, factorized_models, "pulsedb_vital", displays)
    secondary = summarize(mechanism_frame, mechanism_models, "pulsedb_vital", displays)
    factorized["evaluation_scope"] = "complete five-fold subject-grouped OOF"
    secondary["evaluation_scope"] = "complete five-fold subject-grouped OOF"
    mechanism = pd.concat([factorized, secondary], ignore_index=True)
    delta_mechanism = frame_difference(reference_mechanism, mechanism)
    if delta_mechanism.select_dtypes("number").to_numpy().max() > 7e-5:
        raise AssertionError("Mechanism table differs from released reference")
    mechanism.to_csv(args.output_dir / "mechanism_window_metrics.csv", index=False, float_format="%.6f")
    report = {"status": "PASS", "main_rows": len(main_table), "external_rows": len(external), "protocol_rows": len(protocol), "mechanism_rows": len(mechanism), "max_abs_main_delta": float(delta.select_dtypes("number").to_numpy().max()), "max_abs_external_delta": float(delta_ext.select_dtypes("number").to_numpy().max()), "max_abs_protocol_delta": float(delta_protocol.select_dtypes("number").to_numpy().max()), "max_abs_mechanism_delta": float(delta_mechanism.select_dtypes("number").to_numpy().max())}
    (args.output_dir / "main_table_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
