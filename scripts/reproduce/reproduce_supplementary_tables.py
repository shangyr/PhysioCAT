from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import hashlib
import pandas as pd


SOURCES = [
    "artifacts/cohorts/cohort_registry.csv",
    "artifacts/provenance/baseline_implementation_provenance.csv",
    "artifacts/metrics/main/main_window_metrics.csv",
    "artifacts/metrics/bootstrap/cluster_bootstrap_ci.csv",
    "artifacts/metrics/statistics/paired_subject_tests.csv",
    "artifacts/metrics/stability/three_seed_subject_grouped_summary.csv",
    "artifacts/metrics/protocol/random_split_window_metrics.csv",
    "artifacts/metrics/mechanism/mechanism_window_metrics.csv",
    "artifacts/metrics/mechanism/random_mask_20_seed_metrics.csv",
    "artifacts/metrics/mechanism/mask_row_offset_audit.csv",
    "artifacts/attention/attention_alignment_summary.csv",
    "artifacts/metrics/external/external_window_metrics.csv",
    "artifacts/metrics/external/source_model_three_seed_metrics.csv",
    "artifacts/metrics/external/source_model_three_seed_summary.csv",
    "artifacts/metrics/statistics/repeated_measures_agreement.csv",
    "artifacts/metrics/subgroups/subgroup_subject_weighted.csv",
    "artifacts/metrics/deployment/risk_coverage.csv",
    "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz",
    "data/folds/pulsedb_vital_random_segment_5fold_summary.csv",
    "artifacts/metrics/secondary/delay_band_sweep.csv",
    "artifacts/metrics/secondary/equal_width_offset_sweep.csv",
    "artifacts/metrics/secondary/normalization_factorial.csv",
    "artifacts/metrics/secondary/calibration_diagnostics.csv",
    "artifacts/metrics/secondary/conditional_bp_performance.csv",
    "artifacts/metrics/secondary/pat_stratified_model_comparison.csv",
    "artifacts/metrics/secondary/pat_group_interaction_contrasts.csv",
    "artifacts/metrics/secondary/deployment_profile.csv",
    "artifacts/metrics/secondary/external_threshold_metrics.csv",
    "artifacts/metrics/secondary/external_zero_shot_summary.csv",
    "artifacts/metrics/secondary/error_tail_composition.csv",
    "artifacts/metrics/secondary/few_shot_calibration.csv",
    "artifacts/metrics/secondary/figure_7_subgroup_source.csv",
    "artifacts/metrics/secondary/mimic_bp_protocol_audit.csv",
    "artifacts/metrics/secondary/mimic_bp_retention_bridge.csv",
    "artifacts/metrics/secondary/model_footprint.csv",
    "artifacts/metrics/secondary/negative_controls.csv",
    "artifacts/metrics/secondary/quality_rejection.csv",
    "artifacts/metrics/secondary/retention_sensitivity.csv",
    "artifacts/metrics/secondary/repository_scalar_sensitivity.csv",
    "artifacts/metrics/secondary/sqi_reference_validation.csv",
    "artifacts/metrics/secondary/abp_reference_quality_validation.csv",
    "artifacts/metrics/secondary/abp_reference_quality_sensitivity.csv",
    "artifacts/metrics/secondary/representative_windows.csv",
    "artifacts/metrics/secondary/same_density_mask_controls.csv",
    "artifacts/metrics/secondary/source_shift_projection.csv",
    "artifacts/metrics/secondary/sqi_strata.csv",
    "artifacts/metrics/secondary/subgroup_analysis.csv",
    "artifacts/metrics/secondary/subgroup_definitions.csv",
    "artifacts/metrics/secondary/secondary_source_inventory.csv",
    "artifacts/metrics/secondary/selection_bias.csv",
    "artifacts/cohorts/label_source_audit.csv",
    "artifacts/cohorts/target_formation_selection_audit.csv",
    "artifacts/quality/sqi_validation_annotations.csv.gz",
    "artifacts/quality/abp_reference_quality_annotations.csv.gz",
    "artifacts/predictions/pulsedb_vital_target_formed_predictions.csv.gz",
    "artifacts/protocol/source_model_protocol.csv",
    "artifacts/protocol/external_prediction_manifest.csv",
    "artifacts/protocol/source_model_prediction_lineage.csv",
    "artifacts/protocol/source_model_three_seed_protocol.csv",
    "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz",
    "data/folds/pulsedb_vital_validation_membership.csv.gz",
    "data/folds/pulsedb_vital_test_membership.csv.gz",
    "artifacts/logs/training/formal_training_run_ledger.csv.gz",
    "artifacts/logs/training/stability_run_ledger.csv.gz",
    "artifacts/logs/training/source_model_training_run_ledger.csv.gz",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Validate and index all machine-readable supplementary table sources")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        frame = pd.read_csv(path, compression="infer", low_memory=False)
        if frame.empty:
            raise AssertionError(f"Empty supplementary source: {relative}")
        rows.append({"source": relative, "rows": len(frame), "columns": len(frame.columns), "sha256": sha256(path), "status": "PASS"})
    index = pd.DataFrame(rows)
    index.to_csv(args.output_dir / "supplementary_table_source_index.csv", index=False)
    report = {"status": "PASS", "sources": len(index), "total_rows": int(index.rows.sum())}
    (args.output_dir / "supplementary_table_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
