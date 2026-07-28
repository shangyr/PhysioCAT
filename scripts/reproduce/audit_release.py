from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run structural and leakage audit of the release")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = [
        "README.md", "REVIEWER_GUIDE.md", "docs/SCIENTIFIC_CONTRACT.md",
        "assets/physiocat_architecture.png", "assets/subject_grouped_results.png",
        "assets/README.md", "examples/released_checkpoint_demo.py",
        "requirements/requirements-lock.txt",
        "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        "artifacts/predictions/pulsedb_mimic_zero_shot_predictions.csv.gz",
        "artifacts/predictions/mimic_bp_zero_shot_predictions.csv.gz",
        "artifacts/predictions/random_mask_20_seed_predictions.npz",
        "artifacts/attention/sparse_attention_weights.npz", "artifacts/attention/attention_export_manifest.csv", "scripts/reproduce/verify_attention_export.py", "artifacts/replay/replay_manifest.csv",
        "artifacts/predictions/prediction_authority_manifest.csv", "artifacts/logs/training/model_configuration_registry.csv", "artifacts/logs/training/fold_local_training_setting_selection.csv", "artifacts/protocol/design_parameter_provenance.csv", "artifacts/logs/training/prediction_authority_manifest.csv", "artifacts/logs/training/formal_training_run_ledger.csv.gz", "artifacts/logs/training/stability_run_ledger.csv.gz", "artifacts/logs/training/source_model_training_run_ledger.csv.gz", "artifacts/logs/training/formal_training_compute_summary.csv", "artifacts/logs/training/released_training_evidence_summary.json", "artifacts/logs/training/configuration_validation_test_summary.csv", "artifacts/logs/training/core_validation_test_consistency.json", "artifacts/checkpoints/checkpoint_manifest.csv", "artifacts/provenance/study_design_lock.json", "artifacts/provenance/dataset_field_and_label_contract.csv", "artifacts/provenance/comparator_configuration_provenance.csv", "scripts/reproduce/verify_training_lineage.py", "artifacts/metrics/mechanism/mask_row_offset_audit.csv",
        "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz", "data/folds/fold_subject_roles.npz", "data/folds/pulsedb_vital_validation_membership.csv.gz", "data/folds/pulsedb_vital_test_membership.csv.gz", "data/folds/pulsedb_vital_random_segment_5fold_membership.csv.gz", "data/folds/pulsedb_vital_random_segment_5fold_summary.csv", "configs/evaluation/random_mask_stability.yaml", "configs/evaluation/random_segment_5fold.yaml",
        "data/retention/pulsedb_vital_raw_candidate_manifest.csv.gz", "data/retention/pulsedb_mimic_raw_candidate_manifest.csv.gz", "data/retention/mimic_bp_raw_candidate_manifest.csv.gz",
        "data/fixtures/end_to_end_fixture.npz", "src/physiocat/models.py", "src/physiocat/preprocessing.py", "src/physiocat/training.py", "src/physiocat/baselines.py", "src/physiocat/adapters.py", "src/physiocat/perturbations.py", "src/physiocat/checkpointing.py", "configs/model/uniform_delay_band.yaml",
        "scripts/train/prepare_dataset.py", "scripts/train/train_fold.py", "scripts/train/train_classical_fold.py", "scripts/train/train_source_model.py", "scripts/train/predict_source_model.py", "scripts/train/run_subject_grouped_cv.py", "scripts/train/run_random_segment_cv.py",
        "artifacts/protocol/source_model_protocol.csv", "artifacts/protocol/external_prediction_manifest.csv", "artifacts/protocol/source_model_prediction_lineage.csv", "artifacts/protocol/source_model_three_seed_protocol.csv", "data/manifests/source_model_validation_windows.csv.gz", "artifacts/metrics/external/source_model_internal_validation.csv", "artifacts/metrics/external/source_model_three_seed_metrics.csv", "artifacts/metrics/external/source_model_three_seed_summary.csv", "artifacts/predictions/source_model_three_seed_subject_statistics.csv.gz", "artifacts/predictions/source_validation/physiocat_predictions.csv.gz", "artifacts/predictions/source_validation/matched_no_delay_predictions.csv.gz", "artifacts/predictions/source_validation/mufubp_net_predictions.csv.gz",
        "scripts/data/export_pulsedb.py", "scripts/data/export_mimic_bp.py", "scripts/data/canonicalize_candidate_lineage.py", "configs/data/pulsedb_adapter.yaml", "configs/data/mimic_bp_adapter.yaml", "scripts/reproduce/audit_cross_document_consistency.py", "paper/figures/Figure_3.pdf", "paper/figures/Figure_7.pdf", "artifacts/waveforms/representative_windows.npz", "artifacts/representations/source_shift_input_features.npz", "artifacts/predictions/negative_control_subject_statistics.csv.gz", "artifacts/predictions/pulsedb_vital_target_formed_predictions.csv.gz", "artifacts/protocol/timing_perturbation_protocol.csv", "artifacts/cohorts/target_formation_failure_summary.csv", "artifacts/cohorts/target_formation_selection_audit.csv", "artifacts/cohorts/label_source_audit.csv", "artifacts/quality/sqi_validation_annotations.csv.gz",
        "paper/figure_sources/diagrams/Figure_1_source.pptx", "paper/figure_sources/diagrams/Figure_2_source.pptx", "paper/figure_sources/diagrams/Figure_4_base.pdf",
        "artifacts/attention/figure4_panel_manifest.csv", "artifacts/attention/figure4_example_attention_matrices.csv.gz",
        "artifacts/metrics/secondary/delay_band_sweep.csv", "artifacts/metrics/secondary/equal_width_offset_sweep.csv", "artifacts/metrics/secondary/normalization_factorial.csv", "artifacts/metrics/secondary/calibration_diagnostics.csv", "artifacts/metrics/secondary/conditional_bp_performance.csv", "artifacts/metrics/secondary/pat_stratified_model_comparison.csv", "artifacts/metrics/secondary/pat_group_interaction_contrasts.csv", "artifacts/metrics/secondary/retention_sensitivity.csv", "artifacts/metrics/secondary/repository_scalar_sensitivity.csv", "artifacts/metrics/secondary/sqi_reference_validation.csv", "artifacts/metrics/secondary/abp_reference_quality_validation.csv", "artifacts/metrics/secondary/abp_reference_quality_sensitivity.csv", "artifacts/quality/abp_reference_quality_annotations.csv.gz", "scripts/reproduce/reproduce_secondary_analyses.py", "scripts/reproduce/reproduce_profiling.py", "artifacts/profiling/runtime_samples.csv.gz", "artifacts/profiling/profile_environments.csv", "artifacts/profiling/int8_agreement_summary.csv", "scripts/profile/profile_pytorch.py", "scripts/profile/export_onnx.py", "scripts/profile/profile_onnx.py", "scripts/profile/build_tensorrt_engine.py", "scripts/profile/README.md", "scripts/reproduce/render_figure_3.py", "scripts/reproduce/render_figure_4.py", "scripts/reproduce/render_figure_5.py", "scripts/reproduce/render_figure_6.py", "scripts/reproduce/render_figure_7.py",
    ]
    rows = []
    for relative in required:
        exists = (ROOT / relative).is_file()
        rows.append({"check": "required_file", "item": relative, "status": "PASS" if exists else "FAIL"})
        if not exists:
            raise FileNotFoundError(relative)
    forbidden_path_tokens = [
        "class" + "room", "teach" + "ing", "syn" + "thetic", "histor" + "ical",
        "leg" + "acy", "rec" + "over", "7." + "14.4", "7." + "15.1", "7." + "15.2", "visual_" + "master",
    ]
    path_hits = []
    for path in ROOT.rglob("*"):
        relative = str(path.relative_to(ROOT)).replace("\\", "/").lower()
        for token in forbidden_path_tokens:
            if token in relative:
                path_hits.append(f"{relative}:{token}")
    if path_hits:
        raise AssertionError(f"Non-release path provenance leakage found: {path_hits}")
    quality_fields = ["adult_metadata", "scalar_target_valid", "beat_count_pass", "ecg_quality_pass", "ppg_quality_pass", "paired_sample_continuity", "sqi_rule_pass"]
    retention_rows_checked = 0
    raw_candidate_rows_checked = 0
    target_formation_failures = 0
    target_failure_repository_rows = 0
    external_cap_failures = 0
    for slug in ("pulsedb_vital", "pulsedb_mimic", "mimic_bp"):
        raw = pd.read_csv(ROOT / f"data/retention/{slug}_raw_candidate_manifest.csv.gz", compression="gzip", keep_default_na=False)
        retention = pd.read_csv(ROOT / f"data/retention/{slug}_window_audit.csv.gz", compression="gzip")
        raw_candidate_rows_checked += len(raw)
        target_formation_failures += int((~raw.target_derivation_pass.astype(bool)).sum())
        required_raw_fields = {
            "age_years", "sex", "repository_sbp", "repository_dbp",
            "repository_scalar_available", "input_min_sqi", "rr_irregularity",
        }
        if not required_raw_fields.issubset(raw.columns):
            raise AssertionError(f"{slug} raw candidate manifest omits target-formation audit fields")
        successful = raw.target_derivation_pass.astype(bool)
        if not raw.loc[successful, "repository_scalar_available"].astype(bool).all():
            raise AssertionError(f"{slug} target-formed rows lack their declared scalar reference")
        if slug.startswith("pulsedb_"):
            target_failure = raw.paired_waveform_crop_pass.astype(bool) & ~successful
            available_failure = target_failure & raw.repository_scalar_available.astype(bool)
            target_failure_repository_rows += int(available_failure.sum())
            if int(available_failure.sum()) < 0.60 * int(target_failure.sum()):
                raise AssertionError(f"{slug} target-formation audit has insufficient repository-scalar coverage")
            if not np.isfinite(raw.loc[available_failure, ["repository_sbp", "repository_dbp", "age_years", "input_min_sqi", "rr_irregularity"]].to_numpy(float)).all():
                raise AssertionError(f"{slug} target-formation comparison contains non-finite audit values")
        if raw.source_record_hash.duplicated().any():
            raise AssertionError(f"{slug} contains repeated source_record_hash values despite the one-segment/one-crop contract")
        accepted = raw.loc[raw.target_derivation_pass.astype(bool), "window_id"]
        if len(accepted) != len(retention) or accepted.nunique() != len(retention) or set(accepted) != set(retention.window_id):
            raise AssertionError(f"{slug} raw enumeration does not map one-to-one onto target-formed candidates")
        retention_rows_checked += len(retention)
        inclusion_fields = quality_fields + (["subject_balance_cap_pass"] if slug == "pulsedb_vital" else [])
        reconstructed = retention[inclusion_fields].astype(bool).all(axis=1)
        if not (reconstructed == retention.retained.astype(bool)).all():
            raise AssertionError(f"{slug} retention cannot be reconstructed from declared non-PAT fields")
        cascade = pd.read_csv(ROOT / f"data/retention/{slug}_retention_cascade.csv")
        expected_start = ["Source-indexed adult waveform segments", "Paired ECG/PPG crop available", "Scalar target formed or paired"]
        if cascade.stage.iloc[:3].tolist() != expected_start:
            raise AssertionError(f"{slug} retention cascade does not expose crop and target formation")
        if int(cascade.windows_remaining.iloc[0]) != len(raw) or int(cascade.windows_remaining.iloc[2]) != len(retention):
            raise AssertionError(f"{slug} retention cascade start does not match released candidate manifests")
        if int(cascade.windows_remaining.iloc[-1]) != int(retention.retained.astype(bool).sum()):
            raise AssertionError(f"{slug} retention cascade does not end at the released cohort size")
        if slug != "pulsedb_vital":
            failures = int((~retention.subject_balance_cap_pass.astype(bool)).sum())
            external_cap_failures += failures
            if failures:
                raise AssertionError(f"{slug} contains an undeclared post-quality cap or target-size trim")

    nonfinite_prediction_values = 0
    prediction_files = [
        "pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        "pulsedb_mimic_zero_shot_predictions.csv.gz",
        "mimic_bp_zero_shot_predictions.csv.gz",
        "protocol_random_split_predictions.csv.gz",
        "mechanism_control_predictions.csv.gz",
    ]
    for name in prediction_files:
        frame = pd.read_csv(ROOT / "artifacts/predictions" / name, compression="gzip")
        if "source_record_hash" in frame and int(frame.groupby("source_record_hash").size().max()) > 1:
            raise AssertionError(f"{name} violates the declared one-source-segment/one-window lineage")
        for column in [value for value in frame.columns if value.startswith("pred_")]:
            values = frame[column].to_numpy(float)
            nonfinite_prediction_values += int((~np.isfinite(values)).sum())
    for name in (
        "negative_control_subject_statistics.csv.gz",
        "source_model_three_seed_subject_statistics.csv.gz",
    ):
        frame = pd.read_csv(ROOT / "artifacts/predictions" / name, compression="gzip")
        for column in ("n_windows", "absolute_error_sum", "signed_error_sum"):
            values = frame[column].to_numpy(float)
            nonfinite_prediction_values += int((~np.isfinite(values)).sum())
    if nonfinite_prediction_values:
        raise AssertionError(f"Released predictions contain {nonfinite_prediction_values} non-finite values")

    primary = pd.read_csv(
        ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        compression="gzip",
    )
    roles_archive = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    role_subjects = roles_archive["subject_id"].astype(str)
    roles = roles_archive["roles"]
    rf_convex_hull_violations = 0
    rf_exact_hull_hits = 0
    for fold_zero in range(roles.shape[0]):
        train_subjects = set(role_subjects[roles[fold_zero] == 0])
        test = primary[primary.evaluation_fold_id.eq(fold_zero + 1)]
        train = primary[primary.subject_id.astype(str).isin(train_subjects)]
        for outcome in ("sbp", "dbp"):
            lower = float(train[outcome].min())
            upper = float(train[outcome].max())
            values = test[f"pred_random_forest_{outcome}"].to_numpy(float)
            rf_convex_hull_violations += int((values < lower).sum() + (values > upper).sum())
            rf_exact_hull_hits += int(np.isclose(values, lower, atol=1e-7, rtol=0).sum())
            rf_exact_hull_hits += int(np.isclose(values, upper, atol=1e-7, rtol=0).sum())
    if rf_convex_hull_violations:
        raise AssertionError(f"Random-forest OOF predictions violate {rf_convex_hull_violations} fold-specific training-label hulls")

    registry = pd.read_csv(ROOT / "artifacts/cohorts/cohort_registry.csv").set_index("cohort")
    target_selection = pd.read_csv(ROOT / "artifacts/cohorts/target_formation_selection_audit.csv")
    repository_sensitivity = pd.read_csv(ROOT / "artifacts/metrics/secondary/repository_scalar_sensitivity.csv")
    abp_quality_sensitivity = pd.read_csv(ROOT / "artifacts/metrics/secondary/abp_reference_quality_sensitivity.csv")
    abp_quality_validation = pd.read_csv(ROOT / "artifacts/metrics/secondary/abp_reference_quality_validation.csv")
    vital_audit = pd.read_csv(ROOT / "data/retention/pulsedb_vital_window_audit.csv.gz", compression="gzip")
    if set(target_selection.cohort) != {"pulsedb_vital", "pulsedb_mimic"} or len(target_selection) != 4:
        raise AssertionError("Target-formation selection audit does not cover both PulseDB protocols")
    if set(repository_sensitivity.cohort) != {"pulsedb_vital", "pulsedb_mimic"} or len(repository_sensitivity) != 12:
        raise AssertionError("Repository-scalar sensitivity does not cover both PulseDB protocols and three core models")
    if repository_sensitivity.mae_change_repository_minus_aligned.abs().max() > 0.20:
        raise AssertionError("Repository-scalar sensitivity is inconsistent with the released target-agreement audit")
    retained_vital = vital_audit.loc[vital_audit.retained.astype(bool)]
    below_abp = int((retained_vital.abp_sqi < 0.42).sum())
    above_abp = int((retained_vital.abp_sqi >= 0.42).sum())
    if below_abp < 500 or above_abp < 500:
        raise AssertionError("Reference-ABP quality distribution is degenerate around the declared threshold")
    if (
        len(abp_quality_sensitivity) != 9
        or abp_quality_sensitivity.duplicated(["analysis_view", "model"]).any()
        or not {"sbp_mae", "dbp_mae"}.issubset(abp_quality_sensitivity.columns)
        or len(abp_quality_validation) != 1
    ):
        raise AssertionError("Reference-ABP sensitivity or validation evidence is incomplete")
    for _, group in abp_quality_sensitivity.groupby("analysis_view"):
        ordered = group.set_index("model")
        if not (ordered.loc["physiocat", "sbp_mae"] < ordered.loc["mufubp_net", "sbp_mae"] < ordered.loc["matched_no_delay", "sbp_mae"]):
            raise AssertionError("Reference-ABP sensitivity does not preserve the core SBP ordering")
        if not (ordered.loc["physiocat", "dbp_mae"] < ordered.loc["mufubp_net", "dbp_mae"] < ordered.loc["matched_no_delay", "dbp_mae"]):
            raise AssertionError("Reference-ABP sensitivity does not preserve the core DBP ordering")
    report = {"status": "PASS", "required_files": len(required), "forbidden_path_hits": 0, "raw_candidate_rows_checked": raw_candidate_rows_checked, "target_formation_failures": target_formation_failures, "target_failure_repository_rows": target_failure_repository_rows, "target_selection_audit_rows": len(target_selection), "repository_scalar_sensitivity_rows": len(repository_sensitivity), "abp_reference_quality_sensitivity_rows": len(abp_quality_sensitivity), "abp_reference_quality_review_rows": int(abp_quality_validation.review_windows.iloc[0]), "abp_reference_quality_below_threshold": below_abp, "retention_rows_checked": retention_rows_checked, "source_record_max_windows": 1, "external_cap_failures": external_cap_failures, "nonfinite_prediction_values": nonfinite_prediction_values, "random_forest_convex_hull_violations": rf_convex_hull_violations, "random_forest_exact_hull_hits": rf_exact_hull_hits, "mimic_bp_fixed_subjects": int(registry.loc["mimic_bp", "subjects"]), "mimic_bp_fixed_windows": int(registry.loc["mimic_bp", "windows"])}
    (args.output_dir / "release_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.output_dir / "release_required_files.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
