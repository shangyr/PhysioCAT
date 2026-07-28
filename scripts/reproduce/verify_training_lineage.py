from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

RUN_RECEIPT_FIELDS = (
    "run_id", "configuration_id", "model", "outer_fold_id", "test_subjects_sha256",
    "train_subjects", "validation_subjects", "test_subjects", "train_windows",
    "validation_windows", "test_windows", "initialization_seed", "data_order_seed",
    "mask_seed", "selected_epoch", "stopped_epoch", "maximum_epochs",
    "early_stopping_patience", "selected_validation_mean_mae",
    "selected_validation_sbp_mae", "selected_validation_dbp_mae", "test_mean_mae",
    "test_sbp_mae", "test_dbp_mae", "configuration_sha256", "code_snapshot_sha256",
    "checkpoint_sha256", "checkpoint_release_status", "prediction_shard_sha256",
    "prediction_authority", "gpu_model", "gpu_worker", "run_started_utc",
    "run_finished_utc", "wall_clock_hours", "gpu_hours", "status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_prediction_digest(frame: pd.DataFrame, sbp_column: str, dbp_column: str) -> str:
    ordered = frame.sort_values("window_id", kind="mergesort")
    digest = hashlib.sha256()
    for window_id in ordered.window_id.astype(str):
        digest.update(window_id.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(ordered[[sbp_column, dbp_column]].to_numpy(np.float32)).tobytes())
    return digest.hexdigest()


def run_receipt(row) -> str:
    integer_fields = {
        "outer_fold_id", "train_subjects", "validation_subjects", "test_subjects",
        "train_windows", "validation_windows", "test_windows", "initialization_seed",
        "data_order_seed", "mask_seed", "selected_epoch", "stopped_epoch",
        "maximum_epochs", "early_stopping_patience",
    }
    metric_fields = {
        "selected_validation_mean_mae", "selected_validation_sbp_mae",
        "selected_validation_dbp_mae", "test_mean_mae", "test_sbp_mae", "test_dbp_mae",
        "wall_clock_hours", "gpu_hours",
    }
    parts = []
    for name in RUN_RECEIPT_FIELDS:
        value = getattr(row, name)
        if name in metric_fields:
            parts.append(f"{float(value):.8f}")
        elif name in integer_fields:
            parts.append(str(int(value)))
        else:
            parts.append(str(value))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def assert_nonoverlap(frame: pd.DataFrame, label: str) -> None:
    if (frame.wall_clock_hours.astype(float) <= 0).any() or (frame.gpu_hours.astype(float) <= 0).any():
        raise AssertionError(f"{label} contains non-positive compute durations")
    for worker, rows in frame.groupby("gpu_worker", sort=False):
        ordered = rows.assign(
            start=pd.to_datetime(rows.run_started_utc, utc=True, format="mixed"),
            finish=pd.to_datetime(rows.run_finished_utc, utc=True, format="mixed"),
        ).sort_values("start")
        if not (ordered.finish > ordered.start).all():
            raise AssertionError(f"{label} contains invalid timestamps on {worker}")
        if len(ordered) > 1 and (ordered.start.iloc[1:].to_numpy() < ordered.finish.iloc[:-1].to_numpy()).any():
            raise AssertionError(f"{label} contains overlapping jobs on {worker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify five-fold training, inference, and checkpoint lineage")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    folds = pd.read_csv(ROOT / "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz")
    tests = pd.read_csv(ROOT / "data/folds/pulsedb_vital_test_membership.csv.gz")
    representativeness = pd.read_csv(ROOT / "data/folds/fold_subset_representativeness.csv")
    authorities = pd.read_csv(ROOT / "artifacts/logs/training/prediction_authority_manifest.csv")
    ledger = pd.read_csv(ROOT / "artifacts/logs/training/formal_training_run_ledger.csv.gz", keep_default_na=False)
    stability = pd.read_csv(ROOT / "artifacts/logs/training/stability_run_ledger.csv.gz")
    source_runs = pd.read_csv(ROOT / "artifacts/logs/training/source_model_training_run_ledger.csv.gz")
    checkpoints = pd.read_csv(ROOT / "artifacts/checkpoints/checkpoint_manifest.csv", keep_default_na=False)
    source_lineage = pd.read_csv(ROOT / "artifacts/protocol/source_model_prediction_lineage.csv")
    summary = json.loads((ROOT / "artifacts/logs/training/released_training_evidence_summary.json").read_text(encoding="utf-8"))
    consistency = json.loads((ROOT / "artifacts/logs/training/core_validation_test_consistency.json").read_text(encoding="utf-8"))
    validation_test_summary = pd.read_csv(ROOT / "artifacts/logs/training/configuration_validation_test_summary.csv")

    trained = registry[registry.outer_folds.eq(5)].copy()
    random_masks = trained[trained.model.str.startswith("attention_edge_ablation_seed_")].copy()
    if len(registry) != 50 or len(trained) != 50 or len(random_masks) != 20:
        raise AssertionError("Configuration registry does not contain 50 complete five-fold configurations including 20 random-mask topologies")
    if not registry.outer_folds.eq(5).all():
        raise AssertionError("Every registered configuration must use the complete five-fold protocol")
    if random_masks.mask_seed.nunique() != 20 or set(random_masks.mask_seed.astype(int)) != set(range(82000, 82020)):
        raise AssertionError("Random-mask topology seeds are incomplete")
    if not random_masks.initialization_seed.eq(42).all() or not random_masks.data_order_seed.eq(42).all():
        raise AssertionError("Random-mask topology controls must hold initialization and data order fixed")
    if len(folds) != 5 or set(folds.fold_id.astype(int)) != {1, 2, 3, 4, 5}:
        raise AssertionError("Five-fold subject-grouped manifest is incomplete")
    if tests.test_subject_id.nunique() != 2714 or tests.groupby("test_subject_id").size().ne(1).any():
        raise AssertionError("Every subject must occur in exactly one outer-test fold")
    if representativeness.selection_used_model_predictions.astype(bool).any() or len(representativeness) != 5:
        raise AssertionError("Outer-fold representativeness audit is incomplete")

    if len(authorities) != len(trained) or set(authorities.configuration_id) != set(trained.configuration_id):
        raise AssertionError("Prediction authorities do not cover every trained configuration")
    subject_to_fold = dict(zip(tests.test_subject_id.astype(str), tests.fold_id.astype(int), strict=True))
    prediction_cache: dict[str, pd.DataFrame] = {}
    registry_by_configuration = registry.set_index("configuration_id")
    fold_metrics: dict[tuple[str, int], tuple[float, float, float]] = {}
    for row in authorities.itertuples(index=False):
        path = ROOT / row.prediction_authority
        if path.suffix.lower() == ".npz":
            archive = np.load(path, allow_pickle=False)
            seed = int(registry_by_configuration.loc[row.configuration_id, "mask_seed"])
            matches = np.flatnonzero(archive["seed"].astype(int) == seed)
            if len(matches) != 1:
                raise AssertionError(f"Random-mask prediction seed is missing or duplicated: {seed}")
            index = int(matches[0])
            frame = pd.DataFrame(
                {
                    "window_id": archive["window_id"].astype(str),
                    "subject_id": archive["subject_id"].astype(str),
                    "sbp": archive["reference_sbp"].astype(float),
                    "dbp": archive["reference_dbp"].astype(float),
                    row.sbp_column: archive["predicted_sbp"][index].astype(float),
                    row.dbp_column: archive["predicted_dbp"][index].astype(float),
                }
            )
        else:
            frame = prediction_cache.setdefault(row.prediction_authority, pd.read_csv(path))
        if set(frame.subject_id.astype(str)) != set(subject_to_fold):
            raise AssertionError(f"Prediction authority subject scope mismatch: {row.configuration_id}")
        if int(row.outer_folds) != 5 or int(row.rows) != len(frame):
            raise AssertionError(f"Prediction authority row/fold mismatch: {row.configuration_id}")
        if row.prediction_authority_sha256 != sha256_file(path):
            raise AssertionError(f"Prediction authority file hash mismatch: {row.configuration_id}")
        if row.prediction_values_sha256 != canonical_prediction_digest(frame, row.sbp_column, row.dbp_column):
            raise AssertionError(f"Prediction value digest mismatch: {row.configuration_id}")
        mapped = frame.subject_id.astype(str).map(subject_to_fold)
        for fold_id, shard in frame.groupby(mapped, sort=True):
            sbp = float(np.mean(np.abs(shard[row.sbp_column] - shard.sbp)))
            dbp = float(np.mean(np.abs(shard[row.dbp_column] - shard.dbp)))
            fold_metrics[(row.configuration_id, int(fold_id))] = (sbp, dbp, 0.5 * (sbp + dbp))

    if len(ledger) != 250 or set(ledger.configuration_id) != set(trained.configuration_id):
        raise AssertionError("Formal five-fold training ledger is incomplete")
    if not ledger.groupby("configuration_id").outer_fold_id.nunique().eq(5).all():
        raise AssertionError("Each trained configuration must contain five formal fits")
    if not ledger.run_receipt_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise AssertionError("Formal run receipts are malformed")
    for row in ledger.itertuples(index=False):
        if row.run_receipt_sha256 != run_receipt(row):
            raise AssertionError(f"Formal run receipt mismatch: {row.run_id}")
        expected = fold_metrics[(row.configuration_id, int(row.outer_fold_id))]
        observed = (float(row.test_sbp_mae), float(row.test_dbp_mae), float(row.test_mean_mae))
        if not np.allclose(observed, expected, atol=5e-8, rtol=0):
            raise AssertionError(f"Formal test metrics differ from prediction authority: {row.run_id}")
        if abs(0.5 * (row.selected_validation_sbp_mae + row.selected_validation_dbp_mae) - row.selected_validation_mean_mae) > 5e-8:
            raise AssertionError(f"Validation components do not reproduce checkpoint objective: {row.run_id}")
    assert_nonoverlap(ledger, "formal ledger")

    if len(validation_test_summary) != 50 or set(validation_test_summary.configuration_id) != set(trained.configuration_id):
        raise AssertionError("Configuration-level validation/test summary is incomplete")
    if consistency.get("status") != "PASS" or int(consistency.get("paired_folds", 0)) != 5:
        raise AssertionError("Core validation/test consistency summary is incomplete")

    random_mask_runs = ledger[ledger.model.str.startswith("attention_edge_ablation_seed_")]
    if len(random_mask_runs) != 100 or random_mask_runs.configuration_id.nunique() != 20:
        raise AssertionError("Independently trained random-mask topology ledger is incomplete")
    if not random_mask_runs.groupby("configuration_id").outer_fold_id.nunique().eq(5).all():
        raise AssertionError("Every random-mask topology must contain five independent training fits")
    if not random_mask_runs.initialization_seed.eq(42).all() or not random_mask_runs.data_order_seed.eq(42).all():
        raise AssertionError("Random-mask formal runs do not hold initialization and data order fixed")

    if len(stability) != 40 or set(stability.initialization_seed) != {1337, 2025}:
        raise AssertionError("Additional optimization-seed stability ledger is incomplete")
    if set(stability.model) != {"physiocat", "matched_no_delay", "mufubp_net", "ppg_leading_mirror"}:
        raise AssertionError("Stability ledger model scope is incorrect")
    if not stability.groupby(["model", "initialization_seed"]).outer_fold_id.nunique().eq(5).all():
        raise AssertionError("Each additional seed/model pair must cover all five outer folds")
    assert_nonoverlap(stability, "stability ledger")
    if len(source_runs) != 9 or set(source_runs.model) != {"physiocat", "matched_no_delay", "mufubp_net"}:
        raise AssertionError("Source-model training ledger is incomplete")
    if set(source_runs.initialization_seed) != {42, 1337, 2025} or source_runs.groupby("model").initialization_seed.nunique().ne(3).any():
        raise AssertionError("Each source-transfer model must contain all three source-training seeds")
    if source_runs.target_tuning.astype(bool).any():
        raise AssertionError("Source-model seed repeats may not tune on target cohorts")
    assert_nonoverlap(source_runs, "source-model ledger")
    assert_nonoverlap(pd.concat([ledger, stability, source_runs], ignore_index=True, sort=False), "complete training campaign")

    representative = checkpoints[checkpoints.checkpoint_role.eq("representative_outer_fold")]
    frozen_source = checkpoints[checkpoints.checkpoint_role.eq("frozen_source_model")]
    if len(representative) != 10 or set(representative.outer_fold_id.astype(int)) != {1, 2, 3, 4, 5}:
        raise AssertionError("Complete paired core-model checkpoint set is missing")
    if set(representative.model) != {"physiocat", "matched_no_delay"} or not representative.groupby("outer_fold_id").size().eq(2).all():
        raise AssertionError("Core checkpoint pairing is incomplete")
    if len(frozen_source) != 3 or len(source_lineage) != 9 or len(checkpoints) != 13:
        raise AssertionError("Checkpoint/source-model lineage is incomplete")
    for row in checkpoints.itertuples(index=False):
        for relative in (row.checkpoint, row.audit_inputs, row.audit_predictions, row.training_log):
            if not (ROOT / relative).is_file():
                raise AssertionError(f"Missing checkpoint artifact: {relative}")
        if sha256_file(ROOT / row.checkpoint) != row.checkpoint_sha256:
            raise AssertionError(f"Checkpoint hash mismatch: {row.checkpoint}")
        history = pd.read_csv(ROOT / row.training_log)
        selected = history[history.selected_checkpoint.astype(bool)]
        if len(selected) != 1 or int(selected.epoch.iloc[0]) != int(row.selected_epoch):
            raise AssertionError(f"Checkpoint-selection trace mismatch: {row.training_log}")
        if row.checkpoint_role == "representative_outer_fold":
            validation_path = ROOT / row.validation_predictions
            if not validation_path.is_file() or sha256_file(validation_path) != row.validation_predictions_sha256:
                raise AssertionError(f"Validation shard missing or altered: {row.validation_predictions}")
            validation = pd.read_csv(validation_path)
            sbp = float(np.mean(np.abs(validation.predicted_sbp - validation.reference_sbp)))
            dbp = float(np.mean(np.abs(validation.predicted_dbp - validation.reference_dbp)))
            if validation.subject_id.nunique() != 271 or not np.allclose(
                [sbp, dbp, 0.5 * (sbp + dbp)],
                [row.selected_validation_sbp_mae, row.selected_validation_dbp_mae, row.selected_validation_mean_mae],
                atol=5e-8, rtol=0,
            ):
                raise AssertionError(f"Validation shard does not reproduce checkpoint metric: {row.validation_predictions}")

    expected_summary = {
        "five_fold_configurations": 50,
        "independently_trained_random_mask_configurations": 20,
        "prediction_authorities_verified": 50,
        "formal_runs": 250,
        "additional_stability_runs": 40,
        "source_model_runs": 9,
        "additional_source_model_seed_runs": 6,
        "random_mask_training_runs": 100,
        "outer_folds": 5,
        "representative_outer_fold_checkpoints": 10,
        "frozen_source_model_checkpoints": 3,
    }
    if summary.get("status") != "PASS" or any(int(summary.get(key, -1)) != value for key, value in expected_summary.items()):
        raise AssertionError("Released training-evidence summary is inconsistent")
    if "complete five-fold subject-grouped OOF" not in summary.get("scope", ""):
        raise AssertionError("Training-evidence scope is not declared accurately")

    report = {
        "status": "PASS",
        **expected_summary,
        "released_checkpoints_verified": len(checkpoints),
        "representative_validation_prediction_rows": int(representative.validation_prediction_rows.sum()),
        "source_prediction_lineage_rows": len(source_lineage),
        "formal_gpu_hours": float(ledger.gpu_hours.sum()),
        "stability_gpu_hours": float(stability.gpu_hours.sum()),
        "source_model_gpu_hours": float(source_runs.gpu_hours.sum()),
        "core_validation_test_ordering": consistency,
        "scope": summary["scope"],
    }
    (args.output_dir / "training_lineage_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
