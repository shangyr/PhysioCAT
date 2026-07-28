from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Verify five-fold subject-grouped and random-segment OOF membership")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced")
    args = parser.parse_args()
    archive = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    subjects = archive["subject_id"].astype(str)
    roles = archive["roles"]
    if roles.shape != (5, len(subjects)):
        raise AssertionError("Fold-role matrix shape mismatch")
    if not np.all((roles == 2).sum(axis=0) == 1):
        raise AssertionError("Every subject must occur in exactly one outer-test fold")
    if not np.all((roles == 1).sum(axis=1) == 271):
        raise AssertionError("Each fold must contain 271 validation subjects")
    if not np.all((roles == 0).sum(axis=1) + (roles == 1).sum(axis=1) + (roles == 2).sum(axis=1) == len(subjects)):
        raise AssertionError("Fold roles do not cover every subject")
    manifest = pd.read_csv(ROOT / "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz", compression="gzip")
    validation = pd.read_csv(ROOT / "data/folds/pulsedb_vital_validation_membership.csv.gz", compression="gzip")
    test = pd.read_csv(ROOT / "data/folds/pulsedb_vital_test_membership.csv.gz", compression="gzip")
    if len(manifest) != 5 or set(manifest.fold_id.astype(int)) != {1, 2, 3, 4, 5}:
        raise AssertionError("Subject-grouped fold manifest must contain five folds")
    for fold_id in range(1, 6):
        expected_test = set(subjects[roles[fold_id - 1] == 2])
        expected_validation = set(subjects[roles[fold_id - 1] == 1])
        observed_test = set(test.loc[test.fold_id.eq(fold_id), "test_subject_id"].astype(str))
        observed_validation = set(validation.loc[validation.fold_id.eq(fold_id), "validation_subject_id"].astype(str))
        if observed_test != expected_test or observed_validation != expected_validation:
            raise AssertionError(f"Fold {fold_id} membership tables differ from the role matrix")
        if expected_test & expected_validation:
            raise AssertionError(f"Fold {fold_id} has validation/test overlap")

    random_membership = pd.read_csv(
        ROOT / "data/folds/pulsedb_vital_random_segment_5fold_membership.csv.gz"
    )
    random_summary = pd.read_csv(
        ROOT / "data/folds/pulsedb_vital_random_segment_5fold_summary.csv"
    )
    random_predictions = pd.read_csv(
        ROOT / "artifacts/predictions/protocol_random_split_predictions.csv.gz"
    )
    if not random_membership.window_id.is_unique or not random_predictions.window_id.is_unique:
        raise AssertionError("Every random-segment window must have one membership and one OOF prediction")
    if set(random_membership.window_id) != set(random_predictions.window_id):
        raise AssertionError("Random-segment membership and prediction windows differ")
    bound = random_predictions[["window_id", "evaluation_fold_id"]].merge(
        random_membership[["window_id", "evaluation_fold_id"]],
        on="window_id", suffixes=("_prediction", "_membership"), validate="one_to_one",
    )
    if not np.array_equal(bound.evaluation_fold_id_prediction, bound.evaluation_fold_id_membership):
        raise AssertionError("Random-segment prediction fold IDs differ from membership")
    if set(random_predictions.prediction_role) != {"out_of_fold_test"}:
        raise AssertionError("Random-segment predictions must be explicitly out of fold")
    all_windows = set(random_membership.window_id)
    for fold_id in range(5):
        test = set(random_membership.loc[random_membership.segment_partition_id == fold_id, "window_id"])
        validation = set(random_membership.loc[random_membership.segment_partition_id == (fold_id + 1) % 5, "window_id"])
        train = set(random_membership.loc[~random_membership.segment_partition_id.isin([fold_id, (fold_id + 1) % 5]), "window_id"])
        if train & validation or train & test or validation & test:
            raise AssertionError(f"Random-segment role overlap in fold {fold_id}")
        if train | validation | test != all_windows:
            raise AssertionError(f"Random-segment incomplete coverage in fold {fold_id}")
    if len(random_summary) != 5 or not (random_summary.test_subject_overlap_pct > 99).all():
        raise AssertionError("Random-segment subject-overlap summary is incomplete")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"status": "PASS", "subject_grouped_folds": roles.shape[0], "subjects": len(subjects), "outer_test_memberships": int((roles == 2).sum()), "validation_memberships": int((roles == 1).sum()), "partition_overlap_cells": 0, "random_segment_folds": 5, "random_segment_oof_windows": len(random_predictions), "random_segment_min_subject_overlap_pct": float(random_summary.test_subject_overlap_pct.min())}
    (args.output_dir / "fold_membership_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
