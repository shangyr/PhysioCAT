from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def fold_roles(membership: pd.DataFrame, fold_id: int) -> dict[str, pd.DataFrame]:
    if fold_id not in range(5):
        raise ValueError("fold_id must be in 0..4")
    test_partition = fold_id
    validation_partition = (fold_id + 1) % 5
    return {
        "train": membership[
            ~membership.segment_partition_id.isin([test_partition, validation_partition])
        ].copy(),
        "validation": membership[
            membership.segment_partition_id == validation_partition
        ].copy(),
        "test": membership[
            membership.segment_partition_id == test_partition
        ].copy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the fixed five-fold random-segment train/validation/test schedule"
    )
    parser.add_argument("--fold-id", type=int, choices=range(5))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced/random_segment_cv")
    args = parser.parse_args()
    membership = pd.read_csv(
        ROOT / "data/folds/pulsedb_vital_random_segment_5fold_membership.csv.gz"
    )
    fold_ids = [args.fold_id] if args.fold_id is not None else list(range(5))
    summaries = []
    all_windows = set(membership.window_id)
    for fold_id in fold_ids:
        roles = fold_roles(membership, fold_id)
        role_sets = {name: set(frame.window_id) for name, frame in roles.items()}
        if role_sets["train"] & role_sets["validation"] or role_sets["train"] & role_sets["test"] or role_sets["validation"] & role_sets["test"]:
            raise AssertionError(f"Role overlap in fold {fold_id}")
        if set().union(*role_sets.values()) != all_windows:
            raise AssertionError(f"Incomplete window coverage in fold {fold_id}")
        fold_dir = args.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for role, frame in roles.items():
            frame[["window_id", "subject_id"]].to_csv(fold_dir / f"{role}_windows.csv.gz", index=False)
        summaries.append({
            "fold_id": fold_id,
            "train_windows": len(roles["train"]),
            "validation_windows": len(roles["validation"]),
            "test_windows": len(roles["test"]),
            "prediction_role": "out_of_fold_test",
        })
    report = {"status": "PASS", "folds_materialized": len(summaries), "folds": summaries}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "schedule.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
