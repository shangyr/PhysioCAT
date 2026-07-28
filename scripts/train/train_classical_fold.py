from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.baselines import EngineeredRandomForest, PATRidge, pat_ridge_features
from physiocat.dataio import load_prepared_npz
from physiocat.training import input_view_for_model, select_waveform_view, subject_partition


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one fold-local classical comparator")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--model", choices=["pat_ridge", "random_forest"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--rf-estimators", type=int, default=500)
    args = parser.parse_args()

    archive = load_prepared_npz(args.archive)
    roles = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    subjects = roles["subject_id"].astype(str)
    row = roles["roles"][args.fold_id - 1]
    test_subject = subjects[np.flatnonzero(row == 2)[0]]
    validation_subjects = set(subjects[row == 1])
    train_idx, _, test_idx = subject_partition(archive["subject_id"], test_subject, validation_subjects)
    input_view = input_view_for_model(args.model)
    ecg_view, ppg_view = select_waveform_view(archive, input_view)
    train_ecg, train_ppg = ecg_view[train_idx], ppg_view[train_idx]
    test_ecg, test_ppg = ecg_view[test_idx], ppg_view[test_idx]
    targets = archive["targets"][train_idx]
    if args.model == "pat_ridge":
        model = PATRidge(args.ridge_alpha).fit(pat_ridge_features(train_ecg, train_ppg), targets)
        prediction = model.predict(pat_ridge_features(test_ecg, test_ppg))
        parameter_count = int(model.coef_.size)
    else:
        model = EngineeredRandomForest(args.rf_estimators).fit(train_ecg, train_ppg, targets)
        prediction = model.predict(test_ecg, test_ppg)
        parameter_count = args.rf_estimators
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"window_id": archive["window_id"][test_idx], "subject_id": archive["subject_id"][test_idx], "reference_sbp": archive["targets"][test_idx, 0], "reference_dbp": archive["targets"][test_idx, 1], "predicted_sbp": prediction[:, 0], "predicted_dbp": prediction[:, 1]})
    result.to_csv(args.output_dir / "test_predictions.csv", index=False)
    mae = np.mean(np.abs(prediction - archive["targets"][test_idx]), axis=0)
    print(json.dumps({"status": "PASS", "model": args.model, "input_view": input_view, "fold_id": args.fold_id, "test_subject": test_subject, "fit_subjects": len(set(archive["subject_id"][train_idx])), "reported_parameter_count": parameter_count, "sbp_mae": float(mae[0]), "dbp_mae": float(mae[1])}, indent=2))


if __name__ == "__main__":
    main()
