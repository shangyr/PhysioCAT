from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_MODELS = ["physiocat", "matched_no_delay", "mufubp_net", "te_sagru", "bp_net", "cnn_bilstm", "random_forest", "pat_ridge"]
EXTERNAL_MODELS = ["physiocat", "matched_no_delay", "mufubp_net"]


def read_predictions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", low_memory=False)


def metric_row(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - y
    return {
        "me": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "sde": float(np.std(residual, ddof=1)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
        "within_5_pct": float(np.mean(np.abs(residual) <= 5) * 100),
        "within_10_pct": float(np.mean(np.abs(residual) <= 10) * 100),
        "within_15_pct": float(np.mean(np.abs(residual) <= 15) * 100),
    }


def summarize(frame: pd.DataFrame, models: list[str], cohort: str, display_names: dict[str, str]) -> pd.DataFrame:
    rows = []
    for model in models:
        for outcome in ("sbp", "dbp"):
            row = metric_row(frame[outcome].to_numpy(), frame[f"pred_{model}_{outcome}"].to_numpy())
            row.update({"cohort": cohort, "model": model, "display_name": display_names[model], "outcome": outcome.upper(), "n_windows": len(frame), "n_subjects": frame.subject_id.nunique()})
            rows.append(row)
    return pd.DataFrame(rows)


def subject_errors(frame: pd.DataFrame, model: str, outcome: str) -> pd.Series:
    work = pd.DataFrame({"subject_id": frame.subject_id, "ae": np.abs(frame[f"pred_{model}_{outcome}"] - frame[outcome])})
    return work.groupby("subject_id", sort=False).ae.mean()


def frame_difference(reference: pd.DataFrame, reproduced: pd.DataFrame, keys=("cohort", "model", "outcome")) -> pd.DataFrame:
    numeric = ["me", "mae", "sde", "rmse", "pearson_r", "within_5_pct", "within_10_pct", "within_15_pct"]
    a = reference.set_index(list(keys))[numeric].astype(float).sort_index()
    b = reproduced.set_index(list(keys))[numeric].astype(float).sort_index()
    if not a.index.equals(b.index):
        raise AssertionError("Reference and reproduced table keys differ")
    delta = (a - b).abs()
    return delta.reset_index()
