from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.checkpointing import load_inference_checkpoint
from physiocat.models import build_model


CHECKPOINT_DIR = (
    ROOT
    / "artifacts"
    / "checkpoints"
    / "source_models"
    / "physiocat"
    / "pulsedb_vital_frozen_source"
)


def run_demo() -> dict[str, object]:
    checkpoint_path = CHECKPOINT_DIR / "checkpoint.npz"
    input_path = CHECKPOINT_DIR / "audit_inputs.npz"
    expected_path = CHECKPOINT_DIR / "audit_predictions.csv"

    metadata, state_dict = load_inference_checkpoint(checkpoint_path)
    model = build_model(str(metadata["model_slug"]))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with np.load(input_path, allow_pickle=False) as archive:
        ecg = torch.from_numpy(archive["ecg"][:, None]).float()
        ppg = torch.from_numpy(archive["ppg"][:, None]).float()
        sqi = torch.from_numpy(archive["sqi_tokens"]).float()
        probe_id = archive["probe_id"].astype(str)

    with torch.inference_mode():
        prediction = model(ecg, ppg, sqi).cpu().numpy()

    expected = pd.read_csv(expected_path).set_index("probe_id").loc[probe_id]
    expected_values = expected[["predicted_sbp", "predicted_dbp"]].to_numpy(float)
    max_abs_delta = float(np.max(np.abs(prediction - expected_values)))
    # The authority was exported from the original GPU run. A 5e-5 mmHg
    # tolerance covers backend-level float32 accumulation differences while
    # remaining orders of magnitude below any reported precision.
    if max_abs_delta > 5e-5:
        raise AssertionError(
            f"Released checkpoint replay differs from its prediction authority: {max_abs_delta}"
        )

    return {
        "status": "PASS",
        "architecture": metadata["architecture"],
        "checkpoint_identity": metadata["checkpoint_identity"],
        "input_view": metadata["input_contract"]["input_view"],
        "windows": int(len(prediction)),
        "output_shape": list(prediction.shape),
        "max_abs_replay_delta": max_abs_delta,
        "first_prediction_mmHg": {
            "sbp": float(prediction[0, 0]),
            "dbp": float(prediction[0, 1]),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2))
