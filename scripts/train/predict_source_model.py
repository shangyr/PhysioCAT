from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.baselines import build_neural_baseline
from physiocat.checkpointing import load_inference_checkpoint
from physiocat.dataio import load_prepared_npz
from physiocat.models import build_model
from physiocat.training import WaveformDataset, evaluate, input_view_for_model


def model_for(name: str):
    if name in {"physiocat", "matched_no_delay"}:
        return build_model(name)
    if name == "mufubp_net":
        return build_neural_baseline(name)
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a trained frozen source checkpoint to one prepared target cohort")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", choices=["physiocat", "matched_no_delay", "mufubp_net"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    archive = load_prepared_npz(args.archive)
    if args.checkpoint.suffix.lower() == ".npz":
        checkpoint, state_dict = load_inference_checkpoint(args.checkpoint)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
    model = model_for(args.model)
    model.load_state_dict(state_dict, strict=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    input_view = input_view_for_model(args.model)
    loader = torch.utils.data.DataLoader(WaveformDataset(archive, input_view=input_view), batch_size=args.batch_size, shuffle=False)
    result = evaluate(model, loader, device)
    output = pd.DataFrame(
        {
            "window_id": result["window_id"],
            "subject_id": result["subject_id"],
            "reference_sbp": result["target"][:, 0],
            "reference_dbp": result["target"][:, 1],
            "predicted_sbp": result["prediction"][:, 0],
            "predicted_dbp": result["prediction"][:, 1],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    report = {
        "status": "PASS",
        "model": args.model,
        "input_view": input_view,
        "windows": len(output),
        "subjects": output.subject_id.nunique(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "target_tuning": False,
        "sbp_mae": result["sbp_mae"],
        "dbp_mae": result["dbp_mae"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
