from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.baselines import build_neural_baseline
from physiocat.models import MatchedNoDelayPhysioCAT, PhysioCAT

REPLAY_TOLERANCE_MMHG = 1e-2


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(slug):
    if slug == "physiocat":
        return PhysioCAT()
    if slug == "matched_no_delay":
        return MatchedNoDelayPhysioCAT()
    if slug in {"mufubp_net", "te_sagru", "bp_net", "cnn_bilstm"}:
        return build_neural_baseline(slug)
    raise KeyError(slug)


def infer(checkpoint_path, input_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_slug"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    archive = np.load(input_path, allow_pickle=False)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(archive["ecg"]), 64):
            predictions.append(model(torch.from_numpy(archive["ecg"][start:start + 64, None]).float(), torch.from_numpy(archive["ppg"][start:start + 64, None]).float(), torch.from_numpy(archive["sqi_tokens"][start:start + 64]).float()).numpy())
    return checkpoint, archive, np.vstack(predictions)


def main():
    parser = argparse.ArgumentParser(description="Verify explicitly scoped functional architecture fixtures")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = pd.read_csv(ROOT / "artifacts/replay/replay_manifest.csv")
    for row in manifest.itertuples(index=False):
        checkpoint_path, input_path = ROOT / row.checkpoint, ROOT / row.input_tensor
        checkpoint, archive, prediction = infer(checkpoint_path, input_path)
        archived_view = str(archive["input_view"][0])
        if archived_view != row.input_view or checkpoint["input_contract"].get("input_view") != row.input_view:
            raise AssertionError(f"Functional fixture input-view mismatch for {row.model}")
        expected = pd.read_csv(ROOT / row.expected_predictions).set_index("window_id").loc[archive["window_id"].astype(str)]
        expected_delta = float(np.max(np.abs(prediction - expected[["predicted_sbp", "predicted_dbp"]].to_numpy())))
        if expected_delta > REPLAY_TOLERANCE_MMHG:
            raise AssertionError(f"Functional fixture replay mismatch for {row.model}: expected={expected_delta}")
        if "not a final subject-grouped outer-fold checkpoint" not in checkpoint.get("architecture_role", ""):
            raise AssertionError(f"Functional replay scope is ambiguous for {row.model}")
        training_log = pd.read_csv(ROOT / row.training_log)
        expected_pretrain = 5 if row.model in {"physiocat", "matched_no_delay"} else 0
        if int((training_log.stage == "contrastive").sum()) != expected_pretrain or int((training_log.stage == "supervised").sum()) != 20:
            raise AssertionError(f"Functional replay stage log mismatch for {row.model}")
        rows.append({"model": row.model, "scope": "separate public functional fixture; not final subject-grouped training", "fold_subject_id": row.fold_subject_id, "n_windows": len(prediction), "architecture": checkpoint["architecture"], "state_tensors": len(checkpoint["state_dict"]), "checkpoint_bytes": checkpoint_path.stat().st_size, "checkpoint_sha256": sha256(checkpoint_path), "max_abs_expected_delta": expected_delta, "max_abs_release_delta": np.nan})

    report = pd.DataFrame(rows)
    report.to_csv(args.output_dir / "replay_verification.csv", index=False)
    summary = {"status": "PASS", "neural_replays": len(report), "functional_fixtures": len(manifest), "windows": int(report.n_windows.sum()), "minimum_checkpoint_bytes": int(report.checkpoint_bytes.min()), "minimum_state_tensors": int(report.state_tensors.min()), "max_abs_delta": float(np.nanmax(report[["max_abs_expected_delta", "max_abs_release_delta"]].to_numpy()))}
    (args.output_dir / "replay_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
