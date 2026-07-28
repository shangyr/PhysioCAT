from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.baselines import build_neural_baseline
from physiocat.checkpointing import save_inference_checkpoint
from physiocat.dataio import load_prepared_npz
from physiocat.models import build_model
from physiocat.training import FitConfig, WaveformDataset, fit_model, input_view_for_model


def model_for(name: str):
    if name in {"physiocat", "matched_no_delay"}:
        return build_model(name)
    if name == "mufubp_net":
        return build_neural_baseline(name)
    raise ValueError("External zero-shot training is defined for physiocat, matched_no_delay, and mufubp_net")


def main():
    parser = argparse.ArgumentParser(description="Train the frozen PulseDB-Vital source model used for external zero-shot evaluation")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", choices=["physiocat", "matched_no_delay", "mufubp_net"], required=True)
    parser.add_argument("--validation-subjects", type=Path, required=True, help="Text file containing one source validation subject per line")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--configuration-manifest", type=Path, default=ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    parser.add_argument("--max-training-windows-per-subject", type=int, default=None, help="Optional deterministic cap for smoke runs; omitted in the reported source-model protocol")
    parser.add_argument("--max-validation-windows-per-subject", type=int, default=None, help="Optional deterministic cap for smoke runs; omitted in the reported source-model protocol")
    args = parser.parse_args()
    archive = load_prepared_npz(args.archive)
    validation_subjects = {line.strip() for line in args.validation_subjects.read_text(encoding="utf-8").splitlines() if line.strip()}
    validation = np.isin(archive["subject_id"].astype(str), list(validation_subjects))
    train = ~validation
    train_indices = []
    subject_ids = archive["subject_id"].astype(str)
    for subject in np.unique(subject_ids[train]):
        subject_indices = np.flatnonzero(train & (subject_ids == subject))
        if args.max_training_windows_per_subject is not None:
            subject_indices = subject_indices[: args.max_training_windows_per_subject]
        train_indices.extend(subject_indices)
    train_indices = np.asarray(train_indices, dtype=int)
    validation_indices = []
    for subject in np.unique(subject_ids[validation]):
        subject_indices = np.flatnonzero(validation & (subject_ids == subject))
        if args.max_validation_windows_per_subject is not None:
            subject_indices = subject_indices[: args.max_validation_windows_per_subject]
        validation_indices.extend(subject_indices)
    validation_indices = np.asarray(validation_indices, dtype=int)
    configs = pd.read_csv(args.configuration_manifest)
    selected = configs[configs.model.astype(str) == args.model]
    if len(selected) != 1:
        raise ValueError(f"Expected one frozen configuration for {args.model}; found {len(selected)}")
    selected = selected.iloc[0]
    input_view = input_view_for_model(args.model)
    if str(selected.input_view) != input_view:
        raise ValueError("Frozen source-model configuration has the wrong input view")
    pretrain = int(selected.pretrain_epochs) if args.model in {"physiocat", "matched_no_delay"} else 0
    model, history = fit_model(
        lambda: model_for(args.model),
        WaveformDataset(archive, train_indices, input_view=input_view),
        WaveformDataset(archive, validation_indices, input_view=input_view),
        args.output_dir,
        FitConfig(
            seed=int(selected.initialization_seed),
            data_order_seed=int(selected.data_order_seed),
            maximum_epochs=int(selected.maximum_epochs),
            pretrain_epochs=pretrain,
            batch_size=int(selected.batch_size),
            learning_rate=float(selected.learning_rate),
            weight_decay=float(selected.weight_decay),
            patience=int(selected.early_stopping_patience),
            pretrain_enabled=pretrain > 0,
        ),
    )
    supervised = [row for row in history if row["stage"] == "supervised"]
    objectives = np.asarray([row["validation_objective"] for row in supervised], dtype=float)
    best_epoch = int(np.argmin(objectives) + 1)
    checkpoint = args.output_dir / "best_checkpoint.npz"
    save_inference_checkpoint(
        checkpoint,
        {
            "model_slug": args.model,
            "configuration_id": str(selected.configuration_id),
            "configuration_sha256": str(selected.configuration_sha256),
            "source_split": "PulseDB-Vital 2,442/272 subject split",
            "selected_epoch": best_epoch,
            "selected_validation_mean_mae": float(objectives[best_epoch - 1]),
            "input_view": input_view,
        },
        model.state_dict(),
    )
    report = {
        "status": "PASS",
        "model": args.model,
        "input_view": input_view,
        "training_windows": len(train_indices),
        "training_subjects": int(np.unique(archive["subject_id"][train]).size),
        "validation_windows": len(validation_indices),
        "validation_subjects": len(validation_subjects),
        "configuration_id": str(selected.configuration_id),
        "configuration_sha256": str(selected.configuration_sha256),
        "epochs_completed": len(supervised),
        "best_epoch": best_epoch,
        "best_validation_mean_mae": float(objectives[best_epoch - 1]),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "external_target_data_read": False,
    }
    (args.output_dir / "source_training_lineage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
