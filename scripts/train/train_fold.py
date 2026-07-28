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
from physiocat.models import PhysioCATConfig, build_model
from physiocat.training import FitConfig, WaveformDataset, evaluate, fit_model, input_view_for_model, subject_partition


def model_for(name):
    if name.startswith("attention_edge_ablation_seed_"):
        return build_model("attention_edge_ablation", PhysioCATConfig(mask_seed=int(name.rsplit("_", 1)[-1])))
    mechanism = {
        "physiocat", "matched_no_delay", "ppg_leading_mirror", "unidirectional_delay",
        "direction_agnostic_local", "shifted_offsets_9_12", "attention_edge_ablation",
        "without_sqi_fusion", "without_delay_and_sqi", "without_pretraining",
        "physiocat_patch_local", "matched_no_delay_patch_local",
        "offset_band_2_5_common", "offset_band_3_6_common", "offset_band_4_7_common",
        "uniform_delay_band",
        "ecg_only", "ppg_only", "early_concat", "late_average", "gated_fusion",
        "se_fusion", "delay_120_300", "delay_120_350", "delay_80_550",
    }
    return build_model(name) if name in mechanism else build_neural_baseline(name)


def main():
    parser = argparse.ArgumentParser(description="Train one exact subject-disjoint fold from prepared ECG/PPG tensors")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--model", default="physiocat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--configuration-manifest", type=Path, default=None, help="CSV containing one fixed configuration row per model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-pretraining", action="store_true", help="Do not run contrastive pretraining for this comparator")
    args = parser.parse_args()
    archive = load_prepared_npz(args.archive)
    roles = np.load(ROOT / "data/folds/fold_subject_roles.npz", allow_pickle=False)
    subjects = roles["subject_id"].astype(str)
    if not 1 <= args.fold_id <= roles["roles"].shape[0]:
        raise ValueError("fold-id out of range")
    row = roles["roles"][args.fold_id - 1]
    test_subjects = set(subjects[row == 2])
    validation_subjects = set(subjects[row == 1])
    train_idx, val_idx, test_idx = subject_partition(archive["subject_id"], test_subjects, validation_subjects)
    selected = {}
    if args.configuration_manifest is not None:
        table = pd.read_csv(args.configuration_manifest)
        row_selected = table[table.model.astype(str) == args.model]
        if len(row_selected) != 1:
            raise ValueError(f"Expected exactly one fixed configuration row; found {len(row_selected)}")
        selected = row_selected.iloc[0].to_dict()
    batch_size = int(args.batch_size if args.batch_size is not None else selected.get("batch_size", 32))
    learning_rate = float(args.learning_rate if args.learning_rate is not None else selected.get("learning_rate", 3e-4))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else selected.get("weight_decay", 1e-4))
    maximum_epochs = int(args.epochs if args.epochs is not None else selected.get("maximum_epochs", 80))
    pretrain_epochs = int(args.pretrain_epochs if args.pretrain_epochs is not None else selected.get("pretrain_epochs", 5))
    selected_seed = int(selected.get("initialization_seed", selected.get("seed", args.seed)))
    data_order_seed = int(selected.get("data_order_seed", selected_seed))
    mask_seed = int(selected.get("mask_seed", 82_000))
    checkpoint_rule = str(selected.get("checkpoint_rule", "minimum validation mean MAE; earliest epoch breaks ties"))
    input_view = input_view_for_model(args.model)
    declared_input_view = str(selected.get("input_view", input_view))
    if declared_input_view != input_view:
        raise ValueError(f"Configuration input view {declared_input_view!r} conflicts with model contract {input_view!r}")
    if selected:
        configuration_id = str(selected.get("configuration_id", "missing_configuration_id"))
        evaluation_scope = str(selected.get("evaluation_scope", "complete five-fold subject-grouped OOF prediction authority"))
        evaluation_subjects = int(selected.get("evaluation_subjects", len(subjects)))
        outer_folds = int(selected.get("outer_folds", len(subjects)))
        expected_sha = hashlib.sha256(
            f"{args.model}|{configuration_id}|{learning_rate:.8g}|{weight_decay:.8g}|{batch_size}|{pretrain_epochs}|{maximum_epochs}|{selected_seed}|{data_order_seed}|{mask_seed}|{checkpoint_rule}|{input_view}|{evaluation_scope}|{evaluation_subjects}|{outer_folds}".encode("utf-8")
        ).hexdigest()
        supplied_sha = str(selected.get("configuration_sha256", expected_sha))
        if supplied_sha != expected_sha:
            raise ValueError("Prespecified configuration SHA does not match its fields")
    else:
        configuration_id = "manual-cli-override"
    pretrain_enabled = not args.disable_pretraining and (args.model.startswith("attention_edge_ablation_seed_") or args.model in {
        "physiocat", "matched_no_delay", "ppg_leading_mirror", "unidirectional_delay",
        "direction_agnostic_local", "shifted_offsets_9_12", "attention_edge_ablation",
        "without_sqi_fusion", "without_delay_and_sqi",
        "physiocat_patch_local", "matched_no_delay_patch_local",
        "offset_band_2_5_common", "offset_band_3_6_common", "offset_band_4_7_common",
        "uniform_delay_band",
        "early_concat", "late_average", "gated_fusion", "se_fusion",
        "delay_120_300", "delay_120_350", "delay_80_550",
    })
    model, history = fit_model(lambda: model_for(args.model), WaveformDataset(archive, train_idx, input_view=input_view), WaveformDataset(archive, val_idx, input_view=input_view), args.output_dir, FitConfig(seed=selected_seed, data_order_seed=data_order_seed, maximum_epochs=maximum_epochs, pretrain_epochs=pretrain_epochs, batch_size=batch_size, learning_rate=learning_rate, weight_decay=weight_decay, pretrain_enabled=pretrain_enabled))
    supervised_history = [row for row in history if row["stage"] == "supervised"]
    selected_history = min(supervised_history, key=lambda row: (row["validation_objective"], row["epoch"]))
    inference_checkpoint = args.output_dir / "best_checkpoint.npz"
    save_inference_checkpoint(
        inference_checkpoint,
        {
            "model_slug": args.model,
            "configuration_id": configuration_id,
            "configuration_sha256": str(selected.get("configuration_sha256", expected_sha if selected else "manual_cli_override")),
            "outer_fold_id": int(args.fold_id),
            "test_subjects": len(test_subjects),
            "selected_epoch": int(selected_history["epoch"]),
            "selected_validation_mean_mae": float(selected_history["validation_objective"]),
            "input_view": input_view,
        },
        model.state_dict(),
    )
    import torch
    result = evaluate(model, torch.utils.data.DataLoader(WaveformDataset(archive, test_idx, input_view=input_view), batch_size=batch_size), next(model.parameters()).device)
    prediction_path = args.output_dir / "test_predictions.csv"
    pd.DataFrame({"window_id": result["window_id"], "subject_id": result["subject_id"], "reference_sbp": result["target"][:, 0], "reference_dbp": result["target"][:, 1], "predicted_sbp": result["prediction"][:, 0], "predicted_dbp": result["prediction"][:, 1]}).to_csv(prediction_path, index=False)
    checkpoint_path = inference_checkpoint
    lineage = {
        "status": "PASS",
        "fold_id": args.fold_id,
        "model": args.model,
        "input_view": input_view,
        "test_subjects": len(test_subjects),
        "configuration_id": configuration_id,
        "configuration_sha256": str(selected.get("configuration_sha256", expected_sha if selected else "manual_cli_override")),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "test_predictions_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
        "seed": selected_seed,
        "data_order_seed": data_order_seed,
        "maximum_epochs": maximum_epochs,
        "pretrain_epochs": pretrain_epochs if pretrain_enabled else 0,
        "epochs_completed": int(sum(row["stage"] == "supervised" for row in history)),
    }
    (args.output_dir / "fold_lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    print(json.dumps({**lineage, "train_windows": len(train_idx), "validation_windows": len(val_idx), "test_windows": len(test_idx), "learning_rate": learning_rate, "weight_decay": weight_decay, "batch_size": batch_size, "configuration_manifest": str(args.configuration_manifest) if args.configuration_manifest else None, "sbp_mae": result["sbp_mae"], "dbp_mae": result["dbp_mae"]}, indent=2))


if __name__ == "__main__":
    main()
