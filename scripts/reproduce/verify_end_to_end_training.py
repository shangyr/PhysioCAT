from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.dataio import load_prepared_npz
from physiocat.models import PhysioCAT, PhysioCATConfig
from physiocat.training import FitConfig, WaveformDataset, fit_model


def main():
    parser = argparse.ArgumentParser(description="Execute a compact contrastive + supervised training chain on the packaged waveform fixture")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced/end_to_end_training")
    args = parser.parse_args()
    archive = load_prepared_npz(ROOT / "data/fixtures/end_to_end_fixture.npz")
    subjects = np.unique(archive["subject_id"].astype(str))
    validation = np.isin(archive["subject_id"].astype(str), subjects[-2:])
    config = PhysioCATConfig(hidden=32, heads=4, ffn_multiplier=2, patch_mlp_multiplier=2, temporal_layers=1, dropout=0.0)
    fit_config = FitConfig(batch_size=4, pretrain_epochs=2, maximum_epochs=2, patience=2)
    _, history = fit_model(lambda: PhysioCAT(config), WaveformDataset(archive, np.flatnonzero(~validation)), WaveformDataset(archive, np.flatnonzero(validation)), args.output_dir, fit_config)
    contrastive = [row for row in history if row["stage"] == "contrastive"]
    supervised = [row for row in history if row["stage"] == "supervised"]
    expected_warmup = [fit_config.learning_rate * 0.2, fit_config.learning_rate * 0.4]
    if [row["optimizer_phase"] for row in contrastive] != ["contrastive", "contrastive"]:
        raise AssertionError("Contrastive optimizer phase was not recorded")
    if [row["optimizer_phase"] for row in supervised] != ["supervised", "supervised"]:
        raise AssertionError("Supervised optimizer phase was not recorded")
    if not np.allclose([row["learning_rate"] for row in contrastive], expected_warmup, atol=1e-12):
        raise AssertionError("Contrastive warm-up trace is incorrect")
    if not np.allclose([row["learning_rate"] for row in supervised], expected_warmup, atol=1e-12):
        raise AssertionError("Supervised warm-up did not restart from phase epoch one")
    checkpoint = args.output_dir / "best_checkpoint.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size < 100_000:
        raise AssertionError("End-to-end smoke checkpoint was not created")
    report = {"status": "PASS", "fixture_windows": len(archive["ecg"]), "subjects": len(subjects), "history_rows": len(history), "checkpoint_bytes": checkpoint.stat().st_size, "stages": sorted({row["stage"] for row in history}), "phase_local_warmup_restart": "PASS"}
    (args.output_dir.parent / "end_to_end_training_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
