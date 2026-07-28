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
    parser = argparse.ArgumentParser(description="Run the packaged preprocessing-output -> contrastive -> supervised -> checkpoint smoke chain")
    parser.add_argument("--fixture", type=Path, default=ROOT / "data/fixtures/end_to_end_fixture.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced/end_to_end_smoke")
    args = parser.parse_args()
    archive = load_prepared_npz(args.fixture)
    subjects = np.unique(archive["subject_id"].astype(str))
    validation_subjects = set(subjects[-2:])
    validation = np.isin(archive["subject_id"].astype(str), list(validation_subjects))
    train = ~validation
    smoke_config = PhysioCATConfig(hidden=32, heads=4, ffn_multiplier=2, patch_mlp_multiplier=2, temporal_layers=1, dropout=0.0)
    _, history = fit_model(lambda: PhysioCAT(smoke_config), WaveformDataset(archive, np.flatnonzero(train)), WaveformDataset(archive, np.flatnonzero(validation)), args.output_dir, FitConfig(batch_size=4, pretrain_epochs=1, maximum_epochs=1, patience=1))
    print(json.dumps({"status": "PASS", "fixture_windows": len(train) + int(validation.sum()), "training_windows": int(train.sum()), "validation_windows": int(validation.sum()), "history_rows": len(history), "checkpoint": str((args.output_dir / 'best_checkpoint.pt').relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
