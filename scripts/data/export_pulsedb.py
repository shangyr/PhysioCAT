from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.adapters import export_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Export legally obtained PulseDB files to the PhysioCAT standard HDF5 schema")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/pulsedb_adapter.yaml")
    parser.add_argument("--dataset-name", choices=("pulsedb_vital", "pulsedb_mimic"), required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.input_root, args.output, args.config, args.dataset_name), indent=2))


if __name__ == "__main__":
    main()
