from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import shutil


def main():
    parser = argparse.ArgumentParser(description="Collect submitted Figures 1--7 into a reviewer bundle")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced/figure_bundle")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for i in range(1, 8):
        source = ROOT / "paper/figures" / f"Figure_{i}.pdf"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output_dir / source.name)
        copied.append(source.name)
    print(json.dumps({"status": "PASS", "figures": copied}, indent=2))


if __name__ == "__main__":
    main()
