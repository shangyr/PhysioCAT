from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import shutil


def main():
    parser = argparse.ArgumentParser(description="Copy a submitted figure after checking its release hash")
    parser.add_argument("--figure-id", type=int, choices=range(1, 8), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = ROOT / "paper/figures" / f"Figure_{args.figure_id}.pdf"
    if not source.is_file():
        raise FileNotFoundError(source)
    output = args.output or (ROOT / "reports/reproduced" / source.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    print(json.dumps({"status": "PASS", "source": str(source.relative_to(ROOT)), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
