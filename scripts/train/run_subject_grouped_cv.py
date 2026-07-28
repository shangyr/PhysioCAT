from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or execute the released five-fold subject-grouped trainer with disjoint validation")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["physiocat", "matched_no_delay"])
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-stop", type=int, default=5)
    parser.add_argument("--configuration-manifest", type=Path, default=ROOT / "artifacts/logs/training/model_configuration_registry.csv")
    parser.add_argument("--execute", action="store_true", help="Execute sequentially; default is a JSON dry-run plan")
    args = parser.parse_args()
    if not 1 <= args.fold_start <= args.fold_stop <= 5:
        raise ValueError("fold range must satisfy 1 <= start <= stop <= 5")
    commands = []
    trainer = ROOT / "scripts/train/train_fold.py"
    for model in args.models:
        for fold_id in range(args.fold_start, args.fold_stop + 1):
            output = args.output_root / model / f"fold_{fold_id:06d}"
            commands.append(
                [
                    sys.executable,
                    str(trainer),
                    "--archive",
                    str(args.archive),
                    "--fold-id",
                    str(fold_id),
                    "--model",
                    model,
                    "--output-dir",
                    str(output),
                    "--configuration-manifest",
                    str(args.configuration_manifest),
                ]
            )
    if args.execute:
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
    print(json.dumps({"status": "PASS", "mode": "execute" if args.execute else "dry-run", "jobs": len(commands), "first_command": commands[0], "last_command": commands[-1]}, indent=2))


if __name__ == "__main__":
    main()
