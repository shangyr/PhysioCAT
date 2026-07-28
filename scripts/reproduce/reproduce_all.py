from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import subprocess
import importlib.util


COMMANDS = [
    "validate_configs.py", "verify_fold_membership.py", "reproduce_main_tables.py", "reproduce_statistics.py", "verify_attention_export.py", "reproduce_attention.py", "reproduce_secondary_analyses.py", "reproduce_profiling.py", "reproduce_supplementary_tables.py",
    "verify_training_lineage.py", "verify_end_to_end_training.py", "verify_replays.py", "reproduce_figures.py", "build_figure_bundle.py", "audit_cross_document_consistency.py", "verify_hashes.py", "audit_release.py",
]

REQUIRED_IMPORTS = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "matplotlib": "matplotlib", "yaml": "PyYAML", "pywt": "PyWavelets",
    "h5py": "h5py", "torch": "torch", "sklearn": "scikit-learn",
    "fitz": "PyMuPDF",
}


def verify_dependencies():
    missing = [distribution for module, distribution in REQUIRED_IMPORTS.items() if importlib.util.find_spec(module) is None]
    if missing:
        raise SystemExit(
            "Missing reproduction dependencies: " + ", ".join(missing)
            + ". Install the locked environment first with: "
            + f"{sys.executable} -m pip install -r requirements/requirements-lock.txt"
        )


def main():
    parser = argparse.ArgumentParser(description="Run all reviewer-facing reproduction and audit commands")
    parser.parse_args()
    verify_dependencies()
    results = []
    for name in COMMANDS:
        command = [sys.executable, str(ROOT / "scripts/reproduce" / name)]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        results.append({"script": name, "returncode": completed.returncode})
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise SystemExit(completed.returncode)
    output = ROOT / "reports/reproduced/all_commands.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "PASS", "commands": results}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "commands_completed": len(results)}, indent=2))


if __name__ == "__main__":
    main()
