from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import subprocess


def main():
    parser = argparse.ArgumentParser(description="Recompute all packaged numeric results without rendering figures")
    parser.parse_args()
    scripts = ["verify_fold_membership.py", "verify_training_lineage.py", "reproduce_main_tables.py", "reproduce_statistics.py", "verify_attention_export.py", "reproduce_attention.py", "reproduce_secondary_analyses.py", "reproduce_profiling.py", "reproduce_supplementary_tables.py", "verify_end_to_end_training.py", "verify_replays.py", "audit_cross_document_consistency.py"]
    for script in scripts:
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce" / script)], cwd=ROOT)
        if completed.returncode:
            raise SystemExit(completed.returncode)
    print(json.dumps({"status": "PASS", "scripts": scripts}, indent=2))


if __name__ == "__main__":
    main()
