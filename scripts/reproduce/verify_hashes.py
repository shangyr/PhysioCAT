from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import hashlib


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Verify release SHA-256 inventory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = ROOT / "artifacts/hashes/package_files.sha256"
    checked = 0
    for line in inventory.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        actual = digest(path)
        if actual != expected:
            raise AssertionError(f"Hash mismatch: {relative}")
        checked += 1
    report = {"status": "PASS", "files_verified": checked}
    (args.output_dir / "hash_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
