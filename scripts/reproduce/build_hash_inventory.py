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


def build_inventory(root: Path = ROOT):
    output = root / "artifacts/hashes/package_files.sha256"
    output.parent.mkdir(parents=True, exist_ok=True)
    excluded_parts = {"__pycache__", ".pytest_cache", ".git"}
    paths = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path == output or any(part in excluded_parts for part in relative.parts) or relative.parts[:1] == ("reports",):
            continue
        paths.append(path)
    lines = [f"{digest(path)}  {str(path.relative_to(root)).replace(chr(92), '/')}" for path in sorted(paths)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, len(lines)


def main():
    parser = argparse.ArgumentParser(description="Build deterministic SHA-256 inventory")
    parser.parse_args()
    output, count = build_inventory()
    print(json.dumps({"status": "PASS", "files_hashed": count, "inventory": str(output.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
