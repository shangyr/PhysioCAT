from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import yaml


def dotted(data, path):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(path)
        current = current[part]
    return current


def main():
    parser = argparse.ArgumentParser(description="Validate operational YAML configurations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema = json.loads((ROOT / "configs/config_schema.json").read_text(encoding="utf-8"))
    rows = []
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        kind = data.get("kind") if isinstance(data, dict) else None
        if kind not in schema["kinds"]:
            raise AssertionError(f"Unknown or missing kind in {path}")
        missing = []
        for field in schema["kinds"][kind]["required"]:
            try:
                dotted(data, field)
            except KeyError:
                missing.append(field)
        if missing:
            raise AssertionError(f"Missing fields in {path}: {missing}")
        rows.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"), "kind": kind, "required_fields": len(schema["kinds"][kind]["required"]), "status": "PASS"})
    import pandas as pd
    pd.DataFrame(rows).to_csv(args.output_dir / "config_validation.csv", index=False)
    report = {"status": "PASS", "yaml_files": len(rows), "kinds": sorted({r["kind"] for r in rows})}
    (args.output_dir / "config_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
