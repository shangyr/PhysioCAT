from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from physiocat.dataio import iter_hdf5_records
from physiocat.preprocessing import REQUIRED_RETENTION_FIELDS, PreprocessingConfig, preprocess_window


def membership_hash(window_ids) -> str:
    digest = hashlib.sha256()
    for value in sorted(map(str, window_ids)):
        digest.update(value.encode("utf-8")); digest.update(b"\n")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Prepare normalized 8-s ECG/PPG tensors from a legally obtained normalized HDF5 export")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate-hz", type=float, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--subject-cap", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    candidates, prepared_records = [], []
    for index, record in enumerate(iter_hdf5_records(args.input, sample_rate_hz=args.sample_rate_hz)):
        if args.limit is not None and index >= args.limit:
            break
        try:
            prepared = preprocess_window(
                record.ecg,
                record.ppg,
                record.sample_rate_hz,
                PreprocessingConfig(),
                abp=record.abp,
                age_years=record.age_years,
                released_sbp=record.sbp,
                released_dbp=record.dbp,
                label_source=record.label_source,
            )
            flags = prepared["retention_flags"]
            successful = True
        except (ValueError, FloatingPointError):
            prepared = None
            flags = {name: False for name in REQUIRED_RETENTION_FIELDS}
            successful = False
        window_id = record.window_id or f"{record.record_id}-W{index + 1:06d}"
        row = {"window_id": window_id, "subject_id": record.subject_id, "record_id": record.record_id, "age_years": record.age_years, "sex": record.sex, "preprocessing_success": successful, **flags}
        row["retained_before_subject_cap"] = bool(successful and all(flags.values()))
        candidates.append(row)
        prepared_records.append(prepared)
    if not candidates:
        raise RuntimeError("No valid windows were prepared")
    audit = pd.DataFrame(candidates)
    audit["subject_balance_cap_pass"] = False
    rng = np.random.default_rng(args.seed)
    for _, indices in audit[audit.retained_before_subject_cap].groupby("subject_id", sort=True).groups.items():
        indices = np.asarray(list(indices), dtype=int)
        selected = indices if args.subject_cap is None or len(indices) <= args.subject_cap else np.sort(rng.choice(indices, args.subject_cap, replace=False))
        audit.loc[selected, "subject_balance_cap_pass"] = True
    audit["retained"] = audit.retained_before_subject_cap & audit.subject_balance_cap_pass

    kept_indices = np.flatnonzero(audit.retained.to_numpy())
    if not len(kept_indices):
        raise RuntimeError("No windows passed the declared retention policy")
    rows, ecg_patch, ppg_patch, ecg_window, ppg_window, sqi, targets = [], [], [], [], [], [], []
    for source_index in kept_indices:
        prepared = prepared_records[source_index]
        rows.append({"window_id": audit.at[source_index, "window_id"], "subject_id": audit.at[source_index, "subject_id"], "record_id": audit.at[source_index, "record_id"], "row_index": len(ecg_patch), "label_source": prepared["label_source"]})
        ecg_patch.append(prepared["ecg_patch_local"]); ppg_patch.append(prepared["ppg_patch_local"])
        ecg_window.append(prepared["ecg_window_robust"]); ppg_window.append(prepared["ppg_window_robust"])
        sqi.append(prepared["sqi_tokens"]); targets.append([prepared["sbp"], prepared["dbp"]])
    args.output.mkdir(parents=True, exist_ok=True)
    ecg_patch = np.asarray(ecg_patch, np.float32); ppg_patch = np.asarray(ppg_patch, np.float32)
    np.savez_compressed(
        args.output / "prepared_windows.npz",
        ecg=ecg_patch,
        ppg=ppg_patch,
        ecg_patch_local=ecg_patch,
        ppg_patch_local=ppg_patch,
        ecg_window_robust=np.asarray(ecg_window, np.float32),
        ppg_window_robust=np.asarray(ppg_window, np.float32),
        sqi_tokens=np.asarray(sqi, np.float32),
        targets=np.asarray(targets, np.float32),
        subject_id=np.asarray([row["subject_id"] for row in rows]),
        window_id=np.asarray([row["window_id"] for row in rows]),
    )
    pd.DataFrame(rows).to_csv(args.output / "prepared_manifest.csv", index=False)
    audit.to_csv(args.output / "candidate_retention_audit.csv", index=False)
    active = np.ones(len(audit), dtype=bool)
    cascade = []
    for field in REQUIRED_RETENTION_FIELDS:
        active &= audit[field].astype(bool).to_numpy()
        cascade.append({"stage": field, "windows_remaining": int(active.sum()), "subjects_remaining": int(audit.loc[active, "subject_id"].nunique()), "membership_sha256": membership_hash(audit.loc[active, "window_id"])})
    active &= audit.subject_balance_cap_pass.astype(bool).to_numpy()
    cascade.append({"stage": "subject_balance_cap_pass", "windows_remaining": int(active.sum()), "subjects_remaining": int(audit.loc[active, "subject_id"].nunique()), "membership_sha256": membership_hash(audit.loc[active, "window_id"])})
    pd.DataFrame(cascade).to_csv(args.output / "retention_cascade.csv", index=False)
    print(json.dumps({"status": "PASS", "candidate_windows": len(audit), "retained_windows": len(rows), "subjects": len({row['subject_id'] for row in rows}), "retention_membership_sha256": membership_hash([row["window_id"] for row in rows])}, indent=2))


if __name__ == "__main__":
    main()
