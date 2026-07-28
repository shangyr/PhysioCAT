from __future__ import annotations

import argparse
import gzip
import hashlib
import hmac
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "cohort",
    "raw_candidate_id",
    "subject_id",
    "source_patient_hash",
    "source_record_hash",
    "source_file_sha256",
    "source_row",
    "ecg_field",
    "ppg_field",
    "abp_field",
    "label_source",
    "age_years",
    "sex",
    "repository_sbp",
    "repository_dbp",
    "repository_scalar_available",
    "input_min_sqi",
    "rr_irregularity",
    "crop_start_resampled",
    "crop_stop_resampled",
    "paired_waveform_crop_pass",
    "target_derivation_pass",
    "target_failure_reason",
    "status",
    "window_id",
    "record_to_window_policy",
]


def _digest(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _boolean(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "accepted"})


def canonicalize_candidate_audit(
    candidate_audit: pd.DataFrame,
    *,
    dataset_name: str,
    identity_namespace: str,
    hash_key: bytes,
) -> pd.DataFrame:
    """Convert an adapter candidate audit into the released lineage schema.

    Raw repository subject and record identifiers are replaced by keyed hashes.
    ``identity_namespace`` names the shared source ecosystem rather than the
    released cohort. Thus, two separately curated MIMIC-derived protocols can
    be compared in one identity space while retaining different public cohort
    labels and subject IDs. The key is retained by the data custodian and is
    never written to the reviewer package. Source-file hashes, row indices,
    resolved field aliases, crop coordinates, failure stages, and accepted
    window IDs remain auditable.
    """
    required = {
        "candidate_id", "subject_id", "record_id", "source_file_sha256", "source_row",
        "ecg_field", "ppg_field", "abp_field", "crop_start_resampled", "crop_stop_resampled",
        "paired_waveform_crop_pass", "target_derivation_pass", "failure_reason", "status", "window_id",
    }
    missing = sorted(required - set(candidate_audit.columns))
    if missing:
        raise KeyError(f"Candidate audit is missing required adapter fields: {missing}")
    if not hash_key:
        raise ValueError("A non-empty data-custodian hash key is required")
    identity_namespace = identity_namespace.strip().lower()
    if not identity_namespace:
        raise ValueError("A non-empty shared identity namespace is required")

    frame = candidate_audit.copy()
    subject_raw = frame.subject_id.astype(str)
    record_raw = frame.record_id.astype(str)
    subject_levels = {value: index + 1 for index, value in enumerate(sorted(subject_raw.unique()))}
    subject_id = subject_raw.map(lambda value: f"{dataset_name.upper()}-S{subject_levels[value]:05d}")
    patient_hash = subject_raw.map(lambda value: _digest(hash_key, f"{identity_namespace}|patient|{value}"))
    record_hash = [
        _digest(hash_key, f"{identity_namespace}|record|{patient}|{record}")
        for patient, record in zip(
            subject_raw,
            record_raw,
            strict=True,
        )
    ]
    paired_pass = _boolean(frame.paired_waveform_crop_pass)
    target_pass = _boolean(frame.target_derivation_pass)
    label_source = frame.get("label_source", pd.Series("", index=frame.index)).astype(str)
    if "target_policy" in frame:
        fallback = frame.target_policy.astype(str).map(
            {
                "aligned_abp_crop": "aligned_8s_abp_beat_extrema",
                "curated_scalar_same_record": "curated_scalar_same_record",
            }
        ).fillna("")
        label_source = label_source.mask(label_source.eq(""), fallback)
    failure = frame.failure_reason.fillna("").astype(str)
    failure = failure.str.replace(r"^[A-Za-z]+Error:\s*", "", regex=True)

    def optional_numeric(*names: str) -> pd.Series:
        for name in names:
            if name in frame:
                return pd.to_numeric(frame[name], errors="coerce")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    repository_sbp = optional_numeric("repository_sbp", "released_sbp")
    repository_dbp = optional_numeric("repository_dbp", "released_dbp")
    if "repository_scalar_available" in frame:
        repository_available = _boolean(frame.repository_scalar_available)
    else:
        repository_available = repository_sbp.notna() & repository_dbp.notna()
    input_min_sqi = optional_numeric("input_min_sqi")
    if input_min_sqi.isna().all() and {"ecg_sqi", "ppg_sqi"}.issubset(frame.columns):
        input_min_sqi = pd.concat([optional_numeric("ecg_sqi"), optional_numeric("ppg_sqi")], axis=1).min(axis=1)

    output = pd.DataFrame(
        {
            "cohort": dataset_name,
            "raw_candidate_id": frame.candidate_id.astype(str),
            "subject_id": subject_id,
            "source_patient_hash": patient_hash,
            "source_record_hash": record_hash,
            "source_file_sha256": frame.source_file_sha256.astype(str),
            "source_row": frame.source_row.astype(int),
            "ecg_field": frame.ecg_field.fillna("").astype(str),
            "ppg_field": frame.ppg_field.fillna("").astype(str),
            "abp_field": frame.abp_field.fillna("").astype(str),
            "label_source": label_source,
            "age_years": optional_numeric("age_years"),
            "sex": frame.get("sex", pd.Series("", index=frame.index)).fillna("").astype(str),
            "repository_sbp": repository_sbp,
            "repository_dbp": repository_dbp,
            "repository_scalar_available": repository_available,
            "input_min_sqi": input_min_sqi,
            "rr_irregularity": optional_numeric("rr_irregularity"),
            "crop_start_resampled": pd.to_numeric(frame.crop_start_resampled, errors="coerce").fillna(-1).astype(int),
            "crop_stop_resampled": pd.to_numeric(frame.crop_stop_resampled, errors="coerce").fillna(-1).astype(int),
            "paired_waveform_crop_pass": paired_pass,
            "target_derivation_pass": target_pass,
            "target_failure_reason": np.where(target_pass, "", failure),
            "status": np.where(target_pass, "accepted", "rejected"),
            "window_id": frame.window_id.fillna("").astype(str),
            "record_to_window_policy": "one source-listed segment; at most one deterministic centered 8 s crop",
        }
    )
    if output.source_record_hash.duplicated().any():
        raise AssertionError("Canonical source-record hashes are not unique")
    if output.loc[output.target_derivation_pass, "window_id"].eq("").any():
        raise AssertionError("An accepted target-formed row lacks window_id")
    if (output.target_derivation_pass & ~output.paired_waveform_crop_pass).any():
        raise AssertionError("A target cannot form before a paired crop succeeds")
    return output[OUTPUT_COLUMNS]


def write_deterministic_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".gz":
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, lineterminator="\n")
    else:
        frame.to_csv(output, index=False, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize an adapter candidate audit into the released raw-candidate lineage schema")
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--identity-namespace",
        required=True,
        help="Shared source identity space, e.g. mimic-iii for every MIMIC-derived protocol",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-key-env", default="PHYSIOCAT_LINEAGE_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.hash_key_env, "").encode("utf-8")
    if not key:
        raise RuntimeError(f"Set {args.hash_key_env} to a private data-custodian lineage key")
    audit = pd.read_csv(args.candidate_audit, keep_default_na=False)
    output = canonicalize_candidate_audit(
        audit,
        dataset_name=args.dataset_name,
        identity_namespace=args.identity_namespace,
        hash_key=key,
    )
    write_deterministic_csv(output, args.output)
    print(f"Wrote {len(output):,} canonical candidate rows to {args.output}")


if __name__ == "__main__":
    main()
