from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import yaml
from scipy import io as scipy_io
from scipy import signal

from .preprocessing import aligned_abp_scalar_targets


@dataclass(frozen=True)
class AdapterConfig:
    file_glob: str
    source_sample_rate_hz: float
    target_sample_rate_hz: float = 250.0
    window_seconds: float = 8.0
    trim_seconds_each_side: float = 0.0
    target_policy: str = "curated_scalar_same_record"
    ecg_keys: tuple[str, ...] = ("ecg", "ECG")
    ppg_keys: tuple[str, ...] = ("ppg", "PPG", "pleth")
    abp_keys: tuple[str, ...] = ("abp", "ABP", "art")
    sbp_keys: tuple[str, ...] = ("sbp", "SBP")
    dbp_keys: tuple[str, ...] = ("dbp", "DBP")
    subject_keys: tuple[str, ...] = ("subject_id", "Subject")
    record_keys: tuple[str, ...] = ("record_id", "Record")
    age_keys: tuple[str, ...] = ("age_years", "Age", "age")
    sex_keys: tuple[str, ...] = ("sex", "Sex", "gender")
    subject_regex: str | None = None
    record_regex: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "AdapterConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        fields = payload.get("fields", {})
        return cls(
            file_glob=str(payload["file_glob"]),
            source_sample_rate_hz=float(payload["source_sample_rate_hz"]),
            target_sample_rate_hz=float(payload.get("target_sample_rate_hz", 250.0)),
            window_seconds=float(payload.get("window_seconds", 8.0)),
            trim_seconds_each_side=float(payload.get("trim_seconds_each_side", 0.0)),
            target_policy=str(payload.get("target_policy", "curated_scalar_same_record")),
            ecg_keys=tuple(fields.get("ecg", ("ecg", "ECG"))),
            ppg_keys=tuple(fields.get("ppg", ("ppg", "PPG", "pleth"))),
            abp_keys=tuple(fields.get("abp", ("abp", "ABP", "art"))),
            sbp_keys=tuple(fields.get("sbp", ("sbp", "SBP"))),
            dbp_keys=tuple(fields.get("dbp", ("dbp", "DBP"))),
            subject_keys=tuple(fields.get("subject_id", ("subject_id", "Subject"))),
            record_keys=tuple(fields.get("record_id", ("record_id", "Record"))),
            age_keys=tuple(fields.get("age_years", ("age_years", "Age", "age"))),
            sex_keys=tuple(fields.get("sex", ("sex", "Sex", "gender"))),
            subject_regex=payload.get("subject_regex"),
            record_regex=payload.get("record_regex"),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identifier(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        raise ValueError("Empty subject or record identifier")
    return text


def _flatten_hdf5(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            data = value[()]
            if h5py.check_dtype(ref=value.dtype) is not None:
                resolved = []
                for reference in np.asarray(data).reshape(-1):
                    if not reference:
                        continue
                    resolved.append(np.asarray(group.file[reference][()]).squeeze())
                if not resolved:
                    raise ValueError(f"HDF5 reference dataset {key} contains no valid records")
                try:
                    data = np.stack(resolved)
                except ValueError as error:
                    raise ValueError(f"HDF5 referenced records have inconsistent shapes in {key}") from error
            output[key] = data
            output.setdefault(name, output[key])
        elif isinstance(value, h5py.Group):
            output.update(_flatten_hdf5(value, key))
    return output


def _coerce_cell_sequence(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype != object:
        return np.asarray(value)
    sequence = list(np.asarray(value, dtype=object).reshape(-1))
    arrays = [np.asarray(item).squeeze() for item in sequence]
    if not arrays:
        return np.asarray([])
    try:
        return np.stack(arrays)
    except ValueError:
        return np.asarray(arrays, dtype=object)


def _flatten_mat_mapping(value: object, prefix: str = "", output: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Flatten classic MATLAB structs/cells while retaining full field paths."""
    output = {} if output is None else output
    if isinstance(value, dict):
        for name, child in value.items():
            if str(name).startswith("__"):
                continue
            key = f"{prefix}/{name}" if prefix else str(name)
            _flatten_mat_mapping(child, key, output)
        return output
    if hasattr(value, "_fieldnames"):
        for name in value._fieldnames:
            key = f"{prefix}/{name}" if prefix else str(name)
            _flatten_mat_mapping(getattr(value, name), key, output)
        return output
    array = np.asarray(value)
    if array.dtype.names:
        for name in array.dtype.names:
            key = f"{prefix}/{name}" if prefix else str(name)
            _flatten_mat_mapping(array[name], key, output)
        return output
    data = _coerce_cell_sequence(array) if array.dtype == object else array
    if not prefix:
        raise ValueError("MAT payload leaf has no field path")
    output[prefix] = data
    output.setdefault(prefix.rsplit("/", 1)[-1], data)
    return output


def load_source_file(path: Path) -> dict[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    if suffix in {".h5", ".hdf5", ".mat"}:
        try:
            with h5py.File(path, "r") as handle:
                return _flatten_hdf5(handle)
        except OSError:
            if suffix != ".mat":
                raise
    if suffix == ".mat":
        try:
            payload = scipy_io.loadmat(path, simplify_cells=True)
        except TypeError:  # pragma: no cover - compatibility with old SciPy
            payload = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
        return _flatten_mat_mapping(payload)
    raise ValueError(f"Unsupported raw source format: {path}")


def _resolve_with_key(
    payload: dict[str, np.ndarray], aliases: Iterable[str], *, required: bool
) -> tuple[np.ndarray | None, str | None]:
    lower = {key.lower(): key for key in payload}
    for alias in aliases:
        if alias in payload:
            return np.asarray(payload[alias]), alias
        if alias.lower() in lower:
            selected = lower[alias.lower()]
            return np.asarray(payload[selected]), selected
    if required:
        raise KeyError(f"None of the required fields were found: {list(aliases)}")
    return None, None


def _resolve(payload: dict[str, np.ndarray], aliases: Iterable[str], *, required: bool) -> np.ndarray | None:
    value, _ = _resolve_with_key(payload, aliases, required=required)
    return value


def _as_rows(array: np.ndarray, expected_records: int | None = None) -> np.ndarray:
    array = np.asarray(array)
    array = np.squeeze(array)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Waveforms must be one- or two-dimensional after squeezing; observed {array.shape}")
    if expected_records is not None:
        if array.shape[0] == expected_records:
            return array
        if array.shape[1] == expected_records:
            return array.T
    # Physiological records normally have far more samples than records.
    return array if array.shape[1] >= array.shape[0] else array.T


def _identifier_vector(value: np.ndarray | None, n_records: int, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    flat = np.asarray(value).reshape(-1)
    if len(flat) == 1 and n_records > 1:
        flat = np.repeat(flat, n_records)
    if len(flat) != n_records:
        raise ValueError(f"Identifier vector has {len(flat)} rows for {n_records} waveform records")
    return [canonical_identifier(item) for item in flat]


def _regex_identifier(pattern: str | None, path: Path, fallback: str) -> str:
    if not pattern:
        return fallback
    match = re.search(pattern, path.as_posix())
    if not match:
        raise ValueError(f"Identifier regex did not match {path}: {pattern}")
    return canonical_identifier(match.group(1) if match.groups() else match.group(0))


def _resample(values: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if source_hz == target_hz:
        return np.asarray(values, dtype=np.float32)
    from fractions import Fraction

    ratio = Fraction(target_hz / source_hz).limit_denominator(10_000)
    return signal.resample_poly(np.asarray(values, dtype=np.float64), ratio.numerator, ratio.denominator).astype(np.float32)


def _center_crop(values: np.ndarray, samples: int, trim: int) -> tuple[np.ndarray, int, int]:
    if 2 * trim >= len(values):
        raise ValueError("Declared artifact trim removes the complete record")
    left, right = trim, len(values) - trim
    available = right - left
    if available < samples:
        raise ValueError(f"Record has {available} usable samples; {samples} required")
    start = left + (available - samples) // 2
    stop = start + samples
    return np.asarray(values[start:stop], dtype=np.float32), start, stop


def _abp_targets(ecg: np.ndarray, abp: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    """Use the single cardiac-boundary ABP target implementation."""
    sbp, dbp, _, _ = aligned_abp_scalar_targets(ecg, abp, int(round(sample_rate_hz)))
    return sbp, dbp


def _candidate_failure_code(error: Exception, *, paired_crop_pass: bool, target_policy: str) -> str:
    """Map the executable failure path to a stable candidate-audit category."""
    message = str(error)
    if not paired_crop_pass:
        if "Non-finite paired waveform crop" in message:
            return "nonfinite_paired_waveform_crop"
        if "different deterministic crop coordinates" in message:
            return "paired_channel_length_mismatch"
        return "insufficient_common_samples"
    if target_policy == "aligned_abp_crop":
        if "Too few valid ABP beats" in message:
            return "too_few_candidate_abp_beats"
        if "Too few finite ordered ABP beats" in message:
            return "too_few_finite_ordered_abp_beats"
        if "robust within-window rejection" in message:
            return "too_few_abp_beats_after_robust_rejection"
        if "one-dimensional and equal length" in message or "shorter than" in message:
            return "aligned_ecg_abp_length_mismatch"
        return "nonfinite_aligned_ecg_or_abp_crop"
    if isinstance(error, IndexError):
        return "record_target_pairing_failure"
    if "Invalid scalar BP target" in message:
        return "nonfinite_curated_scalar_target" if "nan" in message.lower() else "invalid_curated_scalar_target"
    return "record_target_pairing_failure"


def export_dataset(input_root: Path, output_hdf5: Path, config_path: Path, dataset_name: str) -> dict[str, object]:
    """Convert legally obtained source files and retain the complete candidate audit.

    The exporter never downloads or redistributes source data. Each source row
    is enumerated before crop/target formation; failures are written to a
    candidate-audit sidecar instead of disappearing from the cohort start.
    Every successful source row yields at most one deterministic centered crop.
    """

    config = AdapterConfig.from_yaml(config_path)
    files = sorted(input_root.glob(config.file_glob))
    if not files:
        raise FileNotFoundError(f"No source files matched {config.file_glob!r} under {input_root}")
    if config.target_policy not in {"aligned_abp_crop", "curated_scalar_same_record"}:
        raise ValueError(f"Unsupported target_policy: {config.target_policy}")
    target_samples = int(round(config.window_seconds * config.target_sample_rate_hz))
    trim_source = int(round(config.trim_seconds_each_side * config.source_sample_rate_hz))
    rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    ecg_rows: list[np.ndarray] = []
    ppg_rows: list[np.ndarray] = []
    abp_rows: list[np.ndarray] = []
    targets: list[tuple[float, float]] = []
    field_resolution: list[dict[str, object]] = []

    for source_path in files:
        payload = load_source_file(source_path)
        ecg_payload, ecg_field = _resolve_with_key(payload, config.ecg_keys, required=True)
        raw_ecg = _as_rows(ecg_payload)
        ppg_payload, ppg_field = _resolve_with_key(payload, config.ppg_keys, required=True)
        raw_ppg = _as_rows(ppg_payload, len(raw_ecg))
        raw_abp_payload, abp_field = _resolve_with_key(payload, config.abp_keys, required=False)
        raw_abp = None if raw_abp_payload is None else _as_rows(raw_abp_payload, len(raw_ecg))
        if len(raw_ecg) != len(raw_ppg) or (raw_abp is not None and len(raw_ecg) != len(raw_abp)):
            raise ValueError(f"Channel row counts differ in {source_path}")
        n_records = len(raw_ecg)
        supplied_subject, subject_field = _resolve_with_key(payload, config.subject_keys, required=False)
        supplied_record, record_field = _resolve_with_key(payload, config.record_keys, required=False)
        supplied_sbp, sbp_field = _resolve_with_key(payload, config.sbp_keys, required=False)
        supplied_dbp, dbp_field = _resolve_with_key(payload, config.dbp_keys, required=False)
        supplied_age, age_field = _resolve_with_key(payload, config.age_keys, required=False)
        supplied_sex, sex_field = _resolve_with_key(payload, config.sex_keys, required=False)
        if config.target_policy == "aligned_abp_crop" and raw_abp is None:
            raise KeyError(f"Aligned ABP is required to derive crop-level targets in {source_path}")
        if config.target_policy == "curated_scalar_same_record" and (supplied_sbp is None or supplied_dbp is None):
            raise KeyError(f"Curated scalar SBP/DBP labels are required in {source_path}")
        fallback_subject = _regex_identifier(config.subject_regex, source_path, source_path.stem) if supplied_subject is None else source_path.stem
        fallback_record = _regex_identifier(config.record_regex, source_path, source_path.stem) if supplied_record is None else source_path.stem
        subject = _identifier_vector(supplied_subject, n_records, [fallback_subject] * n_records)
        record = _identifier_vector(supplied_record, n_records, [f"{fallback_record}:{i:06d}" for i in range(n_records)])
        if supplied_sbp is not None:
            supplied_sbp = np.asarray(supplied_sbp).reshape(-1)
        if supplied_dbp is not None:
            supplied_dbp = np.asarray(supplied_dbp).reshape(-1)
        if supplied_age is not None:
            supplied_age = np.asarray(supplied_age).reshape(-1)
        sex_values = _identifier_vector(supplied_sex, n_records, ["Unknown"] * n_records) if supplied_sex is not None else ["Unknown"] * n_records
        source_digest = sha256_file(source_path)
        field_resolution.append(
            {
                "source_file": source_path.relative_to(input_root).as_posix(),
                "records": n_records,
                "ecg_field": ecg_field,
                "ppg_field": ppg_field,
                "abp_field": abp_field or "",
                "sbp_field": sbp_field or "",
                "dbp_field": dbp_field or "",
                "subject_field": subject_field or "path_regex",
                "record_field": record_field or "path_regex+row",
                "age_field": age_field or "",
                "sex_field": sex_field or "",
            }
        )
        for source_row in range(n_records):
            candidate_id = f"{dataset_name}:raw:{len(candidate_rows) + 1:08d}"
            candidate = {
                "candidate_id": candidate_id,
                "subject_id": subject[source_row],
                "record_id": record[source_row],
                "source_file": source_path.relative_to(input_root).as_posix(),
                "source_file_sha256": source_digest,
                "source_row": source_row,
                "ecg_field": ecg_field,
                "ppg_field": ppg_field,
                "abp_field": abp_field or "",
                "target_policy": config.target_policy,
                "crop_start_resampled": "",
                "crop_stop_resampled": "",
                "paired_waveform_crop_pass": False,
                "target_derivation_pass": False,
                "failure_stage": "",
                "label_source": "",
                "status": "",
                "failure_reason": "",
                "window_id": "",
            }
            try:
                e_resampled = _resample(raw_ecg[source_row], config.source_sample_rate_hz, config.target_sample_rate_hz)
                p_resampled = _resample(raw_ppg[source_row], config.source_sample_rate_hz, config.target_sample_rate_hz)
                a_resampled = None if raw_abp is None else _resample(raw_abp[source_row], config.source_sample_rate_hz, config.target_sample_rate_hz)
                trim_target = int(round(trim_source * config.target_sample_rate_hz / config.source_sample_rate_hz))
                e_crop, crop_start, crop_stop = _center_crop(e_resampled, target_samples, trim_target)
                p_crop, p_start, p_stop = _center_crop(p_resampled, target_samples, trim_target)
                if not np.isfinite(e_crop).all() or not np.isfinite(p_crop).all():
                    raise ValueError("Non-finite paired waveform crop")
                if a_resampled is None:
                    a_crop = np.full(target_samples, np.nan, dtype=np.float32)
                    a_start, a_stop = crop_start, crop_stop
                else:
                    a_crop, a_start, a_stop = _center_crop(a_resampled, target_samples, trim_target)
                if (crop_start, crop_stop) != (p_start, p_stop) or (raw_abp is not None and (crop_start, crop_stop) != (a_start, a_stop)):
                    raise ValueError("Aligned channels yielded different deterministic crop coordinates")
                candidate.update(
                    {
                        "crop_start_resampled": crop_start,
                        "crop_stop_resampled": crop_stop,
                        "paired_waveform_crop_pass": True,
                    }
                )
                if config.target_policy == "aligned_abp_crop":
                    sbp, dbp = _abp_targets(e_crop, a_crop, config.target_sample_rate_hz)
                    label_source = "aligned_8s_abp_beat_extrema"
                else:
                    sbp = float(supplied_sbp[source_row if len(supplied_sbp) > 1 else 0])
                    dbp = float(supplied_dbp[source_row if len(supplied_dbp) > 1 else 0])
                    label_source = "curated_scalar_same_record"
                if not (np.isfinite(sbp) and np.isfinite(dbp) and 0.0 < dbp < sbp):
                    raise ValueError(f"Invalid scalar BP target: {sbp}/{dbp}")
                window_id = f"{dataset_name}:{len(rows) + 1:08d}"
                row = {
                    "window_id": window_id,
                    "subject_id": subject[source_row],
                    "record_id": record[source_row],
                    "age_years": float(supplied_age[source_row if len(supplied_age) > 1 else 0]) if supplied_age is not None else np.nan,
                    "sex": sex_values[source_row],
                    "source_file": source_path.relative_to(input_root).as_posix(),
                    "source_file_sha256": source_digest,
                    "source_row": source_row,
                    "source_sample_rate_hz": config.source_sample_rate_hz,
                    "target_sample_rate_hz": config.target_sample_rate_hz,
                    "crop_start_resampled": crop_start,
                    "crop_stop_resampled": crop_stop,
                    "ecg_field": ecg_field,
                    "ppg_field": ppg_field,
                    "abp_field": abp_field or "",
                    "label_source": label_source,
                    "abp_available": raw_abp is not None,
                    "sbp": sbp,
                    "dbp": dbp,
                }
                ecg_rows.append(e_crop)
                ppg_rows.append(p_crop)
                abp_rows.append(a_crop)
                targets.append((sbp, dbp))
                rows.append(row)
                candidate.update(
                    {
                        "status": "accepted",
                        "target_derivation_pass": True,
                        "label_source": label_source,
                        "window_id": window_id,
                    }
                )
            except (ValueError, IndexError, FloatingPointError) as error:
                candidate.update(
                    {
                        "status": "rejected",
                        "failure_stage": "target_formation" if candidate["paired_waveform_crop_pass"] else "paired_crop",
                        "failure_reason": _candidate_failure_code(
                            error,
                            paired_crop_pass=bool(candidate["paired_waveform_crop_pass"]),
                            target_policy=config.target_policy,
                        ),
                    }
                )
            candidate_rows.append(candidate)

    if not rows:
        raise ValueError("No source rows produced a valid deterministic crop and scalar target")
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    strings = h5py.string_dtype("utf-8")
    with h5py.File(output_hdf5, "w") as handle:
        handle.attrs["schema"] = "physiocat-standard-v1"
        handle.attrs["dataset_name"] = dataset_name
        handle.attrs["adapter_config_sha256"] = sha256_file(config_path)
        handle.attrs["sample_rate_hz"] = config.target_sample_rate_hz
        handle.attrs["window_seconds"] = config.window_seconds
        handle.attrs["target_policy"] = config.target_policy
        handle.attrs["record_to_window_policy"] = "one source row; at most one deterministic centered crop"
        handle.create_dataset("ecg", data=np.asarray(ecg_rows, dtype=np.float32), compression="gzip", shuffle=True)
        handle.create_dataset("ppg", data=np.asarray(ppg_rows, dtype=np.float32), compression="gzip", shuffle=True)
        handle.create_dataset("abp", data=np.asarray(abp_rows, dtype=np.float32), compression="gzip", shuffle=True)
        handle.create_dataset("abp_available", data=np.asarray([row["abp_available"] for row in rows], dtype=np.bool_))
        handle.create_dataset("targets", data=np.asarray(targets, dtype=np.float32))
        for key in ("window_id", "subject_id", "record_id", "sex", "label_source", "ecg_field", "ppg_field", "abp_field"):
            handle.create_dataset(key, data=np.asarray([row[key] for row in rows], dtype=object), dtype=strings)
        handle.create_dataset("age_years", data=np.asarray([row["age_years"] for row in rows], dtype=np.float32))
    manifest_path = output_hdf5.with_suffix(".manifest.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidate_path = output_hdf5.with_suffix(".candidate_audit.csv")
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    source_inventory = {
        "dataset_name": dataset_name,
        "schema": "physiocat-standard-v1",
        "adapter_config": str(config_path),
        "adapter_config_sha256": sha256_file(config_path),
        "output_hdf5": str(output_hdf5),
        "output_hdf5_sha256": sha256_file(output_hdf5),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_audit": str(candidate_path),
        "candidate_audit_sha256": sha256_file(candidate_path),
        "source_files": [{"path": path.relative_to(input_root).as_posix(), "sha256": sha256_file(path)} for path in files],
        "field_resolution": field_resolution,
        "source_rows_enumerated": len(candidate_rows),
        "rejected_source_rows": int(sum(row["status"] == "rejected" for row in candidate_rows)),
        "windows": len(rows),
        "subjects": len({str(row["subject_id"]) for row in rows}),
    }
    inventory_path = output_hdf5.with_suffix(".lineage.json")
    inventory_path.write_text(json.dumps(source_inventory, indent=2), encoding="utf-8")
    return source_inventory
