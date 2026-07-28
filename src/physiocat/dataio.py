from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawWindow:
    window_id: str
    subject_id: str
    record_id: str
    sample_rate_hz: float
    ecg: np.ndarray
    ppg: np.ndarray
    abp: np.ndarray | None
    age_years: float | None = None
    sex: str | None = None
    sbp: float | None = None
    dbp: float | None = None
    label_source: str | None = None


CHANNEL_ALIASES = {
    "ecg": ("ecg", "ECG", "II", "lead_ii"),
    "ppg": ("ppg", "PPG", "pleth", "PLETH"),
    "abp": ("abp", "ABP", "art", "ART"),
}


def _find_dataset(group: h5py.Group, aliases: tuple[str, ...], *, required: bool = True):
    for name in aliases:
        if name in group:
            return np.asarray(group[name])
    if required:
        raise KeyError(f"None of {aliases} found in {group.name}")
    return None


def iter_hdf5_records(path: Path, *, sample_rate_hz: float, subject_column: str = "subject_id"):
    """Read a normalized HDF5 export made from a legally obtained source.

    ECG, PPG and scalar BP targets are required.  The standard export records
    whether those targets were derived from the aligned ABP crop or supplied by
    a curated record-level label contract.  ABP is never a model input.
    """
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) == "physiocat-standard-v1":
            required = {"ecg", "ppg", "targets", "window_id", "subject_id", "record_id"}
            missing = required - set(handle.keys())
            if missing:
                raise KeyError(f"Standard HDF5 export missing {sorted(missing)}")
            n = int(handle["ecg"].shape[0])
            if not (handle["ppg"].shape[0] == handle["targets"].shape[0] == n):
                raise ValueError("Standard HDF5 channel row counts differ")
            if "abp" in handle and handle["abp"].shape[0] != n:
                raise ValueError("Optional ABP row count differs")
            if handle["targets"].ndim != 2 or handle["targets"].shape[1] != 2:
                raise ValueError("Standard HDF5 targets must have shape [windows,2]")
            fs = float(handle.attrs.get("sample_rate_hz", sample_rate_hz))

            def decode(dataset, index):
                value = dataset[index]
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)

            for index in range(n):
                age = float(handle["age_years"][index]) if "age_years" in handle else np.nan
                sex = decode(handle["sex"], index) if "sex" in handle else None
                target = np.asarray(handle["targets"][index], dtype=float)
                label_source = decode(handle["label_source"], index) if "label_source" in handle else str(handle.attrs.get("label_source", "scalar_bp_target"))
                abp_available = bool(handle["abp_available"][index]) if "abp_available" in handle else "abp" in handle
                yield RawWindow(
                    window_id=decode(handle["window_id"], index),
                    subject_id=decode(handle["subject_id"], index),
                    record_id=decode(handle["record_id"], index),
                    sample_rate_hz=fs,
                    ecg=np.asarray(handle["ecg"][index]),
                    ppg=np.asarray(handle["ppg"][index]),
                    abp=np.asarray(handle["abp"][index]) if abp_available and "abp" in handle else None,
                    age_years=None if not np.isfinite(age) else age,
                    sex=None if sex in {None, "", "Unknown", "nan"} else sex,
                    sbp=float(target[0]),
                    dbp=float(target[1]),
                    label_source=label_source,
                )
            return
        for record_id, group in handle.items():
            if not isinstance(group, h5py.Group):
                continue
            subject_id = str(group.attrs.get(subject_column, record_id))
            fs = float(group.attrs.get("sample_rate_hz", sample_rate_hz))
            age = group.attrs.get("age_years")
            sex = group.attrs.get("sex")
            yield RawWindow(
                str(group.attrs.get("window_id", record_id)),
                subject_id,
                record_id,
                fs,
                _find_dataset(group, CHANNEL_ALIASES["ecg"]),
                _find_dataset(group, CHANNEL_ALIASES["ppg"]),
                _find_dataset(group, CHANNEL_ALIASES["abp"], required=False),
                float(age) if age is not None else None,
                str(sex) if sex is not None else None,
                float(group.attrs["sbp"]) if "sbp" in group.attrs else None,
                float(group.attrs["dbp"]) if "dbp" in group.attrs else None,
                str(group.attrs.get("label_source", "scalar_bp_target")),
            )


def load_prepared_npz(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    required = {"ecg", "ppg", "sqi_tokens", "targets", "subject_id", "window_id"}
    missing = required - set(archive.files)
    if missing:
        raise KeyError(f"Prepared archive missing {sorted(missing)}")
    return {name: archive[name] for name in archive.files}


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"window_id", "subject_id", "prepared_archive", "row_index"}
    if not required.issubset(frame.columns):
        raise KeyError(f"Manifest missing {sorted(required - set(frame.columns))}")
    return frame
