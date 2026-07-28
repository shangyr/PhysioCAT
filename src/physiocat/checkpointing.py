from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch


FIXED_ZIP_TIME = (2026, 7, 21, 0, 0, 0)


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def save_inference_checkpoint(path: Path, metadata: dict[str, object], state_dict: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")), dtype="U8192")
    }
    for name, tensor in state_dict.items():
        arrays[f"state__{name}"] = tensor.detach().cpu().contiguous().numpy()
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info._compresslevel = 9
            archive.writestr(info, _npy_bytes(arrays[name]))


def load_inference_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"]))
    state_dict = {
        name[len("state__") :]: torch.from_numpy(np.asarray(archive[name]).copy())
        for name in archive.files
        if name.startswith("state__")
    }
    return metadata, state_dict
