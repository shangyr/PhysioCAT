from __future__ import annotations

import numpy as np


def token_aligned_circular_shift_ppg(
    ppg: np.ndarray,
    *,
    minimum_shift_tokens: int,
    maximum_shift_tokens: int,
    patch_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Circularly move complete PPG tokens without changing token contents.

    Every realized shift is an integer multiple of ``patch_samples``. The
    operation therefore changes ECG--PPG token alignment while preserving each
    patch's within-token samples, the complete sample multiset, signal energy,
    and Fourier-magnitude spectrum. Circular boundaries avoid zero-fill edges.
    """
    values = np.asarray(ppg)
    if values.ndim < 2:
        raise ValueError("ppg must have a window axis and a sample axis")
    if patch_samples <= 0 or values.shape[-1] % patch_samples:
        raise ValueError("patch_samples must exactly divide the sample axis")
    n_tokens = values.shape[-1] // patch_samples
    if not 1 <= minimum_shift_tokens <= maximum_shift_tokens < n_tokens:
        raise ValueError("token-shift bounds must be positive and shorter than one window")
    rng = np.random.default_rng(seed)
    magnitude = rng.integers(minimum_shift_tokens, maximum_shift_tokens + 1, size=len(values))
    direction = rng.choice(np.asarray([-1, 1], dtype=int), size=len(values))
    shifts = magnitude * direction * int(patch_samples)
    shifted = np.stack(
        [np.roll(values[row], int(shift), axis=-1) for row, shift in enumerate(shifts.tolist())],
        axis=0,
    )
    return shifted, shifts.astype(np.int32)


def cross_subject_ppg_pairing(
    ppg: np.ndarray,
    subject_ids: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair every ECG window with a deterministic PPG window from another subject."""
    values = np.asarray(ppg)
    subjects = np.asarray(subject_ids).astype(str)
    if len(values) != len(subjects):
        raise ValueError("ppg and subject_ids must have the same number of windows")
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError("cross-subject pairing requires at least two subjects")
    rng = np.random.default_rng(seed)
    source_subjects = unique[rng.permutation(len(unique))]
    donor_subjects = np.roll(source_subjects, 1)
    donor_for = dict(zip(source_subjects.tolist(), donor_subjects.tolist(), strict=True))
    donor_index = np.empty(len(values), dtype=np.int64)
    for subject in unique:
        rows = np.flatnonzero(subjects == subject)
        candidates = np.flatnonzero(subjects == donor_for[subject])
        donor_index[rows] = rng.choice(candidates, size=len(rows), replace=len(rows) > len(candidates))
    if np.any(subjects == subjects[donor_index]):
        raise AssertionError("cross-subject pairing produced a same-subject donor")
    return values[donor_index].copy(), donor_index


def reverse_ppg(ppg: np.ndarray) -> np.ndarray:
    """Reverse the sample order of each PPG window without modifying the input array."""
    return np.flip(np.asarray(ppg), axis=-1).copy()
