from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt
from scipy import signal


REQUIRED_RETENTION_FIELDS = (
    "adult_metadata", "scalar_target_valid", "beat_count_pass", "ecg_quality_pass",
    "ppg_quality_pass", "paired_sample_continuity", "sqi_rule_pass",
)


@dataclass(frozen=True)
class PreprocessingConfig:
    sample_rate_hz: int = 250
    window_seconds: int = 8
    ecg_band_hz: tuple[float, float] = (0.5, 40.0)
    ppg_band_hz: tuple[float, float] = (0.4, 12.0)
    abp_band_hz: tuple[float, float] = (0.4, 20.0)
    filter_order: int = 4
    minimum_cycles: int = 6
    token_count: int = 125
    percentile_clip: tuple[float, float] = (0.5, 99.5)
    ecg_quality_min: float = 0.40
    ppg_quality_min: float = 0.40
    # ABP quality is descriptive only and never participates in retention.
    abp_quality_min: float = 0.42


def retention_decision(record) -> bool:
    required = list(REQUIRED_RETENTION_FIELDS)
    if "subject_balance_cap_pass" in record:
        required.append("subject_balance_cap_pass")
    return all(bool(record[name]) for name in required)


def audit_pat(delay_ms: float) -> dict[str, float | bool]:
    value = float(delay_ms)
    detected = np.isfinite(value) and 60 <= value <= 650
    return {"pat_detected": bool(detected), "pat_ms": value if detected else np.nan, "pat_in_model_band": bool(detected and 120 <= value <= 450)}


def resample_signal(values: np.ndarray, source_hz: float, target_hz: float = 250) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if source_hz == target_hz:
        return values.copy()
    gcd = np.gcd(int(round(source_hz)), int(round(target_hz)))
    return signal.resample_poly(values, int(round(target_hz)) // gcd, int(round(source_hz)) // gcd)


def bandpass(values: np.ndarray, fs: float, band_hz: tuple[float, float], order: int = 4) -> np.ndarray:
    sos = signal.butter(order, band_hz, btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(values, dtype=np.float64))


def wavelet_denoise(values: np.ndarray, wavelet: str = "sym4", level: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coeffs = pywt.wavedec(values, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    threshold = sigma * np.sqrt(2 * np.log(max(len(values), 2)))
    denoised = [coeffs[0]] + [pywt.threshold(coef, threshold, mode="soft") for coef in coeffs[1:]]
    return pywt.waverec(denoised, wavelet, mode="symmetric")[: len(values)]


def centered_crop(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) < length:
        raise ValueError(f"Signal shorter than required centered crop: {len(values)} < {length}")
    start = (len(values) - length) // 2
    return values[start:start + length]


def robust_normalize(values: np.ndarray, epsilon: float = 1e-6, clip_percentiles: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lower, upper = np.percentile(values, clip_percentiles)
    values = np.clip(values, lower, upper)
    median = np.median(values)
    scale = np.subtract(*np.percentile(values, [75, 25]))
    if not np.isfinite(scale) or scale < epsilon:
        scale = np.std(values)
    return np.clip((values - median) / max(scale, epsilon), -8.0, 8.0).astype(np.float32)


def patch_local_normalize(
    values: np.ndarray,
    patch_samples: int = 16,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize each non-overlapping model patch without cross-patch access.

    The transform is intentionally separate from the filtered analysis branch
    used for beat detection and SQI.  Every output sample depends only on the
    16-sample analysis-grid patch that will form its model token.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if patch_samples <= 0 or len(values) % patch_samples:
        raise ValueError("Signal length must be divisible by patch_samples")
    patches = values.reshape(-1, patch_samples)
    centers = np.median(patches, axis=1, keepdims=True)
    q75 = np.percentile(patches, 75, axis=1, keepdims=True)
    q25 = np.percentile(patches, 25, axis=1, keepdims=True)
    scales = q75 - q25
    fallback = np.std(patches, axis=1, keepdims=True)
    scales = np.where(np.isfinite(scales) & (scales >= epsilon), scales, fallback)
    scales = np.maximum(scales, epsilon)
    normalized = np.clip((patches - centers) / scales, -8.0, 8.0)
    return normalized.reshape(values.shape).astype(np.float32)


def model_input_views(values: np.ndarray, patch_samples: int = 16) -> dict[str, np.ndarray]:
    """Return the two fixed deterministic waveform representations.

    Main models and comparators use ordinary robust within-window
    normalization so that cross-patch morphology is retained. The strictly
    patch-local representation is kept only for the normalization sensitivity
    control. Neither representation fits subject- or cohort-level statistics.
    """
    values = np.asarray(values, dtype=np.float64)
    return {
        "patch_local": patch_local_normalize(values, patch_samples=patch_samples),
        "window_robust": robust_normalize(values),
    }


def detect_r_peaks(ecg: np.ndarray, fs: int = 250) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=np.float64)
    derivative = np.diff(ecg, prepend=ecg[0])
    energy = signal.savgol_filter(derivative ** 2, 21, 3, mode="interp")
    threshold = np.median(energy) + 2.2 * np.median(np.abs(energy - np.median(energy)))
    peaks, _ = signal.find_peaks(energy, height=threshold, distance=int(0.28 * fs), prominence=max(np.std(energy) * 0.25, 1e-9))
    refined = []
    radius = int(0.05 * fs)
    for peak in peaks:
        lo, hi = max(0, peak - radius), min(len(ecg), peak + radius + 1)
        refined.append(lo + int(np.argmax(ecg[lo:hi])))
    return np.unique(refined).astype(int)


def detect_ppg_onsets(ppg: np.ndarray, fs: int = 250) -> np.ndarray:
    ppg = np.asarray(ppg, dtype=np.float64)
    troughs, _ = signal.find_peaks(-ppg, distance=int(0.28 * fs), prominence=max(np.std(ppg) * 0.08, 1e-6))
    derivative = np.gradient(ppg)
    onsets = []
    lookahead = int(0.24 * fs)
    for trough in troughs:
        hi = min(len(ppg), trough + lookahead)
        if hi <= trough + 2:
            continue
        local = derivative[trough:hi]
        sustained = np.flatnonzero(signal.convolve((local > 0).astype(int), np.ones(3, dtype=int), mode="same") >= 3)
        if len(sustained):
            onsets.append(trough + int(sustained[0]))
    return np.asarray(onsets, dtype=int)


def matched_pat_ms(r_peaks: np.ndarray, ppg_onsets: np.ndarray, fs: int = 250) -> float:
    """Return the median R-peak-to-PPG-onset interval in milliseconds.

    This is a descriptive apparent PAT estimate.  It is deliberately kept
    separate from paired-stream sample-continuity checks; the PAT value never
    participates in retention.
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    ppg_onsets = np.asarray(ppg_onsets, dtype=int)
    if len(r_peaks) == 0 or len(ppg_onsets) == 0:
        return float("nan")
    intervals = []
    lower = int(round(0.060 * fs))
    upper = int(round(0.650 * fs))
    for r_peak in r_peaks:
        candidates = ppg_onsets[(ppg_onsets >= r_peak + lower) & (ppg_onsets <= r_peak + upper)]
        if len(candidates):
            intervals.append(int(candidates[0] - r_peak))
    if not intervals:
        return float("nan")
    return float(np.median(intervals) * 1000.0 / fs)


def abp_beat_labels(abp: np.ndarray, r_peaks: np.ndarray, fs: int = 250) -> tuple[float, float, np.ndarray, np.ndarray]:
    maxima, minima = [], []
    for left, right in zip(r_peaks[:-1], r_peaks[1:], strict=False):
        if right - left < int(0.28 * fs):
            continue
        beat = abp[left:right]
        maxima.append(float(np.max(beat)))
        minima.append(float(np.min(beat)))
    maxima = np.asarray(maxima, dtype=float)
    minima = np.asarray(minima, dtype=float)
    if len(maxima) < 3:
        raise ValueError("Too few valid ABP beats")
    finite_ordered = np.isfinite(maxima) & np.isfinite(minima) & (minima > 0) & (minima < maxima)
    maxima, minima = maxima[finite_ordered], minima[finite_ordered]
    if len(maxima) < 3:
        raise ValueError("Too few finite ordered ABP beats")
    midpoint = 0.5 * (maxima + minima)
    pulse = maxima - minima
    robust = np.ones(len(maxima), dtype=bool)
    for values in (midpoint, pulse):
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        if mad > 1e-8:
            robust &= np.abs(values - center) <= 3.5 * 1.4826 * mad
    maxima, minima = maxima[robust], minima[robust]
    if len(maxima) < 3:
        raise ValueError("Too few ABP beats after robust within-window rejection")
    return float(np.mean(maxima)), float(np.mean(minima)), maxima, minima


def aligned_abp_scalar_targets(
    ecg: np.ndarray,
    abp: np.ndarray,
    fs: int = 250,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Derive the PulseDB scalar target through the single released beat path.

    ECG is transformed only for cardiac-boundary detection.  ABP extrema are
    then taken inside consecutive detected cardiac boundaries on the aligned
    raw ABP crop and passed through :func:`abp_beat_labels`.  Dataset export
    and downstream label auditing therefore share exactly one implementation.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    abp = np.asarray(abp, dtype=np.float64)
    if ecg.ndim != 1 or abp.ndim != 1 or len(ecg) != len(abp):
        raise ValueError("Aligned ECG and ABP crops must be one-dimensional and equal length")
    if not np.isfinite(ecg).all() or not np.isfinite(abp).all():
        raise ValueError("Aligned ECG and ABP crops must be finite")
    ecg_analysis = wavelet_denoise(bandpass(ecg, fs, (0.5, 40.0), order=4))
    r_peaks = detect_r_peaks(ecg_analysis, fs)
    return abp_beat_labels(abp, r_peaks, fs)


def ecg_sqi(ecg: np.ndarray, r_peaks: np.ndarray, fs: int = 250) -> float:
    ecg = np.asarray(ecg, dtype=float)
    centered = ecg - np.median(ecg)
    total = float(np.sum(centered ** 2)) + 1e-8
    mask = np.zeros(len(ecg), dtype=bool)
    radius = int(0.05 * fs)
    for peak in r_peaks:
        mask[max(0, peak - radius): min(len(ecg), peak + radius + 1)] = True
    qrs_energy = float(np.sum(centered[mask] ** 2)) / total
    kurtosis = float(np.mean(((centered - centered.mean()) / (centered.std() + 1e-8)) ** 4) - 3)
    return float(np.clip(0.5 * np.clip((kurtosis + 3.0) / 20.0, 0, 1) + 0.5 * qrs_energy, 0, 1))


def ppg_sqi(ppg: np.ndarray, fs: int = 250) -> float:
    ppg = np.asarray(ppg, dtype=float)
    epsilon = 1e-6
    centered = ppg - np.median(ppg)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequency = np.fft.rfftfreq(len(centered), d=1.0 / float(fs))
    pulse_band = (frequency >= 0.5) & (frequency <= 4.0)
    broad_band = (frequency >= 0.3) & (frequency <= 8.0)
    spectral_concentration = float(spectrum[pulse_band].sum()) / (float(spectrum[broad_band].sum()) + epsilon)
    derivative = np.gradient(ppg)
    positive = max(float(np.percentile(derivative, 99)), epsilon)
    negative = max(abs(float(np.percentile(derivative, 1))), epsilon)
    slope_balance = np.clip(min(positive / negative, negative / positive), 0, 1)
    return float(np.clip(0.6 * spectral_concentration + 0.4 * slope_balance, 0, 1))


def patch_center_positions(n_samples: int, n_tokens: int) -> np.ndarray:
    """Return the sample-coordinate centers of equal non-overlapping patches."""
    if n_samples <= 0 or n_tokens <= 0 or n_samples % n_tokens:
        raise ValueError("n_samples must be positive and divisible by n_tokens")
    patch_width = n_samples / n_tokens
    return (np.arange(n_tokens, dtype=float) + 0.5) * patch_width - 0.5


def token_sqi(values: np.ndarray, modality: str, fs: int = 250, n_tokens: int = 125) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    window = int(fs)
    if len(values) < window:
        raise ValueError("Token SQI requires at least one second of signal")
    starts = np.arange(0, len(values) - window + 1, dtype=int)  # 4 ms at 250 Hz
    scores = np.empty(len(starts), dtype=np.float32)
    global_peaks = detect_r_peaks(values, fs) if modality == "ecg" else None
    for index, start in enumerate(starts):
        segment = values[start:start + window]
        if modality == "ecg":
            peaks = global_peaks[(global_peaks >= start) & (global_peaks < start + window)] - start
            score = ecg_sqi(segment, peaks, fs) if len(peaks) else 0.0
        elif modality == "ppg":
            score = ppg_sqi(segment, fs)
        else:
            raise KeyError(modality)
        scores[index] = score
    score_centers = starts + (window - 1) / 2
    token_centers = patch_center_positions(len(values), n_tokens)
    return np.interp(token_centers, score_centers, scores, left=scores[0], right=scores[-1]).astype(np.float32)


def abp_sqi(abp: np.ndarray, maxima: np.ndarray, minima: np.ndarray) -> float:
    abp = np.asarray(abp, dtype=float)
    if len(maxima) < 3 or len(minima) < 3 or not np.isfinite(abp).all():
        return 0.0
    pulse = maxima - minima
    pulse_cv = np.std(pulse) / max(np.mean(pulse), 1e-6)
    jump = np.percentile(np.abs(np.diff(abp)), 99) / max(np.percentile(np.abs(abp - np.median(abp)), 75), 1e-6)
    return float(np.clip(0.65 * np.exp(-2.5 * pulse_cv) + 0.35 * np.exp(-0.08 * jump), 0, 1))


def synchronization_stability(
    ecg: np.ndarray,
    ppg: np.ndarray,
    fs: int = 250,
    subwindows: int = 4,
) -> tuple[bool, float, np.ndarray]:
    """Validate synchronized array shape and finite sample continuity.

    Physiological waveform similarity and apparent delay are deliberately not
    used as inclusion criteria. Candidate records are already simultaneous;
    this guard checks only a common one-dimensional finite sample support.
    """
    ecg = np.asarray(ecg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)
    if ecg.shape != ppg.shape or ecg.ndim != 1:
        raise ValueError("ecg and ppg must be one-dimensional arrays of equal length")
    stable = bool(len(ecg) > 0 and np.isfinite(ecg).all() and np.isfinite(ppg).all())
    return stable, 0.0 if stable else float("inf"), np.asarray([], dtype=float)


def preprocess_window(
    ecg,
    ppg,
    source_hz: float,
    config: PreprocessingConfig | None = None,
    *,
    abp=None,
    age_years: float | None = None,
    released_sbp: float | None = None,
    released_dbp: float | None = None,
    label_source: str | None = None,
) -> dict:
    c = config or PreprocessingConfig()
    target_valid = (
        released_sbp is not None and released_dbp is not None
        and np.isfinite(released_sbp) and np.isfinite(released_dbp)
    )
    if not target_valid:
        raise ValueError("Finite scalar SBP and DBP targets are required")
    expected = c.sample_rate_hz * c.window_seconds
    raw_channels = {}
    analysis_channels = {}
    resampled = {
        "ecg": resample_signal(np.asarray(ecg), source_hz, c.sample_rate_hz),
        "ppg": resample_signal(np.asarray(ppg), source_hz, c.sample_rate_hz),
    }
    lengths = {name: len(values) for name, values in resampled.items()}
    common_length = min(lengths.values())
    if common_length < expected:
        raise ValueError(f"Synchronized channels are shorter than the required crop: {lengths}")
    # Use one shared crop index for all channels.  Independent centered crops
    # can create an artificial ECG--PPG delay when source lengths differ.
    shared_start = (common_length - expected) // 2
    for name, band in (("ecg", c.ecg_band_hz), ("ppg", c.ppg_band_hz)):
        values = resampled[name]
        relative_start = shared_start if len(values) == common_length else min(shared_start, len(values) - expected)
        values = values[relative_start:relative_start + expected]
        raw_channels[name] = values.astype(np.float64, copy=True)
        analysis_channels[name] = wavelet_denoise(
            bandpass(values, c.sample_rate_hz, band, c.filter_order)
        )
    r_peaks = detect_r_peaks(analysis_channels["ecg"], c.sample_rate_hz)
    onsets = detect_ppg_onsets(analysis_channels["ppg"], c.sample_rate_hz)
    apparent_pat_ms = matched_pat_ms(r_peaks, onsets, c.sample_rate_hz)
    sbp = float(released_sbp)
    dbp = float(released_dbp)
    resolved_label_source = label_source or "scalar_bp_target"
    abp_waveform = None
    abp_sbp = abp_dbp = a_sqi = float("nan")
    beat_max = np.asarray([], dtype=float)
    beat_min = np.asarray([], dtype=float)
    abp_available = abp is not None and np.asarray(abp).size > 0
    abp_audit_valid = False
    if abp_available:
        try:
            a_resampled = resample_signal(np.asarray(abp), source_hz, c.sample_rate_hz)
            if len(a_resampled) < shared_start + expected:
                raise ValueError("Reference ABP is shorter than the shared ECG/PPG crop")
            a_crop = a_resampled[shared_start:shared_start + expected]
            abp_waveform = wavelet_denoise(bandpass(a_crop, c.sample_rate_hz, c.abp_band_hz, c.filter_order))
            abp_sbp, abp_dbp, beat_max, beat_min = abp_beat_labels(a_crop, r_peaks, c.sample_rate_hz)
            a_sqi = abp_sqi(abp_waveform, beat_max, beat_min)
            abp_audit_valid = bool(np.isfinite(a_sqi) and a_sqi >= c.abp_quality_min)
        except (ValueError, FloatingPointError):
            # Optional reference-audit failure never changes ECG/PPG eligibility.
            abp_waveform = None
    e_sqi = ecg_sqi(analysis_channels["ecg"], r_peaks, c.sample_rate_hz)
    # The scale-invariant PPG SQI is computed on the same zero-phase analysis
    # branch used for beat detection; model inputs remain the raw shared crop
    # transformed separately by the declared deterministic input view.
    p_sqi = ppg_sqi(analysis_channels["ppg"], c.sample_rate_hz)
    array_stable, _, delays = synchronization_stability(
        raw_channels["ecg"], raw_channels["ppg"], c.sample_rate_hz,
    )
    flags = {
        "adult_metadata": bool(age_years is not None and np.isfinite(age_years) and float(age_years) >= 18),
        "scalar_target_valid": bool(np.isfinite(sbp) and np.isfinite(dbp) and 0 < dbp < sbp),
        "beat_count_pass": bool(len(r_peaks) >= c.minimum_cycles),
        "ecg_quality_pass": bool(e_sqi >= c.ecg_quality_min),
        "ppg_quality_pass": bool(p_sqi >= c.ppg_quality_min),
        "paired_sample_continuity": bool(array_stable),
        "sqi_rule_pass": bool(min(e_sqi, p_sqi) >= 0.40 and max(e_sqi, p_sqi) >= 0.55),
    }
    ecg_views = model_input_views(raw_channels["ecg"], patch_samples=expected // c.token_count)
    ppg_views = model_input_views(raw_channels["ppg"], patch_samples=expected // c.token_count)
    return {
        # Backwards-compatible aliases point to the common model-input view.
        "ecg": ecg_views["window_robust"],
        "ppg": ppg_views["window_robust"],
        "ecg_patch_local": ecg_views["patch_local"],
        "ppg_patch_local": ppg_views["patch_local"],
        "ecg_window_robust": ecg_views["window_robust"],
        "ppg_window_robust": ppg_views["window_robust"],
        "abp": None if abp_waveform is None else abp_waveform.astype(np.float32),
        "sqi_tokens": np.stack([
            token_sqi(analysis_channels["ecg"], "ecg", c.sample_rate_hz, c.token_count),
            token_sqi(analysis_channels["ppg"], "ppg", c.sample_rate_hz, c.token_count),
        ]),
        "ecg_sqi": e_sqi,
        "ppg_sqi": p_sqi,
        "abp_sqi": a_sqi,
        "abp_available": bool(abp_available),
        "abp_audit_valid": bool(abp_audit_valid),
        "sample_continuity_pass": array_stable,
        "r_peaks": r_peaks,
        "ppg_onsets": onsets,
        "pat_detected": bool(np.isfinite(apparent_pat_ms)),
        "pat_ms": apparent_pat_ms,
        "pat_in_model_band": bool(np.isfinite(apparent_pat_ms) and 120.0 <= apparent_pat_ms <= 450.0),
        "sbp": sbp,
        "dbp": dbp,
        "abp_derived_sbp": float(abp_sbp),
        "abp_derived_dbp": float(abp_dbp),
        "label_source": resolved_label_source,
        "beat_max": beat_max,
        "beat_min": beat_min,
        "retention_flags": flags,
        "retained_before_subject_cap": retention_decision(flags),
    }
