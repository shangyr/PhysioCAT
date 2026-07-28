from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Recompute attention-delay statistics and subject-centered null checks from sparse weights")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(ROOT / "artifacts/attention/sparse_attention_weights.npz", allow_pickle=False)
    weights = archive["weights"].astype(np.float64)
    lags = archive["allowed_lags_ms"].astype(np.float64)
    pat = archive["pat_ms"].astype(np.float64)
    subject = archive["subject_id"].astype(str)
    r_wave_queries = archive["r_wave_queries"].astype(int)
    row_sum_error = float(np.max(np.abs(weights.sum(axis=1) - 1)))
    if row_sum_error > 7e-4:
        raise AssertionError("Sparse attention rows do not sum to one")
    expected = weights @ lags
    observed_r = float(np.corrcoef(pat, expected)[0, 1])
    frame = pd.DataFrame({"subject_id": subject, "pat": pat, "expected": expected})
    centered_pat = frame.pat - frame.groupby("subject_id").pat.transform("mean")
    centered_expected = frame.expected - frame.groupby("subject_id").expected.transform("mean")
    centered_r = float(np.corrcoef(centered_pat, centered_expected)[0, 1])
    null = []
    for seed in range(100):
        rng = np.random.default_rng(6200 + seed)
        permuted = frame.groupby("subject_id").pat.transform(lambda values: rng.permutation(values.to_numpy()))
        permuted_centered = permuted - frame.groupby("subject_id").pat.transform("mean")
        null.append(float(np.corrcoef(permuted_centered, centered_expected)[0, 1]))
    reference = pd.read_csv(ROOT / "artifacts/attention/attention_alignment_summary.csv")
    lookup = dict(zip(reference.analysis + "|" + reference.metric, reference.estimate))
    expected_values = {
        "Observed sparse attention|Pearson r: apparent PAT vs expected attention delay": observed_r,
        "Within-subject centered association|Pearson r after subject-mean centering": centered_r,
        "Within-subject window permutation|Null centered Pearson r across 100 permutations": float(np.mean(null)),
    }
    deltas = {key: abs(value - float(lookup[key])) for key, value in expected_values.items()}
    if max(deltas.values()) > 8e-4:
        raise AssertionError(f"Attention source mismatch: {deltas}")
    out = pd.DataFrame([{"metric": key, "estimate": value} for key, value in expected_values.items()])
    out.to_csv(args.output_dir / "attention_recomputed.csv", index=False, float_format="%.6f")
    report = {"status": "PASS", "n_windows": len(pat), "n_r_wave_queries": int(r_wave_queries.sum()), "aggregation": "head mean, then R-wave-query mean within window", "row_sum_max_error": row_sum_error, "expected_delay_r": observed_r, "subject_centered_r": centered_r, "within_subject_permutation_null_mean_r": float(np.mean(null)), "excess_over_within_subject_null": observed_r - float(np.mean(null))}
    (args.output_dir / "attention_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
