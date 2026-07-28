from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute deployment-profile medians from released timing traces")
    parser.parse_args()
    raw = pd.read_csv(ROOT / "artifacts/profiling/runtime_samples.csv.gz")
    environments = pd.read_csv(ROOT / "artifacts/profiling/profile_environments.csv")
    reference = pd.read_csv(ROOT / "artifacts/metrics/secondary/deployment_profile.csv")
    quantization = pd.read_csv(ROOT / "artifacts/profiling/int8_agreement_summary.csv")

    if raw.duplicated(["profile_id", "sample_index"]).any():
        raise AssertionError("Profiling sample identifiers are not unique")
    counts = raw.groupby("profile_id").size()
    if not (counts == 5000).all() or set(counts.index) != set(environments.profile_id):
        raise AssertionError("Every profiling environment must contain exactly 5,000 samples")
    if not np.isfinite(raw[["forward_latency_ms", "end_to_end_latency_ms"]]).all().all():
        raise AssertionError("Profiling traces contain non-finite values")
    if not (raw.end_to_end_latency_ms > raw.forward_latency_ms).all():
        raise AssertionError("End-to-end timing must exceed forward timing")

    medians = raw.groupby("profile_id", as_index=False).agg(
        forward_latency_ms=("forward_latency_ms", "median"),
        end_to_end_latency_ms=("end_to_end_latency_ms", "median"),
    )
    observed = environments.merge(medians, on="profile_id", validate="one_to_one")
    observed["throughput_windows_per_s"] = 1000.0 / observed.end_to_end_latency_ms
    observed["compute_only_ratio_vs_8s"] = 8000.0 / observed.end_to_end_latency_ms
    columns = list(reference.columns)
    observed = observed[columns]
    merged = reference.merge(observed, on=["profile_id", "platform", "runtime"], suffixes=("_reference", "_observed"), validate="one_to_one")
    numeric = ["forward_latency_ms", "end_to_end_latency_ms", "throughput_windows_per_s", "compute_only_ratio_vs_8s"]
    max_delta = max(float(np.max(np.abs(merged[f"{name}_reference"] - merged[f"{name}_observed"]))) for name in numeric)
    if max_delta > 5e-6:
        raise AssertionError(f"Profiling summary mismatch: {max_delta}")

    q = quantization.iloc[0]
    if int(q.evaluation_windows) < 10000:
        raise AssertionError("INT8 agreement set is too small")
    if float(q.sbp_mae_int8 - q.sbp_mae_fp16) > 0.10 or float(q.dbp_mae_int8 - q.dbp_mae_fp16) > 0.10:
        raise AssertionError("INT8 accuracy change exceeds the fixed reporting bound")

    output = ROOT / "reports/reproduced"
    output.mkdir(parents=True, exist_ok=True)
    observed.to_csv(output / "deployment_profile.csv", index=False, float_format="%.7f")
    report = {
        "status": "PASS",
        "profiles": len(observed),
        "timing_samples": len(raw),
        "samples_per_profile": 5000,
        "maximum_summary_delta": max_delta,
        "int8_sbp_mae_change_mmHg": float(q.sbp_mae_int8 - q.sbp_mae_fp16),
        "int8_dbp_mae_change_mmHg": float(q.dbp_mae_int8 - q.dbp_mae_fp16),
    }
    (output / "profiling_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
