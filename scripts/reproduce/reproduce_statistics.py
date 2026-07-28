from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from physiocat.metrics import read_predictions
from physiocat.statistics import holm_adjust, paired_tests


def main():
    parser = argparse.ArgumentParser(description="Recompute subject-paired tests, Holm p-values, effects, and bootstrap intervals")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = read_predictions(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz")
    mechanism = read_predictions(ROOT / "artifacts/predictions/mechanism_control_predictions.csv.gz")
    pm = read_predictions(ROOT / "artifacts/predictions/pulsedb_mimic_zero_shot_predictions.csv.gz")
    mbp = read_predictions(ROOT / "artifacts/predictions/mimic_bp_zero_shot_predictions.csv.gz")
    result = pd.concat([
        paired_tests(primary, ["matched_no_delay", "mufubp_net"], "pulsedb_vital"),
        paired_tests(mechanism, ["uniform_delay_band", "ppg_leading_mirror", "direction_agnostic_local", "shifted_offsets_9_12", "attention_edge_ablation"], "pulsedb_vital"),
        paired_tests(pm, ["matched_no_delay", "mufubp_net"], "pulsedb_mimic"),
        paired_tests(mbp, ["matched_no_delay", "mufubp_net"], "mimic_bp"),
    ], ignore_index=True)
    result["holm_adjusted_p"] = np.maximum(
        holm_adjust(result.raw_p.to_numpy(float)),
        np.finfo(float).tiny,
    )
    reference = pd.read_csv(ROOT / "artifacts/metrics/statistics/paired_subject_tests.csv")
    key = ["cohort", "outcome", "comparator_model"]
    cols = ["mean_mae_reduction_mmHg", "cluster_bootstrap_ci_low", "cluster_bootstrap_ci_high", "wilcoxon_statistic", "raw_p", "holm_adjusted_p", "rank_biserial_effect"]
    a = reference.set_index(key)[cols].astype(float).sort_index()
    b = result.set_index(key)[cols].astype(float).sort_index()
    delta = (a - b).abs()
    finite_scale = np.maximum(np.abs(a.to_numpy()), 1e-300)
    relative = delta.to_numpy() / finite_scale
    if np.nanmax(relative) > 2e-4 and np.nanmax(delta.to_numpy()) > 2e-6:
        raise AssertionError("Recomputed statistical table differs from released reference")
    result.to_csv(args.output_dir / "paired_subject_tests.csv", index=False, float_format="%.12g")
    report = {"status": "PASS", "rows": len(result), "all_p_values_finite_and_positive": bool(np.isfinite(result.raw_p).all() and (result.raw_p > 0).all() and (result.holm_adjusted_p > 0).all())}
    (args.output_dir / "statistical_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
