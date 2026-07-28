from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_sample(frame: pd.DataFrame, seed: int, subjects: int, windows_per_subject: int) -> pd.DataFrame:
    counts = frame.groupby("subject_id").size()
    pool = pd.Series(counts[counts >= windows_per_subject].index.astype(str), name="subject_id")
    chosen = pool.sample(n=subjects, random_state=seed).sort_values()
    subject_values = frame.subject_id.astype(str)
    parts = []
    for subject in chosen:
        subject_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{subject}".encode("utf-8")).digest()[:4], "little"
        )
        parts.append(frame.loc[subject_values == subject].sample(n=windows_per_subject, random_state=subject_seed))
    return pd.concat(parts, ignore_index=True).sort_values(["subject_id", "window_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the released held-out attention export and its row-level summary")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(ROOT / "artifacts/attention/attention_export_manifest.csv")
    if len(manifest) != 1:
        raise AssertionError("Attention export manifest must contain exactly one frozen export")
    if manifest.iloc[0].selection != "fixed-seed balanced four-window-per-subject subject-grouped OOF sample":
        raise AssertionError("Attention export is not declared as the fixed balanced multi-window OOF sample")
    export_path = ROOT / str(manifest.iloc[0].export)
    data = np.load(export_path, allow_pickle=False)
    allowed = data["allowed_lags_ms"].astype(float)
    weights = data["weights"].astype(float)
    window_id = data["window_id"].astype(str)
    pat = data["pat_ms"].astype(float)
    summary = pd.read_csv(ROOT / "artifacts/attention/window_level_attention_summary.csv.gz")
    primary = pd.read_csv(
        ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz",
        usecols=["window_id", "subject_id", "pat_detected", "pat_ms", "evaluation_fold_id"],
    )
    eligible = primary[primary.pat_detected.astype(bool)]
    expected_sample = balanced_sample(
        eligible,
        int(manifest.iloc[0].selection_seed),
        int(manifest.iloc[0].subjects),
        int(manifest.iloc[0].windows_per_subject),
    )
    if not np.array_equal(expected_sample.window_id.astype(str).to_numpy(), summary.window_id.astype(str).to_numpy()):
        raise AssertionError("Released attention windows do not reproduce the fixed-seed balanced sample")

    if not np.array_equal(allowed, np.asarray([192.0, 256.0, 320.0, 384.0])):
        raise AssertionError(f"Unexpected released lag support: {allowed.tolist()}")
    if weights.ndim != 2 or weights.shape != (len(summary), len(allowed)):
        raise AssertionError("Attention export dimensions do not match the released window summary")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise AssertionError("Attention weights must be finite and nonnegative")
    row_sum_error = float(np.max(np.abs(weights.sum(axis=1) - 1.0)))
    if row_sum_error > 7e-6:
        raise AssertionError(f"Attention row sums are invalid: {row_sum_error}")
    if not np.array_equal(window_id, summary.window_id.astype(str).to_numpy()):
        raise AssertionError("Attention export window order differs from the released summary")
    subject_fold = primary[["subject_id", "evaluation_fold_id"]].drop_duplicates()
    if subject_fold.subject_id.nunique() != len(subject_fold):
        raise AssertionError("A subject is bound to more than one outer fold")
    fold_by_subject = dict(zip(subject_fold.subject_id.astype(str), subject_fold.evaluation_fold_id.astype(int), strict=True))
    expected_folds = summary.subject_id.astype(str).map(fold_by_subject).to_numpy(np.int32)
    if not np.array_equal(data["outer_fold_id"].astype(np.int32), expected_folds):
        raise AssertionError("Attention export rows are not bound to their subject-grouped outer-test folds")
    if set(summary.evaluation_role.astype(str)) != {"outer_test"} or str(data["evaluation_role"][0]) != "outer_test":
        raise AssertionError("Attention export evaluation role is not outer test")
    if int(manifest.iloc[0].subjects) != summary.subject_id.nunique() or int(manifest.iloc[0].outer_folds) != summary.outer_fold_id.nunique():
        raise AssertionError("Attention export subject/fold counts disagree with its manifest")
    subject_counts = summary.groupby("subject_id").size()
    if not (subject_counts == int(manifest.iloc[0].windows_per_subject)).all():
        raise AssertionError("Attention export is not balanced within subject")
    prediction_sha = sha256_file(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz")
    fold_manifest_sha = sha256_file(ROOT / "data/folds/pulsedb_vital_subject_grouped_fold_manifest.csv.gz")
    if (
        "prediction_authority_sha256" not in summary
        or "fold_manifest_sha256" not in summary
        or set(summary.prediction_authority_sha256.astype(str)) != {prediction_sha}
        or set(summary.fold_manifest_sha256.astype(str)) != {fold_manifest_sha}
        or str(manifest.iloc[0].lineage_binding)
        != "subject-grouped OOF prediction authority plus frozen outer-fold manifest"
    ):
        raise AssertionError("Attention export is not bound to the released OOF predictions and fold manifest")
    if not np.allclose(pat, summary.pat_ms.to_numpy(float), atol=6e-5):
        raise AssertionError("Attention export PAT values differ from the released summary")

    expected = weights @ allowed
    peak = allowed[np.argmax(weights, axis=1)]
    expected_delta = float(np.max(np.abs(expected - summary.attention_expected_delay_ms.to_numpy(float))))
    peak_delta = float(np.max(np.abs(peak - summary.attention_peak_delay_ms.to_numpy(float))))
    if expected_delta > 7e-5 or peak_delta > 7e-5:
        raise AssertionError("Released attention summary is not derived from the sparse export")

    entropy = -(weights * np.log(np.clip(weights, 1e-12, None))).sum(axis=1) / np.log(weights.shape[1])
    report = {
        "status": "PASS",
        "windows": len(summary),
        "r_wave_queries": int(summary.r_wave_queries.sum()),
        "row_sum_max_error": row_sum_error,
        "expected_delay_max_delta": expected_delta,
        "peak_delay_max_delta": peak_delta,
        "normalized_entropy_mean": float(entropy.mean()),
        "selection_reproduced": True,
        "subjects": int(summary.subject_id.nunique()),
        "outer_folds": int(summary.outer_fold_id.nunique()),
        "prediction_rows_lineage_bound": len(summary),
        "scope": "verification of the released subject-grouped outer-test attention export and its prediction/fold lineage",
    }
    (args.output_dir / "attention_export_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
