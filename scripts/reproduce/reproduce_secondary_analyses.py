from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import math
import hashlib
import numpy as np
import pandas as pd


SECONDARY = ROOT / "artifacts/metrics/secondary"


def metric_row(y, pred):
    residual = np.asarray(pred, dtype=float) - np.asarray(y, dtype=float)
    return {
        "me": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "sde": float(np.std(residual, ddof=1)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "pearson_r": float(np.corrcoef(y, pred)[0, 1]),
        "within_5_pct": float(np.mean(np.abs(residual) <= 5) * 100),
        "within_10_pct": float(np.mean(np.abs(residual) <= 10) * 100),
        "within_15_pct": float(np.mean(np.abs(residual) <= 15) * 100),
    }


def bhs_grade(within_5, within_10, within_15):
    if within_5 >= 60 and within_10 >= 85 and within_15 >= 95:
        return "A"
    if within_5 >= 50 and within_10 >= 75 and within_15 >= 90:
        return "B"
    if within_5 >= 40 and within_10 >= 65 and within_15 >= 85:
        return "C"
    return "D"


def full_metric_record(y, pred, prefix):
    row = metric_row(y, pred)
    return {
        f"{prefix}_me": row["me"],
        f"{prefix}_mae": row["mae"],
        f"{prefix}_sde": row["sde"],
        f"{prefix}_rmse": row["rmse"],
        f"{prefix}_r": row["pearson_r"],
        f"{prefix}_within_5_pct": row["within_5_pct"],
        f"{prefix}_within_10_pct": row["within_10_pct"],
        f"{prefix}_within_15_pct": row["within_15_pct"],
        f"{prefix}_bhs_grade": bhs_grade(row["within_5_pct"], row["within_10_pct"], row["within_15_pct"]),
        f"{prefix}_loa_low": row["me"] - 1.96 * row["sde"],
        f"{prefix}_loa_high": row["me"] + 1.96 * row["sde"],
    }


def quality_rejection(predictions):
    ranking = predictions.min_sqi.rank(method="first", ascending=False).to_numpy()
    rows = []
    for coverage in (1.00, 0.95, 0.90, 0.85, 0.80):
        keep = ranking <= int(round(coverage * len(predictions)))
        row = {"coverage": coverage, "n_windows": int(keep.sum())}
        for outcome in ("sbp", "dbp"):
            row.update(full_metric_record(predictions.loc[keep, outcome], predictions.loc[keep, f"pred_physiocat_{outcome}"], outcome))
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_tables(predictions):
    definitions = [
        ("Age 18-40", "Age", "18--40", predictions.age_years.between(18, 40), "18 <= age <= 40 years"),
        ("Age 41-65", "Age", "41--65", predictions.age_years.between(41, 65), "41 <= age <= 65 years"),
        ("Age 66+", "Age", ">=66", predictions.age_years >= 66, "age >= 66 years"),
        ("Female", "Sex", "Female", predictions.sex == "Female", "sex recorded as female"),
        ("Male", "Sex", "Male", predictions.sex == "Male", "sex recorded as male"),
        ("Lower-BP windows", "BP", "Lower-BP windows", (predictions.sbp < 130) & (predictions.dbp < 80), "SBP < 130 and DBP < 80 mmHg; window stratum, not a clinical diagnosis"),
        ("Elevated-BP windows", "BP", "Elevated-BP windows", (predictions.sbp >= 130) | (predictions.dbp >= 80), "SBP >= 130 or DBP >= 80 mmHg; window stratum, not a clinical diagnosis"),
        ("Normal rhythm", "Rhythm", "Regular", predictions.rhythm_group == "Regular", "regular-rhythm windows"),
        ("Irregular rhythm", "Rhythm", "RR-irregular", predictions.rhythm_group == "RR-irregular", "RR-irregular windows"),
        ("Lowest SQI", "SQI", "Lower", predictions.sqi_group == "Lower", "lowest retained signal-quality stratum"),
    ]
    rows, definition_rows = [], []
    for display, family, subgroup, mask, definition in definitions:
        row = {"figure_panel": "A", "display_label": display, "subgroup_family": family, "subgroup": subgroup, "definition": definition, "n_subjects": int(predictions.loc[mask, "subject_id"].nunique()), "n_windows": int(mask.sum())}
        for outcome in ("sbp", "dbp"):
            row[f"{outcome}_mae"] = float(np.mean(np.abs(predictions.loc[mask, f"pred_physiocat_{outcome}"] - predictions.loc[mask, outcome])))
        rows.append(row)
        definition_rows.append({"display_label": display, "family": family, "subgroup": subgroup, "definition": definition})
    sqi_rows = []
    for model in ("physiocat", "matched_no_delay"):
        for stratum, label in (("Higher", "high"), ("Middle", "medium"), ("Lower", "low")):
            mask = predictions.sqi_group == stratum
            row = {"model": model, "sqi_stratum": label, "n_windows": int(mask.sum())}
            for outcome in ("sbp", "dbp"):
                row.update(full_metric_record(predictions.loc[mask, outcome], predictions.loc[mask, f"pred_{model}_{outcome}"], outcome))
            sqi_rows.append(row)
    subgroup = pd.DataFrame(rows)
    return subgroup, pd.DataFrame(definition_rows), pd.DataFrame(sqi_rows)


def failure_attribution(predictions):
    rows = []
    for outcome in ("sbp", "dbp"):
        absolute_error = np.abs(predictions[f"pred_physiocat_{outcome}"] - predictions[outcome]).to_numpy()
        threshold = float(np.quantile(absolute_error, 0.90))
        tail = absolute_error >= threshold
        category = np.full(len(predictions), "Other", dtype=object)
        category[((predictions.sbp < 90) | (predictions.sbp > 180) | (predictions.dbp < 50) | (predictions.dbp > 110)).to_numpy()] = "BP extreme"
        category[(predictions.rhythm_group == "RR-irregular").to_numpy()] = "Irregular rhythm"
        category[((predictions.sbp >= 130) | (predictions.dbp >= 80)).to_numpy()] = "Elevated-BP windows"
        category[(predictions.sqi_group == "Lower").to_numpy()] = "Low SQI"
        tail_n = int(tail.sum())
        for name in ("Low SQI", "Elevated-BP windows", "Irregular rhythm", "BP extreme", "Other"):
            count = int(np.sum(tail & (category == name)))
            rows.append({"endpoint": outcome.upper(), "tail_definition": "absolute error >= endpoint-specific 90th percentile", "threshold_mmhg": threshold, "composition_category": name, "tail_windows": count, "tail_share_pct": 100 * count / tail_n, "exclusive_classification_rule": "Low SQI > Elevated-BP windows > Irregular rhythm > BP extreme > Other"})
    return pd.DataFrame(rows)


def representative_windows(predictions):
    sbp_error = np.abs(predictions.pred_physiocat_sbp - predictions.sbp)
    dbp_error = np.abs(predictions.pred_physiocat_dbp - predictions.dbp)
    combined_error = sbp_error + dbp_error
    easy_pool = predictions[(predictions.sqi_group == "Higher") & (predictions.rhythm_group == "Regular")]
    hard_pool = predictions[(predictions.sqi_group == "Lower") & (predictions.rhythm_group == "RR-irregular")]
    easy_pool = predictions if easy_pool.empty else easy_pool
    hard_pool = predictions if hard_pool.empty else hard_pool
    easy_target = float(combined_error.loc[easy_pool.index].median())
    hard_target = float(combined_error.loc[hard_pool.index].quantile(0.90))
    easy_score = (combined_error.loc[easy_pool.index] - easy_target).abs()
    hard_score = (combined_error.loc[hard_pool.index] - hard_target).abs()
    rows = []
    selections = (
        ("easier", easy_score.idxmin(), "high-SQI regular-rhythm pool; nearest combined-error median"),
        ("harder", hard_score.idxmin(), "low-SQI RR-irregular pool; nearest combined-error 90th percentile"),
    )
    for role, index, selection_rule in selections:
        row = predictions.loc[index]
        rows.append({"role": role, "window_id": row.window_id, "subject_id": row.subject_id, "sbp_mae": abs(row.pred_physiocat_sbp - row.sbp), "dbp_mae": abs(row.pred_physiocat_dbp - row.dbp), "min_sqi": row.min_sqi, "rhythm_group": row.rhythm_group, "reference_sbp": row.sbp, "reference_dbp": row.dbp, "selection_rule": selection_rule})
    result = pd.DataFrame(rows)
    waveform_asset = "artifacts/waveforms/representative_windows.npz"
    digest = hashlib.sha256((ROOT / waveform_asset).read_bytes()).hexdigest()
    result["waveform_asset"] = waveform_asset
    result["waveform_row"] = np.arange(len(result), dtype=int)
    result["waveform_sha256"] = digest
    return result


def external_threshold(cohorts):
    rows = []
    for cohort, predictions in cohorts:
        for model in ("physiocat", "matched_no_delay", "mufubp_net"):
            row = {"cohort": cohort, "model": model, "n_windows": len(predictions), "n_subjects": predictions.subject_id.nunique()}
            for outcome in ("sbp", "dbp"):
                row.update(full_metric_record(predictions[outcome], predictions[f"pred_{model}_{outcome}"], outcome))
            rows.append(row)
    return pd.DataFrame(rows)


def few_shot(predictions, reps=200):
    columns = ["sbp", "dbp", "pred_physiocat_sbp", "pred_physiocat_dbp", "pred_mufubp_net_sbp", "pred_mufubp_net_dbp"]
    groups = [{column: group[column].to_numpy(float) for column in columns} for _, group in predictions.groupby("subject_id", sort=True) if len(group) >= 11]
    calibration_windows = (0, 1, 2, 5, 10)
    prior_strength_windows = 5
    values = {(model, outcome, k): [] for model in ("physiocat", "mufubp_net") for outcome in ("sbp", "dbp") for k in calibration_windows}
    rng = np.random.default_rng(42)
    n_evaluation = 0
    for _ in range(reps):
        sums = {key: 0.0 for key in values}
        n_evaluation = 0
        for group in groups:
            order = rng.permutation(len(group["sbp"]))
            evaluate = order[10:]
            n_evaluation += len(evaluate)
            for model in ("physiocat", "mufubp_net"):
                for outcome in ("sbp", "dbp"):
                    residual = group[outcome] - group[f"pred_{model}_{outcome}"]
                    cumulative = np.cumsum(residual[order[:10]])
                    reference = group[outcome][evaluate]
                    prediction = group[f"pred_{model}_{outcome}"][evaluate]
                    for k in calibration_windows:
                        if k == 0:
                            offset = 0.0
                        else:
                            shrinkage = k / (k + prior_strength_windows)
                            offset = shrinkage * cumulative[k - 1] / k
                        sums[(model, outcome, k)] += float(np.abs(prediction + offset - reference).sum())
        for key in values:
            values[key].append(sums[key] / n_evaluation)
    rows = []
    for k in calibration_windows:
        row = {"calibration_windows": k, "n_resamples": reps, "eligible_subjects": len(groups), "n_evaluation_windows": n_evaluation, "offset_estimator": "none" if k == 0 else "shrunken subject residual mean (prior strength 5 windows)"}
        for model, prefix in (("physiocat", "physiocat"), ("mufubp_net", "baseline")):
            for outcome in ("sbp", "dbp"):
                draws = np.asarray(values[(model, outcome, k)])
                row[f"{prefix}_{outcome}_mae"] = float(draws.mean())
                row[f"{prefix}_{outcome}_resampling_low"] = float(np.quantile(draws, 0.025))
                row[f"{prefix}_{outcome}_resampling_high"] = float(np.quantile(draws, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def deterministic_subject_cap(frame, eligible, cap, seed=2012):
    eligible = np.asarray(eligible, dtype=bool)
    if cap is None:
        return eligible.copy()
    selected = np.zeros(len(frame), dtype=bool)
    rng = np.random.default_rng(seed)
    subject_values = frame.subject_id.astype(str).to_numpy()
    for subject in sorted(pd.unique(subject_values)):
        indices = np.flatnonzero(eligible & (subject_values == subject))
        if len(indices) <= cap:
            selected[indices] = True
        else:
            selected[np.sort(rng.choice(indices, cap, replace=False))] = True
    return selected


def retention_sensitivity(predictions):
    basic = (
        predictions.adult_metadata.astype(bool).to_numpy()
        & predictions.scalar_target_valid.astype(bool).to_numpy()
        & predictions.beat_count_pass.astype(bool).to_numpy()
        & predictions.paired_sample_continuity.astype(bool).to_numpy()
    )
    min_sqi = predictions[["ecg_sqi", "ppg_sqi"]].min(axis=1).to_numpy(float)
    max_sqi = predictions[["ecg_sqi", "ppg_sqi"]].max(axis=1).to_numpy(float)
    views = [
        ("Basic paired/target validity; no SQI threshold", None, None, 96),
        ("Relaxed joint SQI (min 0.30; max 0.45)", 0.30, 0.45, 96),
        ("Default joint SQI (min 0.40; max 0.55)", 0.40, 0.55, 96),
        ("Stricter joint SQI (min 0.50; max 0.65)", 0.50, 0.65, 96),
        ("Default joint SQI; 48-window cap", 0.40, 0.55, 48),
        ("Default joint SQI; no subject cap", 0.40, 0.55, None),
    ]
    rows = []
    for label, lower, upper, cap in views:
        eligible = basic.copy()
        if lower is not None:
            eligible &= (min_sqi >= lower) & (max_sqi >= upper)
        keep = deterministic_subject_cap(predictions, eligible, cap)
        row = {
            "analysis_view": label,
            "minimum_modality_sqi": np.nan if lower is None else lower,
            "maximum_modality_sqi": np.nan if upper is None else upper,
            "subject_cap": "none" if cap is None else cap,
            "subjects": predictions.loc[keep, "subject_id"].nunique(),
            "windows": int(keep.sum()),
            "evaluation_model": "frozen main PhysioCAT subject-grouped OOF model for each held-out fold",
        }
        for outcome in ("sbp", "dbp"):
            row[f"{outcome}_mae"] = float(np.mean(np.abs(predictions.loc[keep, f"pred_physiocat_{outcome}"] - predictions.loc[keep, outcome])))
        rows.append(row)
    return pd.DataFrame(rows)


def sqi_reference_summary(annotations):
    rater_1 = annotations.rater_1_usable.astype(bool).to_numpy()
    rater_2 = annotations.rater_2_usable.astype(bool).to_numpy()
    consensus = annotations.consensus_usable.astype(bool).to_numpy()
    joint_rule = annotations.joint_retention_rule_pass.astype(bool).to_numpy()
    observed = float(np.mean(rater_1 == rater_2))
    p1, p2 = float(rater_1.mean()), float(rater_2.mean())
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / max(1 - expected, 1e-12)
    min_sqi = annotations[["ecg_sqi", "ppg_sqi"]].min(axis=1).to_numpy(float)
    ranks = pd.Series(min_sqi).rank(method="average").to_numpy(float)
    n_positive, n_negative = int(consensus.sum()), int((~consensus).sum())
    auc = (float(ranks[consensus].sum()) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return pd.DataFrame([{
        "windows": len(annotations),
        "subjects": annotations.subject_id.nunique(),
        "consensus_usable_pct": 100 * float(consensus.mean()),
        "dual_rater_observed_agreement": observed,
        "dual_rater_cohen_kappa": kappa,
        "min_modality_sqi_auc": auc,
        "joint_rule_sensitivity": float(np.mean(joint_rule[consensus])),
        "joint_rule_specificity": float(np.mean(~joint_rule[~consensus])),
        "sampling": "fixed-seed equal allocation across min-SQI deciles",
        "blinding": "BP labels, ABP, model predictions, and downstream errors hidden",
    }])


def abp_reference_summary(annotations, audit):
    rater_1 = annotations.rater_1_usable.astype(bool).to_numpy()
    rater_2 = annotations.rater_2_usable.astype(bool).to_numpy()
    consensus = annotations.consensus_usable.astype(bool).to_numpy()
    threshold_pass = annotations.abp_quality_threshold_pass.astype(bool).to_numpy()
    observed = float(np.mean(rater_1 == rater_2))
    p1, p2 = float(rater_1.mean()), float(rater_2.mean())
    expected = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (observed - expected) / max(1 - expected, 1e-12)
    ranks = pd.Series(annotations.abp_sqi.to_numpy(float)).rank(method="average").to_numpy(float)
    n_positive, n_negative = int(consensus.sum()), int((~consensus).sum())
    auc = (float(ranks[consensus].sum()) - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    retained = audit.loc[audit.retained.astype(bool)].copy()
    q1, median, q3 = np.quantile(retained.abp_sqi.to_numpy(float), [0.25, 0.50, 0.75])
    below = int((retained.abp_sqi < 0.42).sum())
    return pd.DataFrame([{
        "retained_windows": len(retained),
        "retained_subjects": retained.subject_id.nunique(),
        "abp_sqi_median": median,
        "abp_sqi_q1": q1,
        "abp_sqi_q3": q3,
        "below_threshold_windows": below,
        "below_threshold_pct": 100.0 * below / len(retained),
        "review_windows": len(annotations),
        "review_subjects": annotations.subject_id.nunique(),
        "consensus_usable_pct": 100.0 * float(consensus.mean()),
        "dual_rater_observed_agreement": observed,
        "dual_rater_cohen_kappa": kappa,
        "abp_sqi_auc": auc,
        "threshold_sensitivity": float(np.mean(threshold_pass[consensus])),
        "threshold_specificity": float(np.mean(~threshold_pass[~consensus])),
        "threshold": 0.42,
        "sampling": "fixed-seed threshold-enriched allocation across four retained ABP-SQI strata",
        "blinding": "ECG/PPG, scalar BP labels, model predictions, and downstream errors hidden",
    }])


def abp_reference_sensitivity(predictions, audit):
    reference = audit.loc[
        audit.retained.astype(bool),
        ["window_id", "subject_id", "abp_sqi", "abp_quality_pass"],
    ]
    merged = predictions.merge(reference, on=["window_id", "subject_id"], how="inner", validate="one_to_one")
    if len(merged) != len(predictions):
        raise AssertionError("ABP-quality sensitivity does not map every primary prediction")
    rows = []
    for label, threshold in (("All retained windows", None), ("ABP SQI >= 0.42", 0.42), ("ABP SQI >= 0.55", 0.55)):
        keep = np.ones(len(merged), dtype=bool) if threshold is None else merged.abp_sqi.to_numpy(float) >= threshold
        for model in ("physiocat", "matched_no_delay", "mufubp_net"):
            row = {
                "analysis_view": label,
                "abp_sqi_threshold": np.nan if threshold is None else threshold,
                "model": model,
                "subjects": merged.loc[keep, "subject_id"].nunique(),
                "windows": int(keep.sum()),
                "abp_sqi_median": float(np.median(merged.loc[keep, "abp_sqi"])),
            }
            for outcome in ("sbp", "dbp"):
                row[f"{outcome}_mae"] = float(np.mean(np.abs(merged.loc[keep, f"pred_{model}_{outcome}"] - merged.loc[keep, outcome])))
            rows.append(row)
    return pd.DataFrame(rows)


def source_shift(cohorts):
    del cohorts
    archive = np.load(ROOT / "artifacts/representations/source_shift_input_features.npz", allow_pickle=False)
    features = archive["input_summary_features"].astype(np.float64)
    feature_names = archive["feature_names"].astype(str).tolist()
    feature_mean = features.mean(axis=0)
    feature_sd = features.std(axis=0, ddof=1)
    standardized = (features - feature_mean) / np.where(feature_sd > 1e-8, feature_sd, 1.0)
    _, _, vt = np.linalg.svd(standardized, full_matrices=False)
    loadings = vt[:2].T
    projected = standardized @ loadings
    if np.corrcoef(projected[:, 0], standardized[:, feature_names.index("ecg_spectral_centroid_hz")])[0, 1] < 0:
        projected[:, 0] *= -1
    if np.corrcoef(projected[:, 1], standardized[:, feature_names.index("age_years")])[0, 1] < 0:
        projected[:, 1] *= -1
    rows = []
    cohort_array = archive["cohort"].astype(str)
    window_array = archive["window_id"].astype(str)
    sample_seeds = {"PulseDB-Vital": 4201, "PulseDB-MIMIC": 4202, "MIMIC-BP": 4203}
    for display in ("PulseDB-Vital", "PulseDB-MIMIC", "MIMIC-BP"):
        indices = np.flatnonzero(cohort_array == display)
        for point_id, index in enumerate(indices, start=1):
            x, y = projected[index]
            rows.append({"cohort": display, "point_id": point_id, "window_id": window_array[index], "projection_1": x, "projection_2": y, "method": "PCA of standardized released demographic, SQI, spectral, and pulse-shape summaries", "sample_seed": sample_seeds[display]})
    return pd.DataFrame(rows)


def external_summary(thresholds):
    rows = []
    for cohort, display in (("pulsedb_mimic", "PulseDB-MIMIC"), ("mimic_bp", "MIMIC-BP")):
        for model, model_display in (("physiocat", "PhysioCAT"), ("matched_no_delay", "Matched no-delay"), ("mufubp_net", "MuFuBP-Net")):
            source = thresholds[(thresholds.cohort == cohort) & (thresholds.model == model)].iloc[0]
            rows.append({"dataset": display, "model": model_display, "sbp_mae": source.sbp_mae, "sbp_sde": source.sbp_sde, "dbp_mae": source.dbp_mae, "dbp_sde": source.dbp_sde})
    return pd.DataFrame(rows)


def cluster_bootstrap(predictions, cohort, reps=2000):
    rng = np.random.default_rng(42)
    rows = []
    for model in ("physiocat", "matched_no_delay", "mufubp_net"):
        for outcome in ("sbp", "dbp"):
            work = pd.DataFrame({"subject_id": predictions.subject_id, "ae": np.abs(predictions[f"pred_{model}_{outcome}"] - predictions[outcome])})
            grouped = work.groupby("subject_id", sort=False).ae.agg(["mean", "count", "sum"]).reset_index()
            n = len(grouped)
            draws = rng.integers(0, n, size=(reps, n), endpoint=False)
            means, sums, counts = grouped["mean"].to_numpy(), grouped["sum"].to_numpy(), grouped["count"].to_numpy()
            subject_boot = means[draws].mean(axis=1)
            window_boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
            rows.extend([
                {"cohort": cohort, "model": model, "outcome": outcome.upper(), "weighting": "window", "estimate": float(sums.sum() / counts.sum()), "ci_low": float(np.quantile(window_boot, 0.025)), "ci_high": float(np.quantile(window_boot, 0.975)), "bootstrap_reps": reps},
                {"cohort": cohort, "model": model, "outcome": outcome.upper(), "weighting": "subject", "estimate": float(means.mean()), "ci_low": float(np.quantile(subject_boot, 0.025)), "ci_high": float(np.quantile(subject_boot, 0.975)), "bootstrap_reps": reps},
            ])
    return pd.DataFrame(rows)


def mimic_bridge(audit, retained, main_bootstrap):
    retained_mask = audit.retained.astype(bool)
    bridge = pd.DataFrame([
        {"analysis_subset": "Target-formed paired ECG+PPG candidates", "subjects": audit.subject_id.nunique(), "windows": len(audit), "subject_coverage_pct": 100.0, "window_coverage_pct": 100.0, "role": "external evaluable candidate pool after disclosed target formation"},
        {"analysis_subset": "Fixed retained MIMIC-BP cohort", "subjects": retained.subject_id.nunique(), "windows": len(retained), "subject_coverage_pct": 100 * retained.subject_id.nunique() / audit.subject_id.nunique(), "window_coverage_pct": 100 * retained_mask.mean(), "role": "zero-shot external validation"},
    ])
    rows = []
    for model, display in (("physiocat", "PhysioCAT"), ("matched_no_delay", "Matched no-delay"), ("mufubp_net", "MuFuBP-Net")):
        row = {"analysis": "MIMIC-BP", "model": display, "subjects": retained.subject_id.nunique(), "windows": len(retained), "role": "zero-shot external validation"}
        for outcome in ("SBP", "DBP"):
            ci = main_bootstrap[(main_bootstrap.model == model) & (main_bootstrap.outcome == outcome) & (main_bootstrap.weighting == "window")].iloc[0]
            row[f"{outcome.lower()}_mae"] = ci.estimate
            row[f"{outcome.lower()}_ci_low"] = ci.ci_low
            row[f"{outcome.lower()}_ci_high"] = ci.ci_high
        rows.append(row)
    return bridge, pd.DataFrame(rows)


def selection_bias(audit):
    frame = audit.copy()
    frame["min_sqi"] = frame[["ecg_sqi", "ppg_sqi"]].min(axis=1)
    frame["group"] = "Removed by fixed eligibility checks"
    frame.loc[frame.retained.astype(bool), "group"] = "Final retained after cap"
    quality_fields = [
        "adult_metadata", "scalar_target_valid", "beat_count_pass", "ecg_quality_pass",
        "ppg_quality_pass", "paired_sample_continuity", "sqi_rule_pass",
    ]
    eligible = frame[quality_fields].astype(bool).all(axis=1)
    cap_only = eligible & ~frame.subject_balance_cap_pass.astype(bool)
    frame.loc[cap_only, "group"] = "Removed only by 96-window cap"
    rows = []
    for group, subset in frame.groupby("group", sort=False):
        def summary(column):
            values = subset[column].to_numpy(float)
            q = np.percentile(values, [25, 50, 75])
            return float(q[1]), float(q[0]), float(q[2])
        sbp_m, sbp_q1, sbp_q3 = summary("sbp")
        dbp_m, dbp_q1, dbp_q3 = summary("dbp")
        age_m, age_q1, age_q3 = summary("age_years")
        sqi_m, sqi_q1, sqi_q3 = summary("min_sqi")
        pat = subset.loc[subset.pat_detected.astype(bool), "pat_ms"].to_numpy(float)
        pat_q1, pat_m, pat_q3 = np.percentile(pat, [25, 50, 75]) if len(pat) else (np.nan, np.nan, np.nan)
        rows.append({"group": group, "n_windows": len(subset), "sbp_median": sbp_m, "sbp_q1": sbp_q1, "sbp_q3": sbp_q3, "dbp_median": dbp_m, "dbp_q1": dbp_q1, "dbp_q3": dbp_q3, "age_median": age_m, "age_q1": age_q1, "age_q3": age_q3, "female_pct": 100 * float((subset.sex == "Female").mean()), "min_sqi_median": sqi_m, "min_sqi_q1": sqi_q1, "min_sqi_q3": sqi_q3, "pat_median": pat_m, "pat_q1": pat_q1, "pat_q3": pat_q3})
    return pd.DataFrame(rows)


def target_formation_selection(raw):
    frame = raw.copy()
    formed_all = frame.loc[frame.target_derivation_pass.astype(bool)].copy()
    failed_all = frame.loc[frame.paired_waveform_crop_pass.astype(bool) & ~frame.target_derivation_pass.astype(bool)].copy()
    formed = formed_all.loc[formed_all.repository_scalar_available.astype(bool)].copy()
    failed = failed_all.loc[failed_all.repository_scalar_available.astype(bool)].copy()
    if formed.empty or failed.empty:
        raise AssertionError("Target-formation selection audit lacks a comparison group")

    def quantiles(values):
        q1, median, q3 = np.quantile(values.to_numpy(float), [0.25, 0.50, 0.75])
        return float(median), float(q1), float(q3)

    def smd(left, right):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0)
        return float((right.mean() - left.mean()) / pooled) if pooled > 0 else 0.0

    differences = {
        "repository_sbp_smd": smd(formed.repository_sbp, failed.repository_sbp),
        "repository_dbp_smd": smd(formed.repository_dbp, failed.repository_dbp),
        "age_smd": smd(formed.age_years, failed.age_years),
        "female_smd": smd((formed.sex == "Female").to_numpy(float), (failed.sex == "Female").to_numpy(float)),
        "input_min_sqi_smd": smd(formed.input_min_sqi, failed.input_min_sqi),
        "rr_irregularity_smd": smd(formed.rr_irregularity, failed.rr_irregularity),
    }
    rows = []
    for group, subset, all_subset in (("Target formed", formed, formed_all), ("Target-formation failure", failed, failed_all)):
        sbp_m, sbp_q1, sbp_q3 = quantiles(subset.repository_sbp)
        dbp_m, dbp_q1, dbp_q3 = quantiles(subset.repository_dbp)
        age_m, age_q1, age_q3 = quantiles(subset.age_years)
        sqi_m, sqi_q1, sqi_q3 = quantiles(subset.input_min_sqi)
        rr_m, rr_q1, rr_q3 = quantiles(subset.rr_irregularity)
        rows.append({
            "cohort": str(frame.cohort.iloc[0]), "group": group, "paired_candidates": len(all_subset),
            "subjects": subset.subject_id.nunique(), "repository_scalar_available": len(subset),
            "repository_scalar_available_pct": 100.0 * len(subset) / len(all_subset),
            "repository_sbp_median": sbp_m, "repository_sbp_q1": sbp_q1, "repository_sbp_q3": sbp_q3,
            "repository_dbp_median": dbp_m, "repository_dbp_q1": dbp_q1, "repository_dbp_q3": dbp_q3,
            "age_median": age_m, "age_q1": age_q1, "age_q3": age_q3,
            "female_pct": 100.0 * float((subset.sex == "Female").mean()),
            "input_min_sqi_median": sqi_m, "input_min_sqi_q1": sqi_q1, "input_min_sqi_q3": sqi_q3,
            "rr_irregularity_median": rr_m, "rr_irregularity_q1": rr_q1, "rr_irregularity_q3": rr_q3,
            **{name: (0.0 if group == "Target formed" else value) for name, value in differences.items()},
        })
    return pd.DataFrame(rows)


def repository_scalar_sensitivity(cohort, predictions, audit):
    retained = audit.loc[audit.retained.astype(bool), ["window_id", "subject_id", "sbp", "dbp", "released_sbp", "released_dbp"]]
    prediction_columns = [f"pred_{model}_{outcome}" for model in ("physiocat", "matched_no_delay", "mufubp_net") for outcome in ("sbp", "dbp")]
    merged = predictions[["window_id", "subject_id", *prediction_columns]].merge(retained, on=["window_id", "subject_id"], how="inner", validate="one_to_one")
    if len(merged) != len(predictions):
        raise AssertionError(f"{cohort} repository-scalar sensitivity does not map every retained row")
    rows = []
    for model in ("physiocat", "matched_no_delay", "mufubp_net"):
        for outcome in ("sbp", "dbp"):
            aligned = merged[outcome].to_numpy(float)
            repository = merged[f"released_{outcome}"].to_numpy(float)
            estimate = merged[f"pred_{model}_{outcome}"].to_numpy(float)
            delta = repository - aligned
            aligned_mae = float(np.mean(np.abs(estimate - aligned)))
            repository_mae = float(np.mean(np.abs(estimate - repository)))
            rows.append({"cohort": cohort, "model": model, "outcome": outcome.upper(), "subjects": merged.subject_id.nunique(), "windows": len(merged), "aligned_target_mae": aligned_mae, "repository_scalar_mae": repository_mae, "mae_change_repository_minus_aligned": repository_mae - aligned_mae, "repository_minus_aligned_target_bias": float(delta.mean()), "repository_vs_aligned_target_mae": float(np.mean(np.abs(delta))), "repository_minus_aligned_target_sd": float(delta.std(ddof=1))})
    return pd.DataFrame(rows)


def rank_biserial(values):
    values = np.asarray(values, dtype=float)
    values = values[np.abs(values) > 1e-12]
    if len(values) == 0:
        return 0.0
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy(float)
    denominator = float(ranks.sum())
    return float((ranks[values > 0].sum() - ranks[values < 0].sum()) / denominator)


def pat_interaction_tables(primary, mechanism):
    combined = primary.merge(
        mechanism[[
            "window_id", "subject_id",
            "pred_ppg_leading_mirror_sbp", "pred_ppg_leading_mirror_dbp",
            "pred_direction_agnostic_local_sbp", "pred_direction_agnostic_local_dbp",
        ]],
        on=["window_id", "subject_id"], how="inner", validate="one_to_one",
    )
    if len(combined) != len(primary):
        raise AssertionError("PAT interaction recomputation lost OOF rows")
    pat = combined.pat_ms.to_numpy(float)
    detected = np.isfinite(pat)
    groups = [
        ("primary", "PAT detected: 120--450 ms", detected & (pat >= 120) & (pat <= 450)),
        ("primary", "PAT detected: outside 120--450 ms", detected & ((pat < 120) | (pat > 450))),
        ("primary", "PAT not detected", ~detected),
        ("detailed", "60--119 ms", detected & (pat >= 60) & (pat < 120)),
        ("detailed", "120--199 ms", detected & (pat >= 120) & (pat < 200)),
        ("detailed", "200--299 ms", detected & (pat >= 200) & (pat < 300)),
        ("detailed", "300--399 ms", detected & (pat >= 300) & (pat < 400)),
        ("detailed", "400--450 ms", detected & (pat >= 400) & (pat <= 450)),
        ("detailed", "451--650 ms", detected & (pat > 450) & (pat <= 650)),
        ("detailed", "PAT not detected", ~detected),
    ]
    models = ("physiocat", "matched_no_delay", "ppg_leading_mirror", "direction_agnostic_local")
    displays = {
        "physiocat": "PhysioCAT", "matched_no_delay": "Matched no-delay",
        "ppg_leading_mirror": "PPG-leading mirror delay band",
        "direction_agnostic_local": "Zero-centered nonzero local",
    }
    rng = np.random.default_rng(25_072)
    rows, subject_differences = [], {}
    for group_type, group, mask in groups:
        subset = combined.loc[mask]
        if subset.empty:
            raise AssertionError(f"Empty apparent-PAT group: {group}")
        for outcome in ("sbp", "dbp"):
            reference = subset[outcome].to_numpy(float)
            phys_error = np.abs(subset[f"pred_physiocat_{outcome}"].to_numpy(float) - reference)
            for model in models:
                error = np.abs(subset[f"pred_{model}_{outcome}"].to_numpy(float) - reference)
                paired = pd.DataFrame({"subject_id": subset.subject_id.to_numpy(str), "difference": error - phys_error}).groupby("subject_id", sort=True).difference.mean()
                subject_differences[(group_type, group, f"{model}:{outcome}")] = paired
                if model == "physiocat":
                    ci_low = ci_high = effect = 0.0
                else:
                    values = paired.to_numpy(float)
                    draws = rng.integers(0, len(values), size=(2000, len(values)))
                    boot = values[draws].mean(axis=1)
                    ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
                    effect = rank_biserial(values)
                rows.append({
                    "group_type": group_type, "pat_group": group, "model": model,
                    "display_name": displays[model], "outcome": outcome.upper(),
                    "subjects": subset.subject_id.nunique(), "windows": len(subset),
                    "mae": float(error.mean()),
                    "mae_reduction_vs_physiocat": float(error.mean() - phys_error.mean()),
                    "subject_paired_ci_low": ci_low, "subject_paired_ci_high": ci_high,
                    "rank_biserial_effect": effect, "bootstrap_reps": 2000,
                })
    contrasts = []
    in_band = "PAT detected: 120--450 ms"
    for comparator in ("matched_no_delay", "ppg_leading_mirror", "direction_agnostic_local"):
        for outcome in ("sbp", "dbp"):
            reference = subject_differences[("primary", in_band, f"{comparator}:{outcome}")]
            for other in ("PAT detected: outside 120--450 ms", "PAT not detected"):
                alternative = subject_differences[("primary", other, f"{comparator}:{outcome}")]
                common = reference.index.intersection(alternative.index)
                contrast = (reference.loc[common] - alternative.loc[common]).to_numpy(float)
                draws = rng.integers(0, len(contrast), size=(2000, len(contrast)))
                boot = contrast[draws].mean(axis=1)
                contrasts.append({
                    "comparator_model": comparator, "outcome": outcome.upper(),
                    "reference_group": in_band, "comparison_group": other,
                    "paired_subjects": len(common),
                    "difference_in_mae_reduction_mmHg": float(contrast.mean()),
                    "cluster_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
                    "cluster_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_reps": 2000,
                })
    return pd.DataFrame(rows), pd.DataFrame(contrasts)


def source_seed_tables():
    statistics = pd.read_csv(
        ROOT / "artifacts/predictions/source_model_three_seed_subject_statistics.csv.gz",
        compression="gzip",
    )
    protocol = pd.read_csv(ROOT / "artifacts/protocol/source_model_three_seed_protocol.csv")
    protocol_lookup = protocol.set_index(["source_seed", "source_model"])
    rows = []
    for keys, group in statistics.groupby(
        ["evaluation_scope", "source_seed", "model", "outcome"], sort=True
    ):
        scope, seed, model, outcome = keys
        spec = protocol_lookup.loc[(int(seed), str(model))]
        counts = group.n_windows.to_numpy(float)
        absolute = group.absolute_error_sum.to_numpy(float)
        signed = group.signed_error_sum.to_numpy(float)
        rows.append({
            "evaluation_scope": scope,
            "source_seed": int(seed),
            "model": str(model),
            "outcome": str(outcome),
            "subjects": group.subject_id.nunique(),
            "windows": int(counts.sum()),
            "window_mae": float(absolute.sum() / counts.sum()),
            "subject_mae": float(np.mean(absolute / counts)),
            "mean_error": float(signed.sum() / counts.sum()),
            "source_training_subjects": int(spec.training_subjects),
            "source_validation_subjects": int(spec.validation_subjects),
            "target_tuning": False,
        })
    metrics = pd.DataFrame(rows)
    summary = metrics.groupby(["evaluation_scope", "model", "outcome"], as_index=False).agg(
        source_seeds=("source_seed", "nunique"), subjects=("subjects", "first"), windows=("windows", "first"),
        window_mae_mean=("window_mae", "mean"), window_mae_sd=("window_mae", "std"),
        window_mae_min=("window_mae", "min"), window_mae_max=("window_mae", "max"),
        subject_mae_mean=("subject_mae", "mean"), subject_mae_sd=("subject_mae", "std"),
    )
    return metrics, summary


def compare(name, actual, keys, output_dir, tolerance=7e-5, reference_dir=SECONDARY):
    reference = pd.read_csv(reference_dir / name)
    missing = set(reference.columns) - set(actual.columns)
    if missing:
        raise AssertionError(f"{name}: missing columns {sorted(missing)}")
    actual = actual[reference.columns].copy()
    reference = reference.sort_values(keys).reset_index(drop=True)
    actual = actual.sort_values(keys).reset_index(drop=True)
    if len(reference) != len(actual):
        raise AssertionError(f"{name}: row-count mismatch")
    numeric = [column for column in reference.columns if pd.api.types.is_numeric_dtype(reference[column])]
    text = [column for column in reference.columns if column not in numeric]
    for column in text:
        if not reference[column].fillna("").astype(str).equals(actual[column].fillna("").astype(str)):
            raise AssertionError(f"{name}: text mismatch in {column}")
    max_delta = 0.0
    for column in numeric:
        delta = np.nanmax(np.abs(reference[column].to_numpy(float) - actual[column].to_numpy(float)))
        max_delta = max(max_delta, float(delta))
    if max_delta > tolerance:
        raise AssertionError(f"{name}: max numeric delta {max_delta} exceeds {tolerance}")
    actual.to_csv(output_dir / name, index=False, float_format="%.6f")
    return {"artifact": name, "rows": len(actual), "max_abs_delta": max_delta, "status": "PASS"}


def main():
    parser = argparse.ArgumentParser(description="Recompute released secondary analyses from prediction, retention, and bootstrap inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/reproduced/secondary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    primary = pd.read_csv(ROOT / "artifacts/predictions/pulsedb_vital_subject_grouped_oof_predictions.csv.gz", compression="gzip", low_memory=False)
    primary_all = pd.read_csv(ROOT / "artifacts/predictions/pulsedb_vital_target_formed_predictions.csv.gz", compression="gzip", low_memory=False)
    sqi_annotations = pd.read_csv(ROOT / "artifacts/quality/sqi_validation_annotations.csv.gz", compression="gzip", low_memory=False)
    abp_annotations = pd.read_csv(ROOT / "artifacts/quality/abp_reference_quality_annotations.csv.gz", compression="gzip", low_memory=False)
    pm = pd.read_csv(ROOT / "artifacts/predictions/pulsedb_mimic_zero_shot_predictions.csv.gz", compression="gzip", low_memory=False)
    mbp = pd.read_csv(ROOT / "artifacts/predictions/mimic_bp_zero_shot_predictions.csv.gz", compression="gzip", low_memory=False)
    mbp_audit = pd.read_csv(ROOT / "data/retention/mimic_bp_window_audit.csv.gz", compression="gzip", low_memory=False)
    primary_audit = pd.read_csv(ROOT / "data/retention/pulsedb_vital_window_audit.csv.gz", compression="gzip", low_memory=False)
    pm_audit = pd.read_csv(ROOT / "data/retention/pulsedb_mimic_window_audit.csv.gz", compression="gzip", low_memory=False)
    vital_raw = pd.read_csv(ROOT / "data/retention/pulsedb_vital_raw_candidate_manifest.csv.gz", compression="gzip", low_memory=False)
    pm_raw = pd.read_csv(ROOT / "data/retention/pulsedb_mimic_raw_candidate_manifest.csv.gz", compression="gzip", low_memory=False)
    mechanism = pd.read_csv(ROOT / "artifacts/predictions/mechanism_control_predictions.csv.gz", compression="gzip", low_memory=False)
    pat_metrics, pat_contrasts = pat_interaction_tables(primary, mechanism)
    source_seed_metrics, source_seed_summary = source_seed_tables()
    subgroup, definitions, sqi = subgroup_tables(primary)
    thresholds = external_threshold([("pulsedb_mimic", pm), ("mimic_bp", mbp)])
    main_bootstrap = cluster_bootstrap(mbp, "mimic_bp")
    bridge, protocol = mimic_bridge(mbp_audit, mbp, main_bootstrap)
    outputs = {
        "quality_rejection.csv": (quality_rejection(primary), ["coverage"]),
        "retention_sensitivity.csv": (retention_sensitivity(primary_all), ["analysis_view"]),
        "sqi_reference_validation.csv": (sqi_reference_summary(sqi_annotations), ["windows"]),
        "abp_reference_quality_validation.csv": (abp_reference_summary(abp_annotations, primary_audit), ["review_windows"]),
        "abp_reference_quality_sensitivity.csv": (abp_reference_sensitivity(primary, primary_audit), ["analysis_view", "model"]),
        "error_tail_composition.csv": (failure_attribution(primary), ["endpoint", "composition_category"]),
        "few_shot_calibration.csv": (few_shot(mbp), ["calibration_windows"]),
        "source_shift_projection.csv": (source_shift([("PulseDB-Vital", primary), ("PulseDB-MIMIC", pm), ("MIMIC-BP", mbp)]), ["cohort", "point_id"]),
        "external_zero_shot_summary.csv": (external_summary(thresholds), ["dataset", "model"]),
        "external_threshold_metrics.csv": (thresholds, ["cohort", "model"]),
        "representative_windows.csv": (representative_windows(primary), ["role"]),
        "figure_7_subgroup_source.csv": (subgroup, ["display_label"]),
        "subgroup_analysis.csv": (subgroup, ["display_label"]),
        "subgroup_definitions.csv": (definitions, ["display_label"]),
        "sqi_strata.csv": (sqi, ["model", "sqi_stratum"]),
        "mimic_bp_retention_bridge.csv": (bridge, ["analysis_subset"]),
        "mimic_bp_protocol_audit.csv": (protocol, ["analysis", "model"]),
        "selection_bias.csv": (selection_bias(primary_audit), ["group"]),
        "repository_scalar_sensitivity.csv": (pd.concat([repository_scalar_sensitivity("pulsedb_vital", primary, primary_audit), repository_scalar_sensitivity("pulsedb_mimic", pm, pm_audit)], ignore_index=True), ["cohort", "model", "outcome"]),
        "pat_stratified_model_comparison.csv": (pat_metrics, ["group_type", "pat_group", "model", "outcome"]),
        "pat_group_interaction_contrasts.csv": (pat_contrasts, ["comparator_model", "outcome", "comparison_group"]),
    }
    reports = []
    for name, (frame, keys) in outputs.items():
        reports.append(compare(name, frame, keys, args.output_dir))
    target_selection = pd.concat([target_formation_selection(vital_raw), target_formation_selection(pm_raw)], ignore_index=True)
    reports.append(compare("target_formation_selection_audit.csv", target_selection, ["cohort", "group"], args.output_dir, reference_dir=ROOT / "artifacts/cohorts"))
    reports.append(compare("source_model_three_seed_metrics.csv", source_seed_metrics, ["evaluation_scope", "source_seed", "model", "outcome"], args.output_dir, reference_dir=ROOT / "artifacts/metrics/external"))
    reports.append(compare("source_model_three_seed_summary.csv", source_seed_summary, ["evaluation_scope", "model", "outcome"], args.output_dir, reference_dir=ROOT / "artifacts/metrics/external"))
    report = {"status": "PASS", "analyses": len(reports), "max_abs_delta": max(row["max_abs_delta"] for row in reports), "artifacts": reports}
    (args.output_dir.parent / "secondary_analysis_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "analyses": len(reports), "max_abs_delta": report["max_abs_delta"]}, indent=2))


if __name__ == "__main__":
    main()
