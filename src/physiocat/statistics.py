from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import subject_errors


def holm_adjust(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def rank_biserial(diff):
    diff = np.asarray(diff)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    ranks = stats.rankdata(np.abs(diff))
    return float((ranks[diff > 0].sum() - ranks[diff < 0].sum()) / ranks.sum())


def paired_tests(frame, comparators, cohort, reps=2000):
    rng = np.random.default_rng(4271 + len(frame))
    rows = []
    for comparator in comparators:
        for outcome in ("sbp", "dbp"):
            ref = subject_errors(frame, "physiocat", outcome)
            cmp_ = subject_errors(frame, comparator, outcome)
            common = ref.index.intersection(cmp_.index)
            diff = (cmp_.loc[common] - ref.loc[common]).to_numpy()
            test = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="approx")
            draws = rng.integers(0, len(diff), size=(reps, len(diff)))
            boot = diff[draws].mean(axis=1)
            rows.append({
                "cohort": cohort, "outcome": outcome.upper(), "reference_model": "physiocat", "comparator_model": comparator,
                "n_subject_pairs": len(diff), "mean_mae_reduction_mmHg": float(diff.mean()),
                "cluster_bootstrap_ci_low": float(np.quantile(boot, 0.025)), "cluster_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
                "wilcoxon_statistic": float(test.statistic), "raw_p": max(float(test.pvalue), np.finfo(float).tiny),
                "rank_biserial_effect": rank_biserial(diff),
            })
    return pd.DataFrame(rows)
