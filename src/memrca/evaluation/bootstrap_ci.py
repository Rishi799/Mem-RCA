"""Bootstrap statistical confidence intervals for RCA evaluation metrics.

Provides rigorous confidence intervals (95% CI) and standard errors over cases,
overcoming the limitation of small-N repetition counts.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def bootstrap_metric_ci(
    ranks: List[int],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute mean and (lower, upper) bootstrap confidence interval for Hit@1, Hit@3, and MRR.

    Returns:
        Dict mapping metric name to (mean, ci_lower, ci_upper)
    """
    if not ranks:
        return {
            "Hit@1": (0.0, 0.0, 0.0),
            "Hit@3": (0.0, 0.0, 0.0),
            "MRR": (0.0, 0.0, 0.0),
        }

    ranks_arr = np.array(ranks, dtype=int)
    n = len(ranks_arr)
    rng = np.random.default_rng(seed)

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100
    upper_pct = (1.0 - alpha) * 100

    boot_indices = rng.integers(0, n, size=(n_bootstraps, n))

    boot_ranks = ranks_arr[boot_indices]
    boot_hit1 = np.mean(boot_ranks <= 1, axis=1)
    boot_hit3 = np.mean(boot_ranks <= 3, axis=1)
    boot_mrr = np.mean(1.0 / boot_ranks, axis=1)

    return {
        "Hit@1": (
            float(np.mean(ranks_arr <= 1)),
            float(np.percentile(boot_hit1, lower_pct)),
            float(np.percentile(boot_hit1, upper_pct)),
        ),
        "Hit@3": (
            float(np.mean(ranks_arr <= 3)),
            float(np.percentile(boot_hit3, lower_pct)),
            float(np.percentile(boot_hit3, upper_pct)),
        ),
        "MRR": (
            float(np.mean(1.0 / ranks_arr)),
            float(np.percentile(boot_mrr, lower_pct)),
            float(np.percentile(boot_mrr, upper_pct)),
        ),
    }


def evaluate_with_confidence_intervals(
    predictions_df: pd.DataFrame,
    group_cols: List[str] = ["system", "fold", "split"],
    n_bootstraps: int = 1000,
) -> pd.DataFrame:
    """Evaluate RCA predictions with 95% bootstrap confidence intervals per group."""
    records = []
    for keys, group in predictions_df.groupby(group_cols):
        ranks = group["predicted_rank"].dropna().astype(int).tolist()
        ci_stats = bootstrap_metric_ci(ranks, n_bootstraps=n_bootstraps)

        rec = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        rec["sample_size"] = len(ranks)
        rec["Hit@1_mean"] = round(ci_stats["Hit@1"][0], 4)
        rec["Hit@1_95CI"] = f"[{ci_stats['Hit@1'][1]:.3f}, {ci_stats['Hit@1'][2]:.3f}]"
        rec["Hit@3_mean"] = round(ci_stats["Hit@3"][0], 4)
        rec["Hit@3_95CI"] = f"[{ci_stats['Hit@3'][1]:.3f}, {ci_stats['Hit@3'][2]:.3f}]"
        rec["MRR_mean"] = round(ci_stats["MRR"][0], 4)
        rec["MRR_95CI"] = f"[{ci_stats['MRR'][1]:.3f}, {ci_stats['MRR'][2]:.3f}]"
        records.append(rec)

    return pd.DataFrame(records)
