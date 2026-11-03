"""Open-set evaluation measures."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def novelty_auroc(known_best_sim: np.ndarray, unseen_best_sim: np.ndarray) -> float:
    """AUROC of the novelty score (1 - best similarity) for unseen vs known incidents."""
    if len(known_best_sim) == 0 or len(unseen_best_sim) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(known_best_sim)), np.ones(len(unseen_best_sim))]
    score = 1.0 - np.r_[known_best_sim, unseen_best_sim]
    return float(roc_auc_score(y, score))


def oscr(known_best_sim: np.ndarray, known_correct: np.ndarray, unseen_best_sim: np.ndarray) -> float:
    """Open-Set Classification Rate: area under correct-classification-rate vs false-positive-rate.

    CCR(t) = fraction of known incidents accepted (sim >= t) AND correctly typed;
    FPR(t) = fraction of unseen incidents accepted. Swept over all thresholds.
    """
    if len(known_best_sim) == 0 or len(unseen_best_sim) == 0:
        return float("nan")
    thresholds = np.unique(np.r_[known_best_sim, unseen_best_sim, np.inf])[::-1]
    ccr = [float(((known_best_sim >= t) & known_correct).mean()) for t in thresholds]
    fpr = [float((unseen_best_sim >= t).mean()) for t in thresholds]
    ccr = np.r_[0.0, ccr]
    fpr = np.r_[0.0, fpr]
    return float(np.trapezoid(ccr, fpr) if hasattr(np, "trapezoid") else np.trapz(ccr, fpr))
