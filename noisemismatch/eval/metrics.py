"""Evaluation metrics: AUC, pAUC and bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (anomalous = positive, higher score = anomalous).

    Args:
        labels: 1 for anomalous, 0 for normal clips.
        scores: Anomaly scores.

    Returns:
        The AUC.
    """
    return float(roc_auc_score(labels, scores))


def pauc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Partial AUC over FPR in [0, max_fpr], McClish-standardized (scikit-learn).

    This is what the DCASE evaluators compute: 0.5 at chance level, 1.0 for a
    perfect detector (see docs/protocol.md, "Metrics").

    Args:
        labels: 1 for anomalous, 0 for normal clips.
        scores: Anomaly scores.
        max_fpr: Upper FPR bound.

    Returns:
        The standardized partial AUC.
    """
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile confidence interval of the AUC, stratified by label.

    Normal and anomalous clips are resampled with replacement separately, keeping
    their counts, so every resample has both classes.

    Args:
        labels: 1 for anomalous, 0 for normal clips.
        scores: Anomaly scores.
        n_resamples: Number of bootstrap resamples.
        confidence: Coverage of the interval (e.g. 0.95).
        seed: Seed of the resampling.

    Returns:
        ``(low, high)`` bounds.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    positives = np.flatnonzero(labels == 1)
    negatives = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for k in range(n_resamples):
        pos = rng.choice(positives, size=len(positives), replace=True)
        neg = rng.choice(negatives, size=len(negatives), replace=True)
        idx = np.concatenate([neg, pos])
        values[k] = roc_auc_score(labels[idx], scores[idx])
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(values, [alpha, 1.0 - alpha])
    return float(low), float(high)
