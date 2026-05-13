"""
ROC (image-level) and summary stats from detection scores / AP precision-recall curves.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
except ImportError as exc:  # pragma: no cover
    raise ImportError("scikit-learn required for detection_curves") from exc


def image_level_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Binary labels and max-detection scores per image. Returns AUC, fpr, tpr, thresholds."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    roc_auc = float(auc(fpr, tpr))
    return roc_auc, fpr, tpr, thr


def max_f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> Tuple[float, int]:
    p = np.asarray(precision, dtype=np.float64)
    r = np.asarray(recall, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    j = int(np.nanargmax(f1))
    return float(f1[j]), j


def precision_recall_curve_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return precision_recall_curve(y_true, y_score)
