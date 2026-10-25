"""Evaluation metrics.

Point-wise metrics are standard scikit-learn; Range/VUS metrics come from the
vendored upstream VUS code.  The computation mirrors the baseline evaluator so
the numbers are directly comparable, and the VUS window uses
``subsequence_length=None`` (mean anomaly length) to match the baselines.
"""
import numpy as np
from sklearn import metrics as skm

from .vus import get_range_vus_roc


def point_wise(labels, scores):
    """Best-F1 point-wise metrics over the precision-recall curve."""
    precision, recall, threshold = skm.precision_recall_curve(y_true=labels, y_score=scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    i = int(np.argmax(f1))
    thr = threshold.astype(np.float64)
    return {
        "f1": float(np.max(f1)),
        "auc_pr": float(skm.auc(recall, precision)),
        "auc_roc": float(skm.roc_auc_score(y_true=labels, y_score=scores)),
        "precision": float(precision[i]),
        "recall": float(recall[i]),
        "threshold": float(thr[min(i, len(thr) - 1)]),
    }


def evaluate(labels, scores, with_vus=True):
    """Return a flat dict of point-wise (+ optional Range/VUS) metrics."""
    labels = np.asarray(labels).astype(float)
    scores = np.asarray(scores).astype(float)
    out = dict(point_wise(labels, scores))
    if with_vus:
        v = get_range_vus_roc(score=scores, labels=labels, subsequence_length=None)
        out.update({k.lower(): float(val) for k, val in v.items()})
    return out
