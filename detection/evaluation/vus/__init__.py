"""Minimal vendored VUS metrics (Range-AUC / VUS-ROC / VUS-PR).

Vendored from the VUS project (Paparrizos et al., "Volume Under the Surface",
PVLDB 2022; https://github.com/TheDatumOrg/VUS).  Kept faithful so numbers match
the published baselines; only the metric path is retained.
"""
import numpy as np

from .metricor import metricor, generate_curve


def get_list_anomaly(labels):
    """Lengths of the contiguous anomaly runs in a 0/1 label array."""
    results = []
    start = 0
    anom = False
    for i, val in enumerate(labels):
        if val == 1:
            anom = True
        else:
            if anom:
                results.append(i - start)
                anom = False
        if not anom:
            start = i
    return results


def get_range_vus_roc(score, labels, subsequence_length=None):
    """Range-AUC and VUS.

    ``subsequence_length=None`` (the baseline setting) fixes the tolerance window
    to the mean labelled-anomaly length of the series.
    """
    grader = metricor()
    if subsequence_length is not None:
        sliding = int(np.mean(get_list_anomaly(labels)))
        margin = max(sliding, subsequence_length)
    else:
        margin = int(np.mean(get_list_anomaly(labels)))
    R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(
        labels=labels, score=score, window=margin, plot_ROC=True)
    _, _, _, _, _, _, VUS_ROC, VUS_PR = generate_curve(labels, score, 2 * margin)
    return {"R_AUC_ROC": R_AUC_ROC, "R_AUC_PR": R_AUC_PR,
            "VUS_ROC": VUS_ROC, "VUS_PR": VUS_PR}
