#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global-reference 1-NN anomaly score + label validation (Power_Demand prototype)
================================================================
Purpose: score test units not by nearest per-series cluster medoid (~4), but by
     1-NN SBD to all pooled normal units across series (global reference ~1000), and
     validate whether label anomalies (test span [anom_s,anom_e]) can be detected (AUC-ROC / AUC-PR).

Comparison:
  - global_nn : test unit -> minimum SBD to all units of the global normal pool (1-NN)
  - series_nn : test unit -> minimum SBD to normal-pool units of the same series (within-series 1-NN)
  - medoid    : test unit -> minimum SBD to the same series' cluster medoids (the current-method baseline)

Normal pool = shapes of the shape-clean units of each series' train (UCR=normal) (robust z-norm -> 256 resample).
"""
import numpy as np
import core as W
from sklearn.metrics import roc_auc_score, average_precision_score

RESAMPLE = W.RESAMPLE_LEN


def unit_shape(seg):
    return W.resample(W.robust_znorm(seg))


def collect():
    """Segment all series and collect (a) global normal pool shapes, (b) each series' medoids,
    (c) all test unit shapes + anomaly labels + series IDs."""
    pool = []                 # global normal pool (clean train units across all series)
    pool_series = []          # series ID of each pool unit
    series_pool = {}          # series ID -> [shape,...] (within-series pool)
    series_medoids = {}       # series ID -> [medoid shape,...]
    test_units = []           # (sid, shape, is_anom)
    for sid in W.domain_series_ids():
        path = W.find_file(sid)
        x = np.loadtxt(path).astype(float)
        train_end, anom_s, anom_e = W.parse_meta(path)
        xt = x[:train_end] if train_end > 200 else x
        period = W.PERIOD if W.PERIOD else W.dominant_period(xt)
        # --- train normal pool (shape-clean units) ---
        spans, clean, _ = W.segment_cycles(xt, period)
        ci = [i for i, c in enumerate(clean) if c]
        beats = [xt[s:e + 1] for s, e in spans]
        shapes = np.array([unit_shape(beats[i]) for i in ci]) if ci else np.zeros((0, RESAMPLE))
        series_pool[sid] = shapes
        for s in shapes:
            pool.append(s); pool_series.append(sid)
        # --- medoids of the current method (pulse+artifact medoids of the two-stage clustering) ---
        if len(ci) >= 3:
            lab, K, info, _ = W.cluster_two_stage([beats[i] for i in ci], shapes)
            meds = []
            for c in sorted(set(int(l) for l in lab)):
                idx = np.where(lab == c)[0]
                sub = shapes[idx]
                meds.append(sub[int(W.sbd_full_matrix(sub).sum(1).argmin())] if len(idx) > 1 else sub[0])
            series_medoids[sid] = np.array(meds)
        else:
            series_medoids[sid] = shapes if len(shapes) else np.zeros((1, RESAMPLE))
        # --- test units + anomaly labels ---
        if 200 < train_end < len(x):
            xtt = x[train_end:].astype(float)
            ts, _tc, _ = W.segment_cycles(xtt, period)
            for (s, e) in ts:
                sh = unit_shape(xtt[s:e + 1])
                gs, ge = s + train_end, e + train_end       # global coordinates
                is_anom = int(not (ge < anom_s or gs > anom_e))  # whether it overlaps the anomaly span
                test_units.append((sid, sh, is_anom))
    return (np.array(pool), np.array(pool_series), series_pool, series_medoids, test_units)


def min_sbd(shape, refs):
    if len(refs) == 0:
        return 1.0
    return min(W.sbd_full_pair(shape, r) for r in refs)


def main():
    pool, pool_series, series_pool, series_medoids, test_units = collect()
    print(f"global normal pool: {len(pool)} units ({len(set(pool_series))} series total)")
    print(f"test units: {len(test_units)} (anomaly {sum(t[2] for t in test_units)} / normal {sum(1 - t[2] for t in test_units)})")

    y, s_glob, s_ser, s_med = [], [], [], []
    for sid, sh, is_anom in test_units:
        y.append(is_anom)
        s_glob.append(min_sbd(sh, pool))
        s_ser.append(min_sbd(sh, series_pool[sid]))
        s_med.append(min_sbd(sh, series_medoids[sid]))
    y = np.array(y)

    print("\n=== detection performance (score = SBD distance to nearest normal unit; larger = more anomalous) ===")
    print(f"{'method':<28}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for name, sc in [("global_nn (global pool 1-NN)", s_glob),
                     ("series_nn (within-series 1-NN)", s_ser),
                     ("medoid (current: nearest series medoid)", s_med)]:
        sc = np.array(sc)
        roc = roc_auc_score(y, sc) if 0 < y.sum() < len(y) else float("nan")
        pr = average_precision_score(y, sc) if 0 < y.sum() < len(y) else float("nan")
        print(f"{name:<28}{roc:>10.3f}{pr:>10.3f}")
    print(f"\n(baseline PR = anomaly ratio = {y.mean():.3f})")


if __name__ == "__main__":
    main()
