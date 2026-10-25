#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-processing version that, after segmenting series 015 (ECG) with core.process,
"splits evenly by the same basis" for clusters that exceed 224 samples and
assigns a new cluster ID to each sub-segment.

Design
----
- Run the usual Stage1+Stage2 clustering in core (= obtain both train + test segments)
- For each pulse cluster c (c < K_pulse):
    n_split_c = ceil(median_L_c / TARGET_L)   (TARGET_L=224)
  if it exceeds 1, split all segments of that cluster evenly into n_split_c pieces
- Assign a new cluster ID to each sub-segment (= "increase the cluster count")
  cluster c, sub 0 -> new cluster (== the leading number of c)
  cluster c, sub 1 -> new cluster
  ...
- artifact clusters (>= K_pulse_orig) are left as-is (= not split)
- test segments are also split at the same positions by the same n_split_c and assigned to the
  new cluster (per sub position) corresponding to the original cluster c

Output
----
  output/015_split224_segments.csv   (= same schema as regular 015)
  output/015_split224.pdf            (= visualized by core.plot)
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import core   # type: ignore


SID = "015"
TARGET_L = 224
PARAMS = {"PERIOD": 470, "SINGLE_FLOOR": 0.16}


def split_equal(span: tuple[int, int], n: int) -> list[tuple[int, int]]:
    """Split [s, e) into n equal-length pieces (last absorbs remainder)."""
    s, e = span
    L = e - s
    base = L // n
    out = []
    for j in range(n):
        sub_s = s + j * base
        sub_e = s + (j + 1) * base if j < n - 1 else e
        out.append((sub_s, sub_e))
    return out


# Equal-thirds split:
# 015's R-peak is in the latter half of the beat (rel=0.45..0.73), so with an R-peak anchor +
# a 224 cap, sub 0 would just carve out the "first 224 samples of the beat", and sub 0's
# shape varies from beat to beat (within-cluster mean SBD worsens from 0.21 to 0.41).
# An even 3-way split cuts every beat at the same relative position (1/3, 2/3), so the
# "same basis" holds and cluster consistency is preserved.
# Guarantee: when L <= 3 * TARGET_L (= 672), each sub <= TARGET_L (015 max=607, ok).
def split_rpeak_3(span: tuple[int, int], x_signal: np.ndarray,
                          target: int = TARGET_L) -> list[tuple[int, int]]:
    """Equal 3-way split (function name kept for backward compat).
    x_signal is unused but kept in signature."""
    return split_equal(span, 3)


def main():
    # ---- Run core.process with PARAMS, save=False (we'll write our own files) ----
    g = vars(core)
    saved = {k: g[k] for k in core.TUNABLE if k in g}
    try:
        for k, v in PARAMS.items():
            if k not in core.TUNABLE:
                raise KeyError(f"unknown PARAM '{k}'. valid: {core.TUNABLE}")
            setattr(core, k, v)
        r = core.process(SID, save=False)
    finally:
        for k, v in saved.items():
            setattr(core, k, v)

    if r is None:
        print(f"[{SID}] core.process failed.")
        return

    segs_train = [s for s in r["segments"] if s.get("region", "train") == "train"]
    segs_test  = [s for s in r["segments"] if s.get("region", "train") == "test"]
    K_pulse = int(r["info"]["K_pulse"])

    # ---- Per-cluster median length over CLEAN train segments (cluster < K_pulse) ----
    cluster_lengths: dict[int, list[int]] = defaultdict(list)
    for s in segs_train:
        c = int(s["cluster"])
        L = int(s["end"]) - int(s["start"])
        if 0 <= c < K_pulse:
            cluster_lengths[c].append(L)
    # n_split[c] = 3 if median_L > TARGET_L else 1, fixing the cluster-level N
    # (sub-clusters = pre-peak / peak / post-peak). Per-segment N would differ
    # by beat length and break the "same basis" the user asked for.
    n_split: dict[int, int] = {}
    for c in range(K_pulse):
        if not cluster_lengths.get(c):
            n_split[c] = 1
            continue
        med = int(np.median(cluster_lengths[c]))
        n_split[c] = 3 if med > TARGET_L else 1

    print(f"[{SID}] K_pulse={K_pulse}, TARGET_L={TARGET_L}")
    for c in range(K_pulse):
        n = len(cluster_lengths.get(c, []))
        med = int(np.median(cluster_lengths[c])) if cluster_lengths.get(c) else 0
        print(f"  pulse cluster {c}: n_clean={n}, median_L={med}, "
              f"n_split={n_split[c]}, new sub-clusters: "
              f"{n_split[c]} (split: {n_split[c] > 1})")

    # ---- Build mapping: original cluster c, sub_idx -> new cluster ID ----
    new_id_of = {}
    next_new = 0
    for c in range(K_pulse):
        for j in range(n_split[c]):
            new_id_of[(c, j)] = next_new
            next_new += 1
    new_K_pulse = next_new
    # artifact clusters keep relative offset (= shifted by new_K_pulse)
    artifact_offset = new_K_pulse - K_pulse

    # Load the smoothed train signal early so split_rpeak_3 can locate R-peaks.
    path = core.find_file(SID)
    x_full = np.loadtxt(path).astype(float)
    train_end, _, _ = core.parse_meta(path)
    xt = x_full[:train_end]
    if core.PRE_SMOOTH_WN > 0:
        from scipy.signal import butter, filtfilt
        bb, aa = butter(3, core.PRE_SMOOTH_WN)
        xt_smooth = filtfilt(bb, aa, xt)
    else:
        xt_smooth = xt

    # ---- Re-emit segments (train) using R-peak anchored 3-way split ----
    new_train: list[tuple[int, int, int, int]] = []   # (start, end, cluster, clean)
    for s in segs_train:
        c = int(s["cluster"])
        span = (int(s["start"]), int(s["end"]))
        if c < 0:
            new_train.append((*span, c, 0))
            continue
        if c < K_pulse:
            n = n_split[c]
            if n == 3:
                sub_spans = split_rpeak_3(span, xt_smooth)
            else:
                sub_spans = [span]
            for j, (sub_s, sub_e) in enumerate(sub_spans):
                new_c = new_id_of[(c, j)]
                new_train.append((sub_s, sub_e, new_c, 1))   # clean=1
        else:
            new_c = c + artifact_offset
            new_train.append((*span, new_c, 0))

    # ---- Compute medoid waveform per NEW pulse cluster (= sub-position) ----
    # xt / xt_smooth already loaded above for split_rpeak_3.
    members_by_newc: dict[int, list[np.ndarray]] = defaultdict(list)
    for sub_s, sub_e, new_c, clean_flag in new_train:
        if clean_flag != 1:
            continue
        wave = xt_smooth[sub_s: sub_e + 1] if sub_e + 1 <= len(xt_smooth) else xt_smooth[sub_s:]
        if len(wave) < 2:
            continue
        shape_rz = core.resample(core.robust_znorm(wave))
        members_by_newc[new_c].append(shape_rz)
    new_pulse_medoids: dict[int, np.ndarray] = {}
    for new_c, members in members_by_newc.items():
        arr = np.array(members)
        if len(arr) == 1:
            new_pulse_medoids[new_c] = arr[0]
        else:
            D = core.sbd_full_matrix(arr)
            new_pulse_medoids[new_c] = arr[D.sum(1).argmin()]
    print(f"  new pulse medoids: {sorted(new_pulse_medoids.keys())}")

    # ---- Re-emit segments (test): split to <= TARGET_L, then assign nearest ----
    # core.process appended train_end so test span coordinates are absolute in x_full.
    if core.PRE_SMOOTH_WN > 0:
        from scipy.signal import butter, filtfilt
        bb2, aa2 = butter(3, core.PRE_SMOOTH_WN)
        x_test_smooth = filtfilt(bb2, aa2, x_full[train_end:])
    else:
        x_test_smooth = x_full[train_end:]
    x_smooth_full = np.concatenate([xt_smooth, x_test_smooth])

    new_test: list[tuple[int, int, int, int]] = []
    for s in segs_test:
        span = (int(s["start"]), int(s["end"]))
        L = span[1] - span[0]
        # Decide n_split based on length, then use R-peak anchor when n==3.
        # Use the same 3-way R-peak anchor as train so each test beat splits
        # at its own peak (= same basis). For beats already <=TARGET_L, no split.
        if L > TARGET_L:
            sub_spans = split_rpeak_3(span, x_smooth_full)
        else:
            sub_spans = [span]
        for sub_s, sub_e in sub_spans:
            seg_wave = x_smooth_full[sub_s: sub_e + 1] if sub_e + 1 <= len(x_smooth_full) \
                              else x_smooth_full[sub_s:]
            if len(seg_wave) < 2:
                new_test.append((sub_s, sub_e, -1, 0))
                continue
            shape_rz = core.resample(core.robust_znorm(seg_wave))
            dists = {nc: float(core.sbd_full_pair(shape_rz, m))
                       for nc, m in new_pulse_medoids.items()}
            nearest = min(dists, key=dists.get)
            new_test.append((sub_s, sub_e, nearest, 1))

    # ---- Write CSV ----
    os.makedirs(core.OUT_DIR, exist_ok=True)
    out_csv = os.path.join(core.OUT_DIR, f"{SID}_split224_segments.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seg_index", "start", "end", "cluster", "clean", "region"])
        idx = 0
        for s_, e_, c_, cl_ in new_train:
            w.writerow([idx, s_, e_, c_, cl_, "train"]); idx += 1
        for s_, e_, c_, cl_ in new_test:
            w.writerow([idx, s_, e_, c_, cl_, "test"]); idx += 1
    print(f"  Wrote: {out_csv}")
    print(f"  train segments: {len(segs_train)} -> {len(new_train)}")
    print(f"  test  segments: {len(segs_test)}  -> {len(new_test)}")
    print(f"  K_pulse: {K_pulse} -> {new_K_pulse}")

    train_Ls = [e - s for s, e, _, _ in new_train]
    if train_Ls:
        print(f"  new train L: min={min(train_Ls)}, median={int(np.median(train_Ls))}, "
              f"max={max(train_Ls)}  (target<={TARGET_L})")
    test_Ls = [e - s for s, e, _, _ in new_test]
    if test_Ls:
        print(f"  new test  L: min={min(test_Ls)}, median={int(np.median(test_Ls))}, "
              f"max={max(test_Ls)}")

    # ---- Generate the PDF via core.plot (uses the new spans / labels) ----
    try:
        # Re-load x for plotting (core.process loaded it but didn't return).
        path = core.find_file(SID)
        x = np.loadtxt(path).astype(float)
        train_end, anom_s, anom_e = core.parse_meta(path)
        xt = x[:train_end] if train_end > 200 else x
        if core.PRE_SMOOTH_WN > 0:
            from scipy.signal import butter, filtfilt
            bb, aa = butter(3, core.PRE_SMOOTH_WN)
            xt_smooth = filtfilt(bb, aa, xt)
        else:
            xt_smooth = xt

        train_spans_arr = [(s_, e_) for s_, e_, _, _ in new_train]
        train_labels = np.array([c_ for _, _, c_, _ in new_train], dtype=int)
        test_spans = [(s_, e_) for s_, e_, _, _ in new_test]
        test_labels = [int(c_) for _, _, c_, _ in new_test]

        # Re-compute shapes for the new train spans (for the overlay panel).
        shapes_plot = np.array([
            core.znorm(core.resample(xt[s_:e_ + 1])) for s_, e_ in train_spans_arr
        ])

        x_disp = xt_smooth
        if 200 < train_end < len(x):
            x_test_raw = x[train_end:].astype(float)
            if core.PRE_SMOOTH_WN > 0:
                xtt = filtfilt(bb, aa, x_test_raw)
            else:
                xtt = x_test_raw
            x_disp = np.concatenate([xt_smooth, xtt])

        # Adjust output path for the PDF: temporarily redirect OUT_DIR-relative name.
        # core.plot uses f"{series_id}.pdf"; we want f"{series_id}_split224.pdf".
        # Patch by symlinking series_id locally.
        import matplotlib
        matplotlib.use("Agg")
        core.plot(SID + "_split224", x_disp, train_spans_arr, train_labels,
                   new_K_pulse + (max(c for _,_,c,_ in new_train) + 1 - new_K_pulse
                                     if any(c >= new_K_pulse for _,_,c,_ in new_train)
                                     else new_K_pulse),
                   anom_s, anom_e, shapes_plot, train_end,
                   k_pulse=new_K_pulse,
                   test_spans=test_spans, test_labels=test_labels)
        print(f"  Plot saved (via core.plot with id='{SID}_split224')")
    except Exception as e:
        print(f"  PDF generation skipped: {e}")


if __name__ == "__main__":
    main()
