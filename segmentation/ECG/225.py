#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 225 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 225.py   # -> output/225.pdf, output/225_segments.csv
"""
import core

# User note (multiple rounds): same as 224 -> further split 52900 into 2 / consolidate singletons at 79250+ / split c1 into 2. mit14046longtermecg (period 97).
# QRS_SHARP_DETECT + QRS_HEIGHT=0.05 + PEAK_DIST_FRAC=0.45: also detect PVC sharp deflections + weak/close normal QRS (merge resolved; detecting the normal QRS at 52885
# splits 52900 into PVC/normal). SINGLE_FLOOR=0.18: absorb the noisy normal beats at 79xxx (SBD~0.15 same-shape but rough) into c0.
# No KMAX = by user judgment split the PVCs into 2 types. However, the Stage2 (D3) split leans on amplitude and is shape-inconsistent (c1 mixes into c2).
# By user judgment, re-cut on a shape basis + reconsider the number of types (silhouette best at 4 splits). PVC_SHAPE_K=3 re-splits with all-shift SBD complete
# -> 4 types. User confirmation + evaluation AI found c1 (sharp) / c2 (wide) are same-shape at medoid SBD 0.13 = over-segmentation. PVC_MERGE_TAU=0.22
# merges the same-shape PVCs into the main type. K=4 {c0:793 normal, c1:48 main PVC, c2:5 initial-hump type (3750/28000 etc.), c3:2 biphasic (+65/-22)} (2026-06-27).
PARAMS = {"QRS_SHARP_DETECT": True, "QRS_HEIGHT": 0.05, "PEAK_DIST_FRAC": 0.45, "SINGLE_FLOOR": 0.18, "PVC_SHAPE_K": 4, "PVC_MERGE_TAU": 0.22}

if __name__ == "__main__":
    r = core.run("225", PARAMS)
    if r:
        print(f"225: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
