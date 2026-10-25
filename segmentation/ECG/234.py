#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 234 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 234.py   # -> output/234.pdf, output/234_segments.csv
"""
import core

# User note: redo clustering. mit14157longtermecg (same DB as 233). The RR detector's over-detection gave period=77 -> QRS_SHARP_DETECT +
# PERIOD=115 (RR median) gives one segment per beat (2QRS=0). c2 (deep downward QRS, SBD 0.665 = a clearly distinct type/PVC) and c1 (deep dip,
# SBD 0.231 = borderline), with n=5-8<MIN_PATTERN=8, were absorbed to K=1. MIN_PATTERN=3 retains them. K=3 {c0:150 main type, c1:8 deep dip,
# c2:5 deep downward QRS}.
# Additional user note: c1 (deep-dip) should grow more (also 5750/18500). deep-dip is continuous with main and silhouette doesn't separate them
# (even with KMAX_S1=3, K_s1=2). FORCE_SHAPE_K=3 bypasses silhouette and fixes shape-complete to K=3 -> deep-dip c1=9 (incl. 18500).
# (Raising K splits and mixes main, so 3 is clean) (2026-06-27).
PARAMS = {"PERIOD": 115, "QRS_SHARP_DETECT": True, "FORCE_SHAPE_K": 3, "MIN_PATTERN": 3}

if __name__ == "__main__":
    r = core.run("234", PARAMS)
    if r:
        print(f"234: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
