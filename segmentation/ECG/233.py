#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 233 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 233.py   # -> output/233.pdf, output/233_segments.csv
"""
import core

# User note: resolve c1 by joining it to adjacent segments. mit14157longtermecg. Regular up-spike + down-dip beats (period ~113).
# detect_rpeaks also over-detected the small bump after a deep down-dip beat as an R-wave, giving period=77, and the extra cut produced short flat fragments (28-78 samples)
# = c1/c2. QRS_SHARP_DETECT + PERIOD=113 detects only real QRS -> the fragments disappear.
# User note: K=2. The deep down-dip beats (@3410, 16462) are a distinct type with a deeper dip than the main type (2-way silhouette 0.874), but n=2<MIN_PATTERN=8
# absorbs them to K=1. MIN_PATTERN=2 retains the 2nd type. K=2 {c0:166 main type, c1:2 deep-dip type} (2026-06-27).
PARAMS = {"PERIOD": 113, "QRS_SHARP_DETECT": True, "MIN_PATTERN": 2}

if __name__ == "__main__":
    r = core.run("233", PARAMS)
    if r:
        print(f"233: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
