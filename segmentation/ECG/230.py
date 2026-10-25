#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 230 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 230.py   # -> output/230.pdf, output/230_segments.csv
"""
import core

# User note: both segmentation and clustering aren't working. mit14134longtermecg (same DB as 228/229). Sharp QRS (~122 interval).
# The RR detector false-detected in-beat bumps and under-estimated period=71 -> segmentation broke, and KMAX_S1=1 forced K=1. QRS_SHARP_DETECT +
# PERIOD=122 (RR median) gives one segment per beat (2QRS=0, 0QRS=0, n_seg=155). The clusters are "downward QRS vs upward QRS".
# KMAX_S2=1 (suppress c0's amplitude sub-split). K=2 {c0:114 downward QRS, c1:38 upward QRS} (2026-06-27).
PARAMS = {"PERIOD": 122, "QRS_SHARP_DETECT": True, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("230", PARAMS)
    if r:
        print(f"230: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
