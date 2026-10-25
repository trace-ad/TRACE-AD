#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 207 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 207.py   # -> output/207.pdf, output/207_segments.csv
"""
import core

# User note: CHARIS repeats humps (pulses, period ~37). The RR detector picked up sub-peaks within the hump (period 20) and over-segmented.
# The strongest autocorrelation peak = 37 is the true hump period. With PERIOD=37, one hump = one segment. The humps are smooth and have no QRS, so
# SHARP_QRS_MIN=1.0 (relax the QRS requirement). All humps same shape, K=1 (2026-06-26).
PARAMS = {"PERIOD": 37, "SHARP_QRS_MIN": 1.0}

if __name__ == "__main__":
    r = core.run("207", PARAMS)
    if r:
        print(f"207: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
