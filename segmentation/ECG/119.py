#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 119 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 119.py   # -> output/119.pdf, output/119_segments.csv
"""
import core

# User note: ECG1. The abnormal beat at 9250 (PVC-like, deep negative dip) was treated as having no QRS at sharp_raw=2.88<3.0 and merged into a normal beat
# . With SHARP_QRS_MIN=2.5 the PVC is picked up and 9250 is split into 2 beats (T-waves at 1.5-2.4 are still rejected). The inverted-QRS beats (n=3) are
# separated into c1, giving K=2 (c0:107 normal + c1:3 inverted) (2026-06-26).
PARAMS = {"SHARP_QRS_MIN": 2.5, "KMAX_S1": 2, "MIN_PATTERN": 3, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("119", PARAMS)
    if r:
        print(f"119: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
