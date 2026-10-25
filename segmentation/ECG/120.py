#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 120 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 120.py   # -> output/120.pdf, output/120_segments.csv
"""
import core

# User note (same as 119, ECG2): abnormal beats (inverted QRS, PVC-like) were treated as having no QRS at sharp_raw<3.0 and merged into normal beats,
# producing over-long segments. With SHARP_QRS_MIN=2.5 the PVC is picked up and split, separating the inverted-QRS beats (n=5) into c1, K=2 (2026-06-26).
PARAMS = {"SHARP_QRS_MIN": 2.5, "KMAX_S1": 2, "MIN_PATTERN": 3, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("120", PARAMS)
    if r:
        print(f"120: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
