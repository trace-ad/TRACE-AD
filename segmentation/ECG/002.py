#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 002 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 002.py   # -> output/002.pdf, output/002_segments.csv
"""
import core

# User fix (same policy as 001, sddb40 family; shape changed from QRS-fragment merging, so MIN_PATTERN re-tuned 2026-06-26):
# c0 normal + c1/c2 biphasic 2 types, K=3.
PARAMS = {"SHARP_QRS_MIN": 2.0, "KMAX_S1": 3, "MIN_PATTERN": 4, "SIL_MIN": 0.10, "ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("002", PARAMS)
    if r:
        print(f"002: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
