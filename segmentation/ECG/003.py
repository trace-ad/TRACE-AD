#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 003 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 003.py   # -> output/003.pdf, output/003_segments.csv
"""
import core

# User fix (same policy as 001, sddb40 family): c0 normal + c1/c2 biphasic 2 types, K=3. MIN_PATTERN re-tuned due to a change in leading-partial merging, 2026-06-26.
PARAMS = {"SHARP_QRS_MIN": 2.0, "KMAX_S1": 3, "MIN_PATTERN": 3, "SIL_MIN": 0.10, "SINGLE_FLOOR": 0.15, "ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("003", PARAMS)
    if r:
        print(f"003: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
