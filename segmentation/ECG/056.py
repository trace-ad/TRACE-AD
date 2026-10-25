#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 056 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 056.py   # -> output/056.pdf, output/056_segments.csv
"""
import core

# User note: a deep-negative QRS type (n=4, -30 to -47 -> large positive, SBD0.58 from normal beats, a distinct morphology, DISTORTEDapneaecg) is
# buried in c0. Lower MIN_PATTERN to separate the 2nd beat type as c1. c0:46 (normal) + c1:4 (deep negative), K=2 (2026-06-26).
PARAMS = {"KMAX_S1": 2, "MIN_PATTERN": 4, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("056", PARAMS)
    if r:
        print(f"056: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
