#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 104 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 104.py   # -> output/104.pdf, output/104_segments.csv
"""
import core

# User note: large PVC/dome-shaped beats (n=6, deep-negative QRS -> large positive, amplitude ~950 vs normal ~350, SBD0.59 a distinct morphology,
# NOISEapneaecg4) are buried in c0. Lower MIN_PATTERN to separate the 2nd beat type (ordinary Stage2 subdivision is suppressed with KMAX_S2=1).
# c0:54 (normal) + c1:6 (PVC), K=2 (2026-06-26).
PARAMS = {"KMAX_S1": 2, "KMAX_S2": 1, "MIN_PATTERN": 6, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("104", PARAMS)
    if r:
        print(f"104: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
