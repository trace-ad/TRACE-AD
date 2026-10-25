#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 164 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 164.py   # -> output/164.pdf, output/164_segments.csv
"""
import core

# User note (apneaecg family): K=2 of normal beats + a few deep-negative PVC beats (large amplitude, SBD0.6+). Stage2 amplitude subdivision is suppressed with KMAX_S2=1,
# and the few PVCs are retained by lowering MIN_PATTERN (2026-06-26).
PARAMS = {"KMAX_S1": 2, "MIN_PATTERN": 4, "SIL_MIN": 0.10, "KMAX_S2": 1, "ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("164", PARAMS)
    if r:
        print(f"164: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
