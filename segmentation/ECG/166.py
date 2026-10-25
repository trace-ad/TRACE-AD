#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 166 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 166.py   # -> output/166.pdf, output/166_segments.csv
"""
import core

# User note (apneaecg family): K=2 of normal beats + a few deep-negative PVC beats (large amplitude, SBD0.6+). Stage2 amplitude subdivision is suppressed with KMAX_S2=1,
# and the few PVCs are retained by lowering MIN_PATTERN (2026-06-26).
PARAMS = {"KMAX_S2": 1, "ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("166", PARAMS)
    if r:
        print(f"166: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
