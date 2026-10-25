#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 229 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 229.py   # -> output/229.pdf, output/229_segments.csv
"""
import core

# User note: 2750 is c1. Revise clusters. mit14134longtermecg (same DB as 228). Sharp biphasic QRS (~134 interval). The RR detector
# false-detected in-beat bumps and under-estimated period=87 -> over-segmentation. QRS_SHARP_DETECT + PERIOD=134 (RR median) gives one segment per beat.
# The clusters are "downward-QRS beats vs upward-QRS beats". KMAX_S1=2 gives 2 types (merging the same-shape downward c0/c2 at SBD0.233).
# K=2 {c0:83 downward QRS, c1:43 upward QRS}. 2750 (upward, low amplitude) is correctly classified into c1 (2026-06-27).
PARAMS = {"PERIOD": 134, "QRS_SHARP_DETECT": True, "KMAX_S1": 2, "KMAX_S2": 1, "RR_CAP_FRAC": 1.1, "QRS_HEIGHT": 0.08}

if __name__ == "__main__":
    r = core.run("229", PARAMS)
    if r:
        print(f"229: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
