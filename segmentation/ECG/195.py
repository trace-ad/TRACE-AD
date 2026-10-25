#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 195 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 195.py   # -> output/195.pdf, output/195_segments.csv
"""
import core

# User note (same sel840mECG1 family as 087): one beat = QRS + T-wave (period ~172) was over-segmented by the RR detector mis-estimating period=114 ->
# fragments became a false K=2. Autocorrelation 172 is the true period. With PERIOD=172, one beat = one segment; all beats same shape, KMAX fixes K=1 (2026-06-26).
PARAMS = {"PERIOD": 172, "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("195", PARAMS)
    if r:
        print(f"195: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
