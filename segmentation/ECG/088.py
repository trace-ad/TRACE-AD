#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 088 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 088.py   # -> output/088.pdf, output/088_segments.csv
"""
import core

# User: K=1
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1, "QRS_SHARP_DETECT": True}   # resolve phase drift / peak-less units caused by T-wave mis-locking

if __name__ == "__main__":
    r = core.run("088", PARAMS)
    if r:
        print(f"088: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
