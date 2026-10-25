#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 084 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 084.py   # -> output/084.pdf, output/084_segments.csv
"""
import core

# User: K=1 is fine
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("084", PARAMS)
    if r:
        print(f"084: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
