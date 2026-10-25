#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 231 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 231.py   # -> output/231.pdf, output/231_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("231", PARAMS)
    if r:
        print(f"231: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
