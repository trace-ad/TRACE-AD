#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 215 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 215.py   # -> output/215.pdf, output/215_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("215", PARAMS)
    if r:
        print(f"215: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
