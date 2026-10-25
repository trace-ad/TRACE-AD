#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 227 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 227.py   # -> output/227.pdf, output/227_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("227", PARAMS)
    if r:
        print(f"227: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
