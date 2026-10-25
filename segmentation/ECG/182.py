#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 182 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 182.py   # -> output/182.pdf, output/182_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("182", PARAMS)
    if r:
        print(f"182: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
