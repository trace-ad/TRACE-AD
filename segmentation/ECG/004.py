#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 004 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 004.py   # -> output/004.pdf, output/004_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("004", PARAMS)
    if r:
        print(f"004: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
