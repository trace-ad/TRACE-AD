#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 075 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 075.py   # -> output/075.pdf, output/075_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("075", PARAMS)
    if r:
        print(f"075: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
