#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 059 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 059.py   # -> output/059.pdf, output/059_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("059", PARAMS)
    if r:
        print(f"059: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
