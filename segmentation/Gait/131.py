#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 131 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 131.py   # -> output/131.pdf, output/131_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("131", PARAMS)
    if r:
        print(f"131: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
