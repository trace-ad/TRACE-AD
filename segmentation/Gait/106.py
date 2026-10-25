#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 106 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 106.py   # -> output/106.pdf, output/106_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("106", PARAMS)
    if r:
        print(f"106: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
