#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 097 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 097.py   # -> output/097.pdf, output/097_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("097", PARAMS)
    if r:
        print(f"097: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
