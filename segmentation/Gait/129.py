#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 129 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 129.py   # -> output/129.pdf, output/129_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("129", PARAMS)
    if r:
        print(f"129: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
