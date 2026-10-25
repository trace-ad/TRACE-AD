#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 239 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 239.py   # -> output/239.pdf, output/239_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("239", PARAMS)
    if r:
        print(f"239: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
