#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 241 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 241.py   # -> output/241.pdf, output/241_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("241", PARAMS)
    if r:
        print(f"241: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
