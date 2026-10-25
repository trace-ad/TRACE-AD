#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 181 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 181.py   # -> output/181.pdf, output/181_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("181", PARAMS)
    if r:
        print(f"181: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
