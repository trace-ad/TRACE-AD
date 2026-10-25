#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 170 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 170.py   # -> output/170.pdf, output/170_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("170", PARAMS)
    if r:
        print(f"170: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
