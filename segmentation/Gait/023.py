#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 023 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 023.py   # -> output/023.pdf, output/023_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("023", PARAMS)
    if r:
        print(f"023: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
