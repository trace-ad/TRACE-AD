#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 019 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 019.py   # -> output/019.pdf, output/019_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("019", PARAMS)
    if r:
        print(f"019: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
