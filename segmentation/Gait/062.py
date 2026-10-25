#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 062 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 062.py   # -> output/062.pdf, output/062_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("062", PARAMS)
    if r:
        print(f"062: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
