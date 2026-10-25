#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 020 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 020.py   # -> output/020.pdf, output/020_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("020", PARAMS)
    if r:
        print(f"020: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
