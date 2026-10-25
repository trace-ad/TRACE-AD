#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 063 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 063.py   # -> output/063.pdf, output/063_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("063", PARAMS)
    if r:
        print(f"063: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
