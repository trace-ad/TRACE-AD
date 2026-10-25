#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 240 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 240.py   # -> output/240.pdf, output/240_segments.csv
"""
import core

# Square wave: cut at the rising-edge crossing to strictly align phase (spike at the left edge). KMAX fixes Kp=1.
PARAMS = {"ANCHOR_MODE": "riseedge", "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("240", PARAMS)
    if r:
        print(f"240: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
