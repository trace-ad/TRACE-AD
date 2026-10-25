#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 130 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 130.py   # -> output/130.pdf, output/130_segments.csv
"""
import core

# Sharp peak at the left edge. KMAX fixes Kp=1.
PARAMS = {"ANCHOR_MODE": "upstroke", "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("130", PARAMS)
    if r:
        print(f"130: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
