#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 064 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 064.py   # -> output/064.pdf, output/064_segments.csv
"""
import core

# Sharp peak at the left edge. KMAX fixes Kp=1.
# Peak at the left edge (upstroke). The low-amplitude region at the start of walking is unsuitable as a reference, so leave it as an artifact (AMP_FLOOR+AMP_RECOVER).
PARAMS = {"ANCHOR_MODE": "upstroke", "KMAX_S1": 1, "KMAX_S2": 1, "AMP_FLOOR_FRAC": 0.6, "AMP_RECOVER_MIN": 0.5}

if __name__ == "__main__":
    r = core.run("064", PARAMS)
    if r:
        print(f"064: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
