#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 022 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 022.py   # -> output/022.pdf, output/022_segments.csv
"""
import core

# Cut the sharp peak at the valley before its upstroke, placing it at the left edge. KMAX fixes Kp=1.
PARAMS = {"ANCHOR_MODE": "upstroke", "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("022", PARAMS)
    if r:
        print(f"022: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
