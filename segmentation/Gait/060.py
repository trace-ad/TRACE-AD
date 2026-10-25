#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 060 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 060.py   # -> output/060.pdf, output/060_segments.csv
"""
import core

# Wide peak: cut at the falling-edge crossing so [flat+peak]=peak is fixed at the right edge (constant phase). KMAX fixes Kp=1.
PARAMS = {"ANCHOR_MODE": "falledge", "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("060", PARAMS)
    if r:
        print(f"060: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
