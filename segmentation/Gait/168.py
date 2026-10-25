#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 168 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 168.py   # -> output/168.pdf, output/168_segments.csv
"""
import core

# Fix the wide peak at the right edge (constant as [flat+peak]). KMAX fixes Kp=1.
PARAMS = {"ANCHOR_MODE": "falledge", "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("168", PARAMS)
    if r:
        print(f"168: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
