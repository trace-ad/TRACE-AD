#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 070 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 070.py   # -> output/070.pdf, output/070_segments.csv
"""
import core

# Evaluation-AI inspection: over-segmentation (same-shape / noise / Stage2 subdivision). Only ~1 real beat type, so suppress with KMAX (2026-06-26, 2nd pass).
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("070", PARAMS)
    if r:
        print(f"070: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
