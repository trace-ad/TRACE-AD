#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 018 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 018.py   # -> output/018.pdf, output/018_segments.csv
"""
import core

# Evaluation-AI inspection: over-segmentation (same-shape / noise / Stage2 subdivision). Only ~1 real beat type, so suppress with KMAX (2026-06-26, 2nd pass).
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("018", PARAMS)
    if r:
        print(f"018: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
