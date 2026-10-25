#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 214 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 214.py   # -> output/214.pdf, output/214_segments.csv
"""
import core

# Evaluation-AI inspection: over-segmentation (same-shape / noise / Stage2 subdivision). Only ~1 real beat type, so suppress with KMAX (2026-06-26, 2nd pass).
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("214", PARAMS)
    if r:
        print(f"214: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
