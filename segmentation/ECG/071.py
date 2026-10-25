#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 071 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 071.py   # -> output/071.pdf, output/071_segments.csv
"""
import core

# Evaluation-AI inspection: over-segmentation (same-shape / noise / Stage2 subdivision). Only ~2 real beat types, so suppress with KMAX (2026-06-26, 2nd pass).
PARAMS = {"KMAX_S1": 2, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("071", PARAMS)
    if r:
        print(f"071: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
