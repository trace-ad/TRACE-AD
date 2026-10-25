#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 217 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 217.py   # -> output/217.pdf, output/217_segments.csv
"""
import core

# Evaluation-AI inspection: over-segmentation (minor phase variants / Stage2 amplitude subdivision / noise clusters). Only ~1 real beat type, so suppress with KMAX (2026-06-26).
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("217", PARAMS)
    if r:
        print(f"217: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
