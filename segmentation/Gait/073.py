#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 073 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 073.py   # -> output/073.pdf, output/073_segments.csv
"""
import core

# The clean sibling from the same recording (181=park3m) has K_pulse=1. Fix the false splits caused by distortion/noise to a single one (re-verified with upstroke splitting 2026-06-25).
PARAMS = {"KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("073", PARAMS)
    if r:
        print(f"073: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
