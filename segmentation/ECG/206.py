#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 206 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 206.py   # -> output/206.pdf, output/206_segments.csv
"""
import core

# User note: the tail is a high-frequency noise region from signal degradation and became many singleton artifacts. With NOISE_ROUGH_TAU=0.20,
# the rough-noise artifacts are consolidated into a single noise cluster (c0 = normal pulses unchanged) (2026-06-26).
PARAMS = {"PERIOD": 37, "SHARP_QRS_MIN": 1.2, "KMAX_S1": 1, "KMAX_S2": 1, "RPEAK_LOCALNORM": True, "NOISE_ROUGH_TAU": 0.20}  # Same type as 202 (clean recovery + K=1 + local-normalized R detection) + consolidate the rough-noise region into one cluster

if __name__ == "__main__":
    r = core.run("206", PARAMS)
    if r:
        print(f"206: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
