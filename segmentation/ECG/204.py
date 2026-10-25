#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 204 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 204.py   # -> output/204.pdf, output/204_segments.csv
"""
import core

# User note: CHARIS is a blood-pressure/physiological-signal pulse (period ~34). It has no sharp QRS but smooth pulses, and the RR detector picked up
# a small notch within each pulse (period 17) and split each pulse into 2. The strongest autocorrelation peak = 34 is the true pulse period. With PERIOD=34,
# one pulse = one segment (cut at the trough); all pulses same shape, K=1 (2026-06-26).
PARAMS = {"PERIOD": 34, "SHARP_QRS_MIN": 1.2, "KMAX_S1": 1, "KMAX_S2": 1, "RPEAK_LOCALNORM": True, "TEST_SPLIT_MULTIBEAT": True}  # Same type as 202: round sinusoidal oscillation. clean recovery + K=1 + local-normalized R detection. TEST_SPLIT_MULTIBEAT=True re-splits the 2-hump-in-1-beat fusions in test (fusions -> 0; existing boundaries kept)

if __name__ == "__main__":
    r = core.run("204", PARAMS)
    if r:
        print(f"204: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
