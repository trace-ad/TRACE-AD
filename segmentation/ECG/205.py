#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 205 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 205.py   # -> output/205.pdf, output/205_segments.csv
"""
import core

# User note: CHARIS is a blood-pressure/physiological-signal pulse (period ~33). The RR detector picked up a small notch within each pulse (period 17) and split each pulse
# into 2. The strongest autocorrelation peak = 33 is the true pulse period. With PERIOD=33, one pulse = one segment; all pulses same shape, K=1 (2026-06-26).
PARAMS = {"PERIOD": 33, "SHARP_QRS_MIN": 1.2, "KMAX_S1": 1, "KMAX_S2": 1, "RPEAK_LOCALNORM": True, "TEST_SPLIT_MULTIBEAT": True}  # Same type as 202: round sinusoidal oscillation. clean recovery + K=1 + local-normalized R detection. TEST_SPLIT_MULTIBEAT=True re-splits the 2-hump-in-1-beat fusions in test (fusions -> 0; existing boundaries kept)

if __name__ == "__main__":
    r = core.run("205", PARAMS)
    if r:
        print(f"205: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
