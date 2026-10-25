#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 202 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 202.py   # -> output/202.pdf, output/202_segments.csv
"""
import core

# User note: CHARIS is a blood-pressure/physiological-signal pulse (period ~33). It has no sharp QRS but smooth pulses, and the RR detector picked up
# a small notch within each pulse (period 17) and split each pulse into 2. The strongest autocorrelation peak = 33 is the true pulse period. With PERIOD=33,
# one pulse = one segment (cut at the trough); all pulses same shape, K=1 (2026-06-26).
PARAMS = {"PERIOD": 33, "SHARP_QRS_MIN": 1.2, "KMAX_S1": 1, "KMAX_S2": 1, "RPEAK_LOCALNORM": True,
          "TEST_SPLIT_MULTIBEAT": True}  # 202 = round sinusoidal oscillation. Relaxing SHARP_QRS_MIN recovers clean, fixes K=1, and local-normalized R detection tracks amplitude variation (resolving fragment / two-hump merging).
                                         # TEST_SPLIT_MULTIBEAT=True re-splits the 2-hump-in-1-beat fusions in test (which remain under the default split_multibeat=False) (fusions 10->0; existing boundaries kept, 20 boundaries added)

if __name__ == "__main__":
    r = core.run("202", PARAMS)
    if r:
        print(f"202: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
