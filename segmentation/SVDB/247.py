#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 247 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 247.py   # -> output/247.pdf, output/247_segments.csv
"""
import core
PARAMS = {"PERIOD": 100, "QRS_SHARP_DETECT": True, "SINGLE_FLOOR": 0.5, "SINGLE_FACTOR": 4.0}
# Period 26 misdetected -> forced 100. SHARP = derivative-energy R detection stabilizes phase (fixes alternating short/long, fusion, spurious Kp=2).
# Relaxed SINGLE: absorb the few artifacts (c1/c2) nearly identical to normal beats into c0 (keep only c3 = leading partial).
if __name__ == "__main__":
    r = core.run("247", PARAMS)
    if r:
        print(f"247: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
