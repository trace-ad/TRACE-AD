#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 094 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 094.py   # -> output/094.pdf, output/094_segments.csv
"""
import core

# User note: 094 has no sharp QRS but repeats a large hump (gentle rise -> peak -> trough) of period ~80. The RR detector
# picked up noise/sub-peaks and mis-estimated period=31, splitting each hump into 3. Autocorrelation 80 is the true period. With PERIOD=80,
# one hump = one segment (cut at the trough) (2026-06-26).
PARAMS = {"PERIOD": 80}

if __name__ == "__main__":
    r = core.run("094", PARAMS)
    if r:
        print(f"094: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
