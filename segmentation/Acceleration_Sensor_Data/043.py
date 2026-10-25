#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 043 (Acceleration Sensor Data). Thin wrapper that calls core.
  python 043.py   # -> output/043.pdf, output/043_segments.csv
"""
import core
# Whale: two-stroke type + low-amplitude glide separated as an artifact (unsuitable as a reference template).
PARAMS = {"PERIOD": 205, "AMP_FLOOR_FRAC": 0.35, "AMP_RECOVER_MIN": 0.4, "FLAT_ARTIFACT_FRAC": 0.5, "AMP_ASSIGN_W": 0.4}
if __name__ == "__main__":
    r = core.run("043", PARAMS)
    if r:
        print(f"043: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
