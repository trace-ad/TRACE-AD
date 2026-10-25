#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 053 (Acceleration Sensor Data). Thin wrapper that calls core.
  python 053.py   # -> output/053.pdf, output/053_segments.csv
"""
import core
# Split the over-long (merged) segments at the valleys in regions where the cadence breaks down.
PARAMS = {"SPLIT_LONG_FRAC": 1.5}
if __name__ == "__main__":
    r = core.run("053", PARAMS)
    if r:
        print(f"053: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
