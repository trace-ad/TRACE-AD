#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 054 (Acceleration Sensor Data). Thin wrapper that calls core.
  python 054.py   # -> output/054.pdf, output/054_segments.csv
"""
import core
# Split the over-long (merged) segments at the valleys in regions where the cadence breaks down.
PARAMS = {"SPLIT_LONG_FRAC": 1.5}
if __name__ == "__main__":
    r = core.run("054", PARAMS)
    if r:
        print(f"054: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
