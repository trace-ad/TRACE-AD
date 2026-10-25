#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 153 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 153.py   # -> output/153.pdf, output/153_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("153", PARAMS)
    if r:
        print(f"153: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
