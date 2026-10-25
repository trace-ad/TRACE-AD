#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 044 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 044.py   # -> output/044.pdf, output/044_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("044", PARAMS)
    if r:
        print(f"044: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
