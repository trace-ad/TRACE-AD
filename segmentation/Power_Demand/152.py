#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 152 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 152.py   # -> output/152.pdf, output/152_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("152", PARAMS)
    if r:
        print(f"152: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
