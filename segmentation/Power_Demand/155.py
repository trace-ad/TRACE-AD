#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 155 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 155.py   # -> output/155.pdf, output/155_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("155", PARAMS)
    if r:
        print(f"155: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
