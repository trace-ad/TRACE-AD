#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 047 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 047.py   # -> output/047.pdf, output/047_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("047", PARAMS)
    if r:
        print(f"047: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
