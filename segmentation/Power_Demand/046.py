#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 046 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 046.py   # -> output/046.pdf, output/046_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("046", PARAMS)
    if r:
        print(f"046: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
