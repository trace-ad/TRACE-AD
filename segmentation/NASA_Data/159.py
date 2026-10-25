#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 159 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 159.py   # -> output/159.pdf, output/159_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("159", PARAMS)
    if r:
        print(f"159: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
