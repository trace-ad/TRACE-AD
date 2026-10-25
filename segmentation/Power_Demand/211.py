#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 211 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 211.py   # -> output/211.pdf, output/211_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("211", PARAMS)
    if r:
        print(f"211: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
