#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 051 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 051.py   # -> output/051.pdf, output/051_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("051", PARAMS)
    if r:
        print(f"051: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
