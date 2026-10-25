#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 052 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 052.py   # -> output/052.pdf, output/052_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("052", PARAMS)
    if r:
        print(f"052: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
