#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 156 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 156.py   # -> output/156.pdf, output/156_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("156", PARAMS)
    if r:
        print(f"156: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
