#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 157 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 157.py   # -> output/157.pdf, output/157_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("157", PARAMS)
    if r:
        print(f"157: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
