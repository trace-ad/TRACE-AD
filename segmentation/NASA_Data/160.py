#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 160 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 160.py   # -> output/160.pdf, output/160_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("160", PARAMS)
    if r:
        print(f"160: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
