#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 050 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 050.py   # -> output/050.pdf, output/050_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("050", PARAMS)
    if r:
        print(f"050: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
