#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 103 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 103.py   # -> output/103.pdf, output/103_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("103", PARAMS)
    if r:
        print(f"103: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
