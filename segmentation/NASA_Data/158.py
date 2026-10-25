#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 158 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 158.py   # -> output/158.pdf, output/158_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("158", PARAMS)
    if r:
        print(f"158: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
