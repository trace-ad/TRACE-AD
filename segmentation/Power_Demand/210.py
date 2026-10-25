#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 210 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 210.py   # -> output/210.pdf, output/210_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("210", PARAMS)
    if r:
        print(f"210: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
