#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 154 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 154.py   # -> output/154.pdf, output/154_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("154", PARAMS)
    if r:
        print(f"154: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
