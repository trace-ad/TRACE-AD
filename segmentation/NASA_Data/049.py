#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 049 (NASA Data): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 049.py   # -> output/049.pdf, output/049_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("049", PARAMS)
    if r:
        print(f"049: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
