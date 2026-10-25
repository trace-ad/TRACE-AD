#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 244 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 244.py   # -> output/244.pdf, output/244_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("244", PARAMS)
    if r:
        print(f"244: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
