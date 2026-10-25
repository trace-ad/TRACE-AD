#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 237 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 237.py   # -> output/237.pdf, output/237_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("237", PARAMS)
    if r:
        print(f"237: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
