#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 238 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 238.py   # -> output/238.pdf, output/238_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("238", PARAMS)
    if r:
        print(f"238: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
