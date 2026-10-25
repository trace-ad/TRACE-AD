#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 173 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 173.py   # -> output/173.pdf, output/173_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("173", PARAMS)
    if r:
        print(f"173: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
