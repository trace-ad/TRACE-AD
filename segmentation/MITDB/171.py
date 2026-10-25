#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 171 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 171.py   # -> output/171.pdf, output/171_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("171", PARAMS)
    if r:
        print(f"171: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
