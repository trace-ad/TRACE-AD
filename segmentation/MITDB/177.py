#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 177 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 177.py   # -> output/177.pdf, output/177_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("177", PARAMS)
    if r:
        print(f"177: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
