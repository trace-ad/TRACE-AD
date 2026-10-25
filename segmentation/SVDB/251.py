#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 251 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 251.py   # -> output/251.pdf, output/251_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("251", PARAMS)
    if r:
        print(f"251: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
