#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 250 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 250.py   # -> output/250.pdf, output/250_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("250", PARAMS)
    if r:
        print(f"250: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
