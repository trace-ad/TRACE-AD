#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 253 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 253.py   # -> output/253.pdf, output/253_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("253", PARAMS)
    if r:
        print(f"253: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
