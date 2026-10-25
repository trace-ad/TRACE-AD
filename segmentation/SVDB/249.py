#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 249 (TSB-AD-U / SVDB) ECG basic-waveform segmentation + clustering.
Thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 249.py   # -> output/249.pdf, output/249_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("249", PARAMS)
    if r:
        print(f"249: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
