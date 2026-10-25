#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 236 (TSB-AD-U / SED) square-pulse-train basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 236.py   # -> output/236.pdf, output/236_segments.csv
"""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("236", PARAMS)
    if r:
        print(f"236: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
