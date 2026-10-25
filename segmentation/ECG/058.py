#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 058 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 058.py   # -> output/058.pdf, output/058_segments.csv
"""
import core

PARAMS = {"ASSIGN_STRICT": True}   # suppress, via a radius guard, the misassignment of small-amplitude c0 beats to c1 (high-amplitude biphasic)

if __name__ == "__main__":
    r = core.run("058", PARAMS)
    if r:
        print(f"058: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
