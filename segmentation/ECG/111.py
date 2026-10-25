#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 111 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 111.py   # -> output/111.pdf, output/111_segments.csv
"""
import core

PARAMS = {"ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("111", PARAMS)
    if r:
        print(f"111: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
