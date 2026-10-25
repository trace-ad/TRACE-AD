#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 109 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 109.py   # -> output/109.pdf, output/109_segments.csv
"""
import core

PARAMS = {"ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("109", PARAMS)
    if r:
        print(f"109: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
