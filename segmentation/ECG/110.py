#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 110 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 110.py   # -> output/110.pdf, output/110_segments.csv
"""
import core

PARAMS = {"ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("110", PARAMS)
    if r:
        print(f"110: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
