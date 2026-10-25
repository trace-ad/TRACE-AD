#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 127 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 127.py   # -> output/127.pdf, output/127_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("127", PARAMS)
    if r:
        print(f"127: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
