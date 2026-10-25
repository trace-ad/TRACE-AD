#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 172 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 172.py   # -> output/172.pdf, output/172_segments.csv
"""
import core

PARAMS = {}

if __name__ == "__main__":
    r = core.run("172", PARAMS)
    if r:
        print(f"172: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
