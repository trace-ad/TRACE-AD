#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 093 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 093.py   # -> output/093.pdf, output/093_segments.csv
"""
import core

PARAMS = {"ASSIGN_STRICT": True, "QRS_SHARP_DETECT": True}   # resolve phase drift / peak-less units caused by T-wave mis-locking

if __name__ == "__main__":
    r = core.run("093", PARAMS)
    if r:
        print(f"093: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
