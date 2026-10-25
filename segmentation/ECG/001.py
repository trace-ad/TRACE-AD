#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 001 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 001.py   # -> output/001.pdf, output/001_segments.csv
"""
import core

# User note: the biphasic QRS (2nd beat type) splits into c1 (clean) / c2 / c3 (non-clean) + further into 2 morphologies (tall positive spike -> negative -> T vs
# small positive spike -> sustained plateau, medoid SBD0.31). With SHARP_QRS_MIN=2.0, merge c2/c3 into c1, then subdivide c1 into 2 morphologies
# (KMAX_S1=3/MIN_PATTERN=3/SIL_MIN=0.10). c0 normal + c1/c2 biphasic 2 types (13/5) (2026-06-26).
PARAMS = {"SHARP_QRS_MIN": 2.0, "KMAX_S1": 3, "MIN_PATTERN": 3, "SIL_MIN": 0.10, "ASSIGN_STRICT": True}

if __name__ == "__main__":
    r = core.run("001", PARAMS)
    if r:
        print(f"001: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
