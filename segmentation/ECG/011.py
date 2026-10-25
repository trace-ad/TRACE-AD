#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 011 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 011.py   # -> output/011.pdf, output/011_segments.csv
"""
import core

# User note: inverted/biphasic QRS beats (n=2, negative dip -> positive recovery, SBD~0.56 from normal c0, a distinct morphology) are buried in c0.
# Lower MIN_PATTERN to separate the 2nd beat type as c1 (DISTORTEDECG family, 2026-06-26).
PARAMS = {"KMAX_S1": 2, "MIN_PATTERN": 2, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("011", PARAMS)
    if r:
        print(f"011: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
