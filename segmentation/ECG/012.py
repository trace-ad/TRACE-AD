#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 012 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 012.py   # -> output/012.pdf, output/012_segments.csv
"""
import core

# User note: inverted/biphasic QRS beats (n=5, negative dip -> positive recovery, SBD~0.56 from normal c0, a distinct morphology) are buried in c0.
# Lower MIN_PATTERN to separate the 2nd beat type as c1 (DISTORTEDECG family, 2026-06-26).
PARAMS = {"KMAX_S1": 2, "MIN_PATTERN": 3, "SIL_MIN": 0.10}

if __name__ == "__main__":
    r = core.run("012", PARAMS)
    if r:
        print(f"012: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
