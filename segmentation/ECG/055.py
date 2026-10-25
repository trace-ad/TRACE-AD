#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 055 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 055.py   # -> output/055.pdf, output/055_segments.csv
"""
import core

# User confirmation: c2 (n=8) has the same shape as c0 (n=82) (SBD0.19; in global reassignment beats move freely between c0/c2 = same beat type),
# but Stage1 complete-linkage mis-grouped it on the c1 side. With MERGE_TAU=0.20, same-shape merging of the final clusters
# consolidates c0+c2. K=2 of c0 (biphasic QRS) + c1 (deep negative -> large positive) (2026-06-26).
PARAMS = {"MERGE_TAU": 0.20}

if __name__ == "__main__":
    r = core.run("055", PARAMS)
    if r:
        print(f"055: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
