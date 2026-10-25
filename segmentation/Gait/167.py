#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 167 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 167.py   # -> output/167.pdf, output/167_segments.csv
"""
import core

# With sub-period fragment merging (2026-06-25), the small phase-truncated cluster grew to n=20, so the MIN_PATTERN=15 absorption
# fell short and produced a false K=2 outlier (medoid SBD 0.151; the gait family is all K=1). MIN_PATTERN=20 absorbs the small skewed group into the main pulse.
PARAMS = {"MIN_PATTERN": 20}

if __name__ == "__main__":
    r = core.run("167", PARAMS)
    if r:
        print(f"167: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
