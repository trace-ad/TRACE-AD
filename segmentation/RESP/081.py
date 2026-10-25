#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 081 (RESP).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 081.py   # -> output/081.pdf, output/081_segments.csv
"""
import core

# Series-specific parameters (override core defaults; an empty dict = all defaults):
#   PERIOD, RESAMPLE_LEN, KMAX_S1, KMAX_S2, MAXLAG_FRAC, MERGE_TAU, SIL_MIN,
#   MIN_PATTERN, MAHAL_RIDGE, STAGE2_SBD, STAGE2_AMP_RANGE, STAGE2_AMP_ACT,
#   STAGE2_AMP_CURV, ARTIFACT_TAU, FLATTOP_MAX, PEAKPOS_MAX, NOISE_PEAKS_MAX
PARAMS = {}

if __name__ == "__main__":
    r = core.run("081", PARAMS)
    if r:
        print(f"081: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
