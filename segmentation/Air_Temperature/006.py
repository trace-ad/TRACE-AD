#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 006 (Air Temperature): base-waveform segmentation + clustering.
A thin wrapper that only calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 006.py   # -> output/006.pdf, output/006_segments.csv
"""
import core

# series-specific parameters (override core defaults; an empty dict means all defaults):
#   PERIOD, RESAMPLE_LEN, PEAK_DIST_FRAC, PEAK_PROM, LOCALNORM_FRAC, SMOOTH_FRAC,
#   LEN_LO_FRAC, LEN_HI_FRAC, FLATTOP_MAX, NOISE_PEAKS_MAX, KMAX_S1, KMAX_S2,
#   MERGE_TAU, SIL_MIN, MIN_PATTERN, MAHAL_RIDGE, STAGE2_SBD, STAGE2_AMP_RANGE,
#   STAGE2_AMP_ACT, STAGE2_AMP_CURV, ARTIFACT_TAU
PARAMS = {}

if __name__ == "__main__":
    r = core.run("006", PARAMS)
    if r:
        print(f"006: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
