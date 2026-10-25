#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 101 (EPG).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 101.py   # -> output/101.pdf, output/101_segments.csv
"""
import core

# Series-specific parameters (override core defaults; an empty dict = all defaults):
#   PERIOD, RESAMPLE_LEN, PEAK_DIST_FRAC, PEAK_PROM, LOCALNORM_FRAC, SMOOTH_FRAC,
#   LEN_LO_FRAC, LEN_HI_FRAC, FLATTOP_MAX, NOISE_PEAKS_MAX, KMAX_S1, KMAX_S2,
#   MERGE_TAU, SIL_MIN, MIN_PATTERN, MAHAL_RIDGE, STAGE2_SBD, STAGE2_AMP_RANGE,
#   STAGE2_AMP_ACT, STAGE2_AMP_CURV, ARTIFACT_TAU
# c1 (artifact) is mostly noisy normal beats with the same shape as c0 (SBD~0.18). Merge all non-clean into c0 to form a single normal cluster.
PARAMS = {'NO_ARTIFACT': True, 'MERGE_SHORT_FRAC': 0.6}
if __name__ == "__main__":
    r = core.run("101", PARAMS)
    if r:
        print(f"101: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
