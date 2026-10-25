#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 038 (EPG).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 038.py   # -> output/038.pdf, output/038_segments.csv
"""
import core

# Series-specific parameters (override core defaults; an empty dict = all defaults):
#   PERIOD, RESAMPLE_LEN, PEAK_DIST_FRAC, PEAK_PROM, LOCALNORM_FRAC, SMOOTH_FRAC,
#   LEN_LO_FRAC, LEN_HI_FRAC, FLATTOP_MAX, NOISE_PEAKS_MAX, KMAX_S1, KMAX_S2,
#   MERGE_TAU, SIL_MIN, MIN_PATTERN, MAHAL_RIDGE, STAGE2_SBD, STAGE2_AMP_RANGE,
#   STAGE2_AMP_ACT, STAGE2_AMP_CURV, ARTIFACT_TAU
# Same recipe as 039-042: merge non-clean into c0 (K=1) + merge short fragments from shallow-notch over-segmentation. cut is a deep valley (down-spike).
PARAMS = {"NO_ARTIFACT": True, "MERGE_SHORT_FRAC": 0.6}

if __name__ == "__main__":
    r = core.run("038", PARAMS)
    if r:
        print(f"038: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
