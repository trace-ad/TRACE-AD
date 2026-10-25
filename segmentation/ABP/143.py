#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic-waveform segmentation + clustering for series 143.
A thin wrapper that just calls the shared engine `core`. Series-specific tuning is done via PARAMS
(overrides core's default values; an empty dict means all defaults).

  python 143.py        # process series 143 -> output/143.pdf, output/143_segments.csv
"""
import core

# Series-specific parameters (keys that can override core's defaults):
#   PERIOD (manual period / None = auto), MIN_PATTERN, MERGE_TAU, MAXLAG_FRAC, SIL_MIN,
#   FLATTOP_MAX, PEAKPOS_MAX, NOISE_PEAKS_MAX, RESAMPLE_LEN, KMAX
PARAMS = {}

if __name__ == "__main__":
    r = core.run("143", PARAMS)
    if r:
        print(f"143: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  "
              f"period={r['period']}")
