#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Basic-waveform segmentation + clustering for series 199.
A thin wrapper that just calls the shared engine `core`. Series-specific tuning is done via PARAMS
(overrides core's default values; an empty dict means all defaults).

  python 199.py        # process series 199 -> output/199.pdf, output/199_segments.csv
"""
import core

# Series-specific parameters (keys that can override core's defaults):
#   PERIOD (manual period / None = auto), MIN_PATTERN, MERGE_TAU, MAXLAG_FRAC, SIL_MIN,
#   FLATTOP_MAX, PEAKPOS_MAX, NOISE_PEAKS_MAX, RESAMPLE_LEN, KMAX
PARAMS = {}

if __name__ == "__main__":
    r = core.run("199", PARAMS)
    if r:
        print(f"199: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  "
              f"period={r['period']}")
