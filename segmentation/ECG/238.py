#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 238 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 238.py   # -> output/238.pdf, output/238_segments.csv
"""
import core

# User note: revise clusters. mit14157longtermecg. QRS_SHARP_DETECT + PERIOD=114 (RR median / autocorrelation).
# Same family as 236/237 but only 2 types: c0 = normal beat (upward QRS), c1 = downward-spike type (-28, polarity-reversed ectopic beat,
# @4873/11990/14486/17913/19028/20369). No deep-dip type. FORCE_SHAPE_K=2 splits by shape into 2. K=2 {c0:179, c1:6} (2026-06-27).
# * 2026-08-06: the PVC (c1) is premature, and in the long RR (compensatory pause / long interval) before it, the rr-anchor cut window
#   (0.78*rr) is placed toward the next beat, so the flat baseline in between is taken by the previous beat, making the previous beat over-long (~155) and the next over-short (~73)
#   in many places. RR_CAP_FRAC=1.05 caps the long RR to the period and splits each beat evenly over ~1 period
#   (e.g. 155/73/155/72 -> 103/125/123/104). Short beats <0.7P train8/test31 -> train5/test22. K_pulse=2 kept.
PARAMS = {"PERIOD": 114, "QRS_SHARP_DETECT": True, "FORCE_SHAPE_K": 2, "SINGLE_FLOOR": 0.16,
          "RR_CAP_FRAC": 1.05, "PEAK_DIST_FRAC": 0.50}

if __name__ == "__main__":
    r = core.run("238", PARAMS)
    if r:
        print(f"238: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
