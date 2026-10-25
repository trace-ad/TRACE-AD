#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 235 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 235.py   # -> output/235.pdf, output/235_segments.csv
"""
import core

# User note: proper segmentation makes c1 disappear + revise clustering. mit14157longtermecg (same DB as 233/234).
# The RR detector's over-detection gave period=76 -> short flat fragments (42, 27 samples) = c1 appeared. QRS_SHARP_DETECT + PERIOD=114 (RR median)
# detects only real QRS -> fragments disappear.
# User note: revise clusters. The deep-dip beats (4604/18732, dip ~2.0) are a distinct type from the main one (same as 233) but n=2<MIN_PATTERN=8 absorbs them.
# The 13/151 split of the main type is SBD 0.06 = a same-shape false split, so ignore it. MIN_PATTERN=2 retains deep-dip.
# Additional user note: under znorm shape, 4604 (flat + sharp spike, small MAD) becomes a deep-dip and mixes into c1, while the raw deep-dips 4411/4490 scatter.
# -> base it on raw dip (A). DEEP_DIP_FRAC=1.5 gathers beats with raw dip > median*1.5 into c1. K=2 {c0:163, c1:3 deep dip (4411/4490/18732)} (2026-06-27).
PARAMS = {"PERIOD": 114, "QRS_SHARP_DETECT": True, "DEEP_DIP_FRAC": 1.5}

if __name__ == "__main__":
    r = core.run("235", PARAMS)
    if r:
        print(f"235: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
