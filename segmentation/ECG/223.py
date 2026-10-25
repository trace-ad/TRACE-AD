#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 223 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 223.py   # -> output/223.pdf, output/223_segments.csv
"""
import core

# User note (multiple rounds): c0 contamination in c2/c3/c4 (mis-segmented) + split 51750 into 2 + reconsider c1/c2. mit14046longtermecg (period 98).
# Normal beats with sharp downward QRS (c0) + upward (reverse-polarity) PVC (c1). QRS_SHARP_DETECT detects the PVC's sharp deflection as an R-wave and separates the merge.
# In addition, the normal beat right after a PVC is close (47<dist) and weak in energy and went undetected -> the merge remained (e.g. 51750). QRS_HEIGHT=0.05 + PEAK_DIST_FRAC
# =0.45 also detects the close weak QRS (detects 51768 and splits 51750 into PVC/normal). c1 is a single PVC type (large amplitude is same-shape under znorm,
# no Stage2 split). Final K=2 {c0:727 normal, c1:40 PVC}, all mis-segmented singletons gone (2026-06-27).
# * core fix: changed detect_qrs_sharp's distance from a fixed 0.55 to being tied to PEAK_DIST_FRAC / R-wave detection always uses the raw signal (avoiding double smoothing).
PARAMS = {"QRS_SHARP_DETECT": True, "QRS_HEIGHT": 0.05, "PEAK_DIST_FRAC": 0.45, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("223", PARAMS)
    if r:
        print(f"223: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
