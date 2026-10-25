#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 193 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 193.py   # -> output/193.pdf, output/193_segments.csv
"""
import core

# User note (same s20101m family as 085): one beat = QRS + large T-wave (period ~108). The RR detector mis-estimated period=33 -> over-segmentation, so
# PERIOD=108. The large T-wave was also detected as a spurious R-wave and produced short segments, so PEAK_DIST_FRAC=0.70
# suppresses the T-wave (one beat = one segment). All beats same shape, KMAX fixes K=1 (2026-06-26).
PARAMS = {"PERIOD": 108, "PEAK_DIST_FRAC": 0.70, "KMAX_S1": 1, "KMAX_S2": 1, "RR_CUT_BEFORE_QRS": True, "QRS_SHARP_DETECT": True}

if __name__ == "__main__":
    r = core.run("193", PARAMS)
    if r:
        print(f"193: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
