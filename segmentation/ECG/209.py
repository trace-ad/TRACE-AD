#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 209 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 209.py   # -> output/209.pdf, output/209_segments.csv
"""
import core

# User note: split 12750 + 13500/13750 are also c1. Fantasia = sharp QRS + T-wave with high-frequency ripple superimposed.
# PRE_SMOOTH_WN=0.067 (low-pass ripple removal, for display/clustering). R-wave detection uses the raw signal (does not crush weak/abnormal-beat QRS,
# two-signal scheme) + QRS_SHARP_DETECT. The disturbed episode [12579:13816] (a continuous span of amplitude drop/inversion that SBD alone cannot capture)
# is force-separated into c1 via ABNORMAL_SPANS (by expert judgment the start is set to 12076, including 12250/12500).
# Normal beats are a single P-QRS-T morphology (2026-06-26).
PARAMS = {"PERIOD": 253, "QRS_SHARP_DETECT": True, "PRE_SMOOTH_WN": 0.067, "ABNORMAL_SPANS": [(12076, 13800)]}

if __name__ == "__main__":
    r = core.run("209", PARAMS)
    if r:
        print(f"209: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
