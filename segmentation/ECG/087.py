#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 087 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 087.py   # -> output/087.pdf, output/087_segments.csv
"""
import core

# User note: segmentation is too fine, c0 should be larger. One beat = QRS + T-wave (period ~172, matching autocorrelation) was
# over-segmented by the RR detector mis-estimating period=113. With PERIOD=172, one beat = one segment. Since all beats are the same shape, false splits from phase jitter are
# merged and KMAX fixes K=1 (c0=98, all beats) (2026-06-26).
PARAMS = {"PERIOD": 172, "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("087", PARAMS)
    if r:
        print(f"087: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
