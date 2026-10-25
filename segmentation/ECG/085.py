#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 085 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 085.py   # -> output/085.pdf, output/085_segments.csv
"""
import core

# User note: one beat = sharp QRS + large T-wave (period ~108). The RR detector picked up both the QRS and the T-wave and mis-estimated period=35,
# over-segmenting one beat into c0 (T part) + c1 (QRS part) etc. Autocorrelation 108 is the true period. With PERIOD=108, one beat = one segment.
# Beats are a single type (false splits from phase/segmentation jitter are merged), fixed to K=1 via KMAX (2026-06-26).
PARAMS = {"PERIOD": 108, "QRS_SHARP_DETECT": True, "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("085", PARAMS)
    if r:
        print(f"085: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
