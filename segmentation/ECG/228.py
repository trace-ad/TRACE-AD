#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 228 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 228.py   # -> output/228.pdf, output/228_segments.csv
"""
import core

# User note: revise segmentation. mit14134longtermecg. Sharp biphasic QRS (~134 interval, RR varies due to arrhythmia). The RR detector
# also false-detected small in-beat bumps as R-waves, giving period=64 (less than half) -> over-segmentation. QRS_SHARP_DETECT detects only real QRS, and PERIOD=134 (RR median)
# makes each segment = one beat (2QRS=0, 0QRS=0, n_seg=86). K=2 {c0:55 downward-QRS beats, c1:29 upward-QRS beats}. period 120 is Stage2 (D3) over-segmentation,
# so KMAX_S2=1 (2026-06-27).
PARAMS = {"PERIOD": 134, "QRS_SHARP_DETECT": True, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("228", PARAMS)
    if r:
        print(f"228: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
