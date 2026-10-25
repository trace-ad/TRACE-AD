#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 248 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 248.py   # -> output/248.pdf, output/248_segments.csv
"""
import core

# weallwalk: 1 step=24 (48 is two steps and looks like "spike+spike+half"). upstroke keeps the peak fixed at the left edge. Flat pauses are aggregated.
PARAMS = {"PERIOD": 24, "ANCHOR_MODE": "upstroke", "KMAX_S1": 1, "KMAX_S2": 1, "FLAT_ARTIFACT_FRAC": 0.7}

if __name__ == "__main__":
    r = core.run("248", PARAMS)
    if r:
        print(f"248: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
