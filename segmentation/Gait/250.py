#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 250 (Gait).
Thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 250.py   # -> output/250.pdf, output/250_segments.csv
"""
import core

# weallwalk: 1 step=24 (48 is two steps and looks like "spike+spike+half"). upstroke keeps the peak fixed at the left edge. Flat pauses are aggregated.
PARAMS = {"PERIOD": 24, "ANCHOR_MODE": "upstroke", "KMAX_S1": 1, "KMAX_S2": 1, "FLAT_ARTIFACT_FRAC": 0.7}

if __name__ == "__main__":
    r = core.run("250", PARAMS)
    if r:
        print(f"250: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
