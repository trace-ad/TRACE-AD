#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 175 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 175.py   # -> output/175.pdf, output/175_segments.csv
"""
import core
PARAMS = {"PERIOD": 297, "QRS_SHARP_DETECT": True}
# Automatic R-R detection picks up T waves/noise as extra R peaks and misdetects the period as 30 -> each beat (true ~297) split into ~10, K_art=25.
# Fix to AC main peak 297. SHARP = derivative-energy R detection avoids misdetecting large T waves and stabilizes the phase.
if __name__ == "__main__":
    r = core.run("175", PARAMS)
    if r:
        print(f"175: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
