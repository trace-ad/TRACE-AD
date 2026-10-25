#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 172 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 172.py   # -> output/172.pdf, output/172_segments.csv
"""
import core
PARAMS = {"PERIOD": 297, "QRS_SHARP_DETECT": True}
# Period misdetected as 74 (~1/4 of the true ~297) -> each beat split into several, too many K_art. Fix to AC main peak 297.
# Use SHARP = derivative-energy R detection so large T-wave domes are not picked up as extra R peaks.
if __name__ == "__main__":
    r = core.run("172", PARAMS)
    if r:
        print(f"172: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
