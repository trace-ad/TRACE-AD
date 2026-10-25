#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 174 (TSB-AD-U / MITDB) ECG basic-waveform segmentation + clustering.
A thin wrapper that just calls the shared engine core. Series-specific tuning goes in PARAMS.
  python 174.py   # -> output/174.pdf, output/174_segments.csv
"""
import core
PARAMS = {"PERIOD": 287, "QRS_SHARP_DETECT": True}
# Period misdetected as 89 (~1/3 of the true ~287) -> beats split within themselves. Fix to AC main peak 287.
# Use SHARP = derivative-energy R detection so the broad hump after a biphasic QRS is not picked up as an extra R.
if __name__ == "__main__":
    r = core.run("174", PARAMS)
    if r:
        print(f"174: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
