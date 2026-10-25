#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series 212 (Power Demand): base-waveform segmentation + clustering. A thin wrapper that calls core.
  python 212.py   # -> output/212.pdf, output/212_segments.csv
"""
import core
# c0/c1/c3 are the same week type with nearly identical shape (medoid full-SBD 0.05-0.11), but c3 has slightly lower amplitude
# (median range ratio 0.29-0.34). With the default STAGE2_AMP_RANGE=0.25, c3 remains an amplitude submode and Kp=3.
# this amplitude fluctuation is weaker than a true low-demand-week submode (e.g. 046 c1=0.52), so loosen to 0.40 and merge as the same week type (Kp=2).
PARAMS = {"STAGE2_AMP_RANGE": 0.40}
if __name__ == "__main__":
    r = core.run("212", PARAMS)
    if r:
        print(f"212: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
