#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 220 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 220.py   # -> output/220.pdf, output/220_segments.csv
"""
import core

# User note: segment with a longer period. STAFFIII has wide P/T + sharp downward QRS (period ~890). The RR detector picked up in-beat ripple
# (period 24) and extremely over-segmented (n_seg=1743). The QRS-spike interval = 890 is the true period. With PERIOD=890,
# one beat = one segment (n_seg=50). All beats same shape (K_s1=1, medoid SBD 0.180), so K=2 is an amplitude-only false split -> KMAX fixes K=1 (2026-06-27).
PARAMS = {"PERIOD": 890, "KMAX_S1": 1, "KMAX_S2": 1}

if __name__ == "__main__":
    r = core.run("220", PARAMS)
    if r:
        print(f"220: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
