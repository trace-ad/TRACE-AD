#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 123 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 123.py   # -> output/123.pdf, output/123_segments.csv
"""
import core

# User note: 123 has no sharp QRS but repeats a large hump (ECG4) of period ~469. The RR detector picked up noise/sub-peaks
# and mis-estimated period=113, over-segmenting each hump. Autocorrelation 469 is the true period. With PERIOD=469, one hump = one segment.
# The leading partial is reclaimed into c0 with SINGLE_FLOOR=0.16 (2026-06-26).
PARAMS = {"PERIOD": 469, "SINGLE_FLOOR": 0.16}

if __name__ == "__main__":
    r = core.run("123", PARAMS)
    if r:
        print(f"123: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
