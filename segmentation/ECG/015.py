#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 015 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 015.py   # -> output/015.pdf, output/015_segments.csv
"""
import core

# User note: 015 has no sharp QRS but repeats a gentle large hump of period ~470 (the RR detector mis-estimated 25/167 from on-hump noise;
# the true period is autocorrelation 469). With PERIOD=470, one hump = one segment (cut at the trough). The leading humps c1/c2, having no QRS,
# fall outside is_cycle_shape and are non-clean, but have the same shape as c0 (SBD0.13). SINGLE_FLOOR=0.16 merges them into c0 (2026-06-26).
PARAMS = {"PERIOD": 470, "SINGLE_FLOOR": 0.16}

if __name__ == "__main__":
    r = core.run("015", PARAMS)
    if r:
        print(f"015: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
