#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 057 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 057.py   # -> output/057.pdf, output/057_segments.csv
"""
import core

# K=2 is appropriate (the evaluation AI also sees c0/c1 2 types, 2026-06-26).
# AMP_GATE_ASSIGN: a short normal beat crammed right before a PVC (large beat c1) gets misassigned to c1 under the amplitude-discarding SBD;
#   the amplitude gate pulls it back to the normal c0 side (2026-08-10).
PARAMS = {"KMAX_S1": 2, "KMAX_S2": 1, "AMP_GATE_ASSIGN": True}

if __name__ == "__main__":
    r = core.run("057", PARAMS)
    if r:
        print(f"057: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
