#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 236 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 236.py   # -> output/236.pdf, output/236_segments.csv
"""
import core

# User note: revise clusters after improving segmentation. mit14157longtermecg. QRS_SHARP_DETECT + PERIOD=115.
# Same 3 types as 237: c0 = normal beat (upward QRS), c1 = deep-dip type (up-spike + deep downward T inversion, @1035 etc.), 
# c2 = downward-spike type (-25, ectopic beat of opposite polarity, @1269/3569/9847). silhouette is continuous and collapses to K=1, so
# FORCE_SHAPE_K=3 splits by shape into 3.
# Additional user note: @10647 (incl. 10750) is a distinct shape within the c1 deep-dip. Investigation: the 4 deep-dip beats are 2 sub-shapes, {3344,9616 sharp deep-dip right after QRS}
# and {1035,10647 broad late deep-dip} (inter-core SBD 0.29-0.32). Per user instruction, split c1 into 2.
# PVC_SHAPE_K=3 splits the 7 non-normal beats (4 deep-dip + 3 down-spike) by shape into 3.
# K=4 {c0:161 normal, c1:3 down-spike, c2:2 sharp deep-dip, c3:2 broad deep-dip (@1035/10647)} (2026-06-27).
PARAMS = {"PERIOD": 115, "QRS_SHARP_DETECT": True, "FORCE_SHAPE_K": 3, "PVC_SHAPE_K": 3}

if __name__ == "__main__":
    r = core.run("236", PARAMS)
    if r:
        print(f"236: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
