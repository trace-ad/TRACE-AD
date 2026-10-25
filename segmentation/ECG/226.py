#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 226 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 226.py   # -> output/226.pdf, output/226_segments.csv
"""
import core

# User note (same as 225): mit14046longtermecg (period 90). Normal beats with sharp downward QRS (c0) + many PVCs.
# QRS_SHARP_DETECT + QRS_HEIGHT=0.05 + PEAK_DIST_FRAC=0.45: also detect PVC sharp deflections + weak/close beats, resolving merges. SINGLE_FLOOR=0.18:
# absorb noisy normal beats into c0 (keep normal beats in one cluster). PVC_SHAPE_K=2: split PVCs into 2 types by shape (all-shift SBD + k-medoids reassignment).
# After re-examination following a user note (c1-c3 aren't working): K=4 (3 types) destabilizes because the 3rd is a singleton seed, mixing shapes into c1.
# User note: 8000 is not c1 (main PVC) but the initial-hump type. Cluster with reference to 225. -> PVC_TEMPLATE="225" assigns 226's PVCs
# to 225's PVC medoids (main / initial-hump / biphasic) by nearest assignment (cross-series consistent classification). K=4 {c0:878 normal, c1:145 main PVC,
# c2 initial-hump type (incl. 8000), c3 biphasic}. 8000 matches 225's initial-hump medoid at SBD 0.000.
# User note: 226's initial-hump type has 2 sub-types, first half = deep pre-dip / second half = twin-peak (a subtype absent in 225). PVC_TEMPLATE_SPLIT_TAU=0.30
# sub-splits the template type by shape (with a guard not to split a diverse single type such as the main PVC). K=5 {c0:878 normal, c1:145 main PVC, c2:8 biphasic,
# c3:7 twin-peak type, c4:6 deep-pre-dip type (incl. 8000)} (2026-06-27).
PARAMS = {"QRS_SHARP_DETECT": True, "QRS_HEIGHT": 0.05, "PEAK_DIST_FRAC": 0.45, "SINGLE_FLOOR": 0.18, "PVC_TEMPLATE": "225", "PVC_TEMPLATE_SPLIT_TAU": 0.30}

if __name__ == "__main__":
    r = core.run("226", PARAMS)
    if r:
        print(f"226: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
