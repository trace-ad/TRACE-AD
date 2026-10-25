#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 224 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 224.py   # -> output/224.pdf, output/224_segments.csv
"""
import core

# User note: loose segmentation (part of c1 = valley->hump segments whose valley isn't captured) + cluster check. mit14046longtermecg (period 95).
# Normal beats with sharp upward QRS (c0) + valley->wide-hump PVC (c1). There are runs of consecutive PVCs (bigeminy-like). QRS_SHARP_DETECT + QRS_HEIGHT=0.05
# also detects the PVC's weak valleys / close beats, making the whole PVC complex from the valley one segment (merge resolved, valley captured).
# * Important: with the default height, "beats whose valley was captured vs beats whose valley was cut off" formed separate clusters, giving a false K=3. Aligning the valleys
# converges the PVCs to a single morphology, so K=2 is correct (resolving false splits from segmentation inconsistency).
# * Additional user note: in the PVC runs (bigeminy) at 69000-71000, the PVC (no sharp QRS, valley->wide hump) is not detected as an R-wave, so the cut falls at the valley bottom
# and the valley goes to the previous segment. VALLEY_RECLAIM=2.0 moves a cut that is 2*MAD below baseline back to the downslope start (TP baseline) to capture the valley (deep-trough 5->0).
# {c0:728 normal, c1:76 PVC} K=2 (2026-06-27).
PARAMS = {"QRS_SHARP_DETECT": True, "QRS_HEIGHT": 0.05, "KMAX_S2": 1, "VALLEY_RECLAIM": 2.0}

if __name__ == "__main__":
    r = core.run("224", PARAMS)
    if r:
        print(f"224: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
