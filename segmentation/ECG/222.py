#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 222 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 222.py   # -> output/222.pdf, output/222_segments.csv
"""
import core

# User note (multiple rounds): revise segmentation + K>=2, further split 17250 into 2, K=3. mit14046longtermecg (period 98).
# Normal beats with sharp narrow QRS (c0) + 2 types of PVC/ectopic beats (morphologies of different ventricular origin, ~10-11 each, silhouette 0.92).
# QRS_SHARP_DETECT=True: detect the PVC's sharp valley as an R-wave -> the cut window lands on the TP baseline, making the whole PVC complex (valley included) one segment
#   (resolving the valley split at 14750). QRS_HEIGHT=0.08: also detect weak PVCs (e.g. 17340) buried under energy normalization because the normal QRS is as tall as +5
#   (splitting [17232:17418], where a normal beat + PVC were merged, into 2). No KMAX = Stage2 legitimately splits the PVCs into 2 types, K=3.
# User confirmation (2026-06-27): after re-examining c1-c3, the true PVC types are 2 (type A with an initial spike / type B without), confirmed = K=3 kept.
# The apparent split in c1 comes from 2 mis-segmented beats (@3763 split at h=0.05; @55088 the PVC valley is close to a normal beat with a weak gradient and
# cannot be detected, so it remains in c1 as a merged PVC+normal QRS) and is not a true 3rd type. Accepted by user judgment (K=3 kept).
PARAMS = {"QRS_SHARP_DETECT": True, "QRS_HEIGHT": 0.05}

if __name__ == "__main__":
    r = core.run("222", PARAMS)
    if r:
        print(f"222: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
