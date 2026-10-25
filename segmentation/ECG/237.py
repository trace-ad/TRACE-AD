#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base-waveform segmentation + clustering for series 237 (ECG).
A thin wrapper that just calls the shared engine core. Series-specific tuning is done via PARAMS.

  python 237.py   # -> output/237.pdf, output/237_segments.csv
"""
import core

# User note: merge the 2nd segment (done) + revise clusters = K=3. mit14157longtermecg. QRS_SHARP_DETECT + PERIOD=106.
# Fixed rule (b) to "re-split only when there are 2+ QRS", resolving the fragmentation of the lead-in partial [0:160].
# Same 3 types as 236: c0 = normal beat, c1 = deep-dip type (up-spike + deep downward T inversion, @10866 etc., 14 beats), 
# c2 = downward-spike type (-20, ectopic beat, @14477/15125/15781/16883). FORCE_SHAPE_K=3 splits by shape into 3.
# K=3 {c0:162, c1:14, c2:4} (+ lead-in partial @0 as an artifact) (2026-06-27).
PARAMS = {"PERIOD": 106, "QRS_SHARP_DETECT": True, "FORCE_SHAPE_K": 3}

if __name__ == "__main__":
    r = core.run("237", PARAMS)
    if r:
        print(f"237: K={r['K']}  segments={r['n_segments']} "
              f"(clean={r['n_clean']})  period={r['period']}")
