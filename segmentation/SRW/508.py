#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRW cleantrain series 508: segmentation + clustering."""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("508", PARAMS)
    if r:
        print(f"508: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
