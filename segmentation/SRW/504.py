#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRW cleantrain series 504: segmentation + clustering."""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("504", PARAMS)
    if r:
        print(f"504: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
