#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SRW cleantrain series 501: segmentation + clustering."""
import core
PARAMS = {}
if __name__ == "__main__":
    r = core.run("501", PARAMS)
    if r:
        print(f"501: K={r['K']}  segments={r['n_segments']} (clean={r['n_clean']})  period={r['period']}")
