#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-run every series in this domain, honoring each `<id>.py`'s PARAMS.
`core.py --all` uses only the core defaults (ignoring series-specific PARAMS),
so use this runner whenever some series carry their own PARAMS.

  python run_all.py
  python run_all.py --series <id> <id>
"""
import argparse
import importlib
import json
import os

import core


def series_params(sid):
    """Fetch PARAMS from `<id>.py` (empty if absent)."""
    try:
        mod = importlib.import_module(sid)
        return getattr(mod, "PARAMS", {}) or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+")
    args = ap.parse_args()
    ids = args.series or core.domain_series_ids()

    results = []
    for sid in ids:
        r = core.run(sid, series_params(sid), save=True)
        if r:
            results.append(r)

    os.makedirs(core.OUT_DIR, exist_ok=True)
    with open(os.path.join(core.OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(core.OUT_DIR, "summary.csv"), "w") as f:
        f.write("series,train_len,period,n_segments,n_clean,K\n")
        for r in results:
            f.write(f"{r['series']},{r['train_len']},{r['period']},"
                    f"{r['n_segments']},{r['n_clean']},{r['K']}\n")
    with open(os.path.join(core.OUT_DIR, "segments.csv"), "w") as f:
        f.write("series,seg_index,start,end,cluster,clean,region\n")
        for r in results:
            for s in r["segments"]:
                f.write(f"{r['series']},{s['i']},{s['start']},{s['end']},"
                        f"{s['cluster']},{int(s['clean'])},{s.get('region', 'train')}\n")
    print(f"\n-> {len(results)} series. output in {core.OUT_DIR}")


if __name__ == "__main__":
    main()
