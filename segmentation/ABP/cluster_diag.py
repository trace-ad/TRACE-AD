#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate diagnostic figures/tables for two-stage clustering.

For each series' final clusters, combine into one figure:
- top: overlay at RAW amplitude (Stage2 subdivides by amplitude, so it cannot be evaluated without visible amplitude)
- bottom: overlay of robust z-norm shapes (check whether the Stage1 type = shape is the same)
- mean D3 features per cluster (range=amplitude, activity, curvature, n_extrema, skew)
and annotate the Stage1/Stage2 structure.

  python cluster_diag.py --all
Output: diagnostics/<id>_2stage.png, diagnostics/cluster_metrics.csv
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core as W

DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")


def process(series_id):
    path = W.find_file(series_id)
    x = np.loadtxt(path).astype(float)
    te, _, _ = W.parse_meta(path)
    xt = x[:te] if te > 200 else x
    period = W.PERIOD if W.PERIOD else W.dominant_period(xt)
    spans, clean, _ = W.segment_cycles(xt, period)
    ci = [i for i, c in enumerate(clean) if c]
    if len(ci) < 2 * W.MIN_PATTERN:
        # also emit one figure for small series that are not subdivided
        ci = [i for i, c in enumerate(clean) if c]
    if len(ci) < 3:
        return None
    beats_raw = [xt[spans[i][0]:spans[i][1] + 1] for i in ci]
    shapes_rz = np.array([W.resample(W.robust_znorm(b)) for b in beats_raw])
    lab, K, info, _ = W.cluster_two_stage(beats_raw, shapes_rz)

    # D3 features per beat
    F = np.array([W.d3_features(b) for b in beats_raw])
    fnames = ["range", "activity", "curv", "n_ext", "skew"]

    fig, axs = plt.subplots(2, K, figsize=(3.6 * K, 5.6), squeeze=False)
    cmap = plt.get_cmap("tab10")
    rows = []
    for c in range(K):
        idx = np.where(lab == c)[0]
        col = cmap(c % 10)
        # top: RAW amplitude
        ax = axs[0, c]
        for i in idx[:: max(1, len(idx) // 40)]:
            ax.plot(beats_raw[i], color=col, alpha=0.25, lw=0.6)
        fmean = F[idx].mean(0)
        ax.set_title(f"cluster {c}  n={len(idx)}\nRAW amp  "
                     f"range~{fmean[0]:.0f}", fontsize=8)
        ax.set_xticks([])
        # bottom: robust z-norm shape
        ax2 = axs[1, c]
        for i in idx[:: max(1, len(idx) // 40)]:
            ax2.plot(shapes_rz[i], color=col, alpha=0.2, lw=0.5)
        ax2.plot(shapes_rz[idx].mean(0), color="black", lw=1.4)
        ax2.set_title("  ".join(f"{n}={v:.2f}" for n, v in zip(fnames, fmean)),
                      fontsize=6)
        ax2.set_xticks([]); ax2.set_yticks([])
        rows.append((c, len(idx), fmean))
    fig.suptitle(f"{series_id}  K={K}  (Stage1 K_s1={info['K_s1']}, "
                 f"Stage2 K_sub={info['K_sub']})   n_clean={len(ci)}  period={period}",
                 fontsize=11)
    fig.tight_layout()
    os.makedirs(DIAG_DIR, exist_ok=True)
    fig.savefig(os.path.join(DIAG_DIR, f"{series_id}_2stage.png"), dpi=110)
    plt.close(fig)
    return {"series": series_id, "K": K, "info": info, "rows": rows, "fnames": fnames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ids = W.abp_series_ids() if args.all else (args.series or ["024", "092", "245", "197"])
    os.makedirs(DIAG_DIR, exist_ok=True)
    with open(os.path.join(DIAG_DIR, "cluster_metrics.csv"), "w") as f:
        f.write("series,K,K_s1,K_sub,cluster,size,range,activity,curv,n_ext,skew\n")
        for sid in ids:
            r = process(sid)
            if not r:
                print(f"{sid}: skip"); continue
            print(f"{sid}: K={r['K']} K_s1={r['info']['K_s1']} K_sub={r['info']['K_sub']}")
            for c, sz, fm in r["rows"]:
                f.write(f"{sid},{r['K']},{r['info']['K_s1']},\"{r['info']['K_sub']}\","
                        f"{c},{sz},{fm[0]:.3f},{fm[1]:.3f},{fm[2]:.3f},{fm[3]:.1f},{fm[4]:.3f}\n")
    print(f"-> diagnostics in {DIAG_DIR}")


if __name__ == "__main__":
    main()
