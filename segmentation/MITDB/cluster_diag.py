#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate evaluation diagnostic figures/tables for two-stage clustering.

So that the evaluation agent can judge by eye, from the mean shapes, whether a cluster
subdivision is a real submode or a threshold/noise-driven false split, for each final cluster:

  Top row: overlay at RAW amplitude (Stage2 subdivides by amplitude -> cannot be evaluated if amplitude is hidden)
  Bottom row: overlay of robust z-norm shapes + mean (Stage1 type = confirm the shape is the same)
  Mean D3 features per cluster [range, activity, curv, n_extrema, skew]

are combined into one figure, also annotating the Stage1/Stage2 structure (K_s1, K_sub).

  python cluster_diag.py --all
  python cluster_diag.py --series 005 113
Output: diagnostics/<id>_2stage.png, diagnostics/cluster_metrics.csv
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib

import core as W

DIAG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnostics")
FNAMES = ["range", "activity", "curv", "n_ext", "skew"]


def _series_params(sid):
    """Get PARAMS from `<id>.py` (empty if none). Keeps the diagnostic figure consistent with production (run_all)."""
    try:
        return getattr(importlib.import_module(sid), "PARAMS", {}) or {}
    except Exception:
        return {}


def process(series_id):
    """Reproduce the two-stage clustering using only clean units and emit one diagnostic figure
    (artifacts are managed separately, so they are not included in the evaluation figure). Honors series-specific PARAMS."""
    g = W.__dict__
    saved = {k: g[k] for k in W.TUNABLE}
    try:
        for k, v in _series_params(series_id).items():
            if k in W.TUNABLE:
                g[k] = v
        return _process(series_id)
    finally:
        for k, v in saved.items():
            g[k] = v


def _process(series_id):
    path = W.find_file(series_id)
    x = np.loadtxt(path).astype(float)
    te, _, _ = W.parse_meta(path)
    xt = x[:te] if te > 200 else x
    W._QRS_SIGNAL = xt                              # R-peak detection always on the raw signal (matches process, avoids double smoothing)
    if W.PRE_SMOOTH_WN > 0:                          # Same high-frequency ripple removal as process (keeps the diagnostic figure consistent with production)
        bb, aa = W.butter(3, W.PRE_SMOOTH_WN)
        xt = W.filtfilt(bb, aa, xt)
    period = W.PERIOD if W.PERIOD else W.dominant_period(xt)
    spans, clean, _ = W.segment_cycles(xt, period)
    ci = [i for i, c in enumerate(clean) if c]
    if len(ci) < 3:
        return None
    beats_raw = [xt[spans[i][0]:spans[i][1] + 1] for i in ci]
    shapes_rz = np.array([W.resample(W.robust_znorm(b)) for b in beats_raw])
    lab, K, info, _ = W.cluster_two_stage(beats_raw, shapes_rz)
    F = np.array([W.d3_features(b) for b in beats_raw])

    fig, axs = plt.subplots(2, K, figsize=(3.6 * K, 5.6), squeeze=False)
    cmap = plt.get_cmap("tab10")
    rows = []
    for c in range(K):
        idx = np.where(lab == c)[0]
        col = cmap(c % 10)
        step = max(1, len(idx) // 40)
        # Top row: RAW amplitude
        ax = axs[0, c]
        for i in idx[::step]:
            ax.plot(beats_raw[i], color=col, alpha=0.25, lw=0.6)
        fmean = F[idx].mean(0)
        ax.set_title(f"cluster {c}  n={len(idx)}\nRAW amp  range~{fmean[0]:.1f}", fontsize=8)
        ax.set_xticks([])
        # Bottom row: robust z-norm shape + mean
        ax2 = axs[1, c]
        for i in idx[::step]:
            ax2.plot(shapes_rz[i], color=col, alpha=0.2, lw=0.5)
        ax2.plot(shapes_rz[idx].mean(0), color="black", lw=1.4)
        ax2.set_title("  ".join(f"{n}={v:.2f}" for n, v in zip(FNAMES, fmean)), fontsize=6)
        ax2.set_xticks([]); ax2.set_yticks([])
        rows.append((c, len(idx), fmean))
    fig.suptitle(f"{series_id}  K={K}  (Stage1 K_s1={info['K_s1']}, Stage2 K_sub={info['K_sub']})"
                 f"   n_clean={len(ci)}  period={period}", fontsize=11)
    fig.tight_layout()
    os.makedirs(DIAG_DIR, exist_ok=True)
    fig.savefig(os.path.join(DIAG_DIR, f"{series_id}_2stage.png"), dpi=110)
    plt.close(fig)
    return {"series": series_id, "K": K, "info": info, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    ids = W.domain_series_ids() if args.all else (args.series or ["127", "167", "170", "181", "248", "239"])
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
