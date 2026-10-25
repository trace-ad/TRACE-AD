#!/usr/bin/env python
"""Run TFVision-AD over a dataset.

The defaults reproduce the single confirmed configuration; every knob is a CLI
option so ablations vary one thing at a time.  Per-series channel scores and a
per-stage timing breakdown are saved; ``--eval`` also fuses and scores each
series with the point-wise + Range/VUS metrics.

Examples
--------
  # UCR, confirmed config, save scores + timing + metrics
  python run.py --dataset UCR \
      --data-dir /path/UCR_Anomaly_FullData \
      --seg-dir  /path/segments/ECG \
      --out-dir  out/ucr_ecg --eval
"""
import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from tfvision_ad.config import Config                     # noqa: E402
from tfvision_ad import data as data_mod                  # noqa: E402
from tfvision_ad.encode import pick_device, load_clip     # noqa: E402
from tfvision_ad.detect import detect                     # noqa: E402
from tfvision_ad.fuse import fuse                          # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="TFVision-AD runner")
    ap.add_argument("--dataset", choices=["UCR", "TSB"], required=True)
    ap.add_argument("--data-dir", required=True, help="raw signals (UCR .txt / TSB .csv)")
    ap.add_argument("--seg-dir", required=True, help="dir with {sid}_segments.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--series", nargs="*", default=None,
                    help="series ids (default: all under --seg-dir)")
    # --- ablation knobs (defaults = confirmed config) ---
    ap.add_argument("--grid-cols", type=int, default=Config.grid_cols)
    ap.add_argument("--phase-tol", type=int, default=Config.phase_tol)
    ap.add_argument("--ng-window", type=int, default=Config.ng_window)
    ap.add_argument("--vscale", type=float, default=Config.vscale)
    ap.add_argument("--tau", type=float, default=Config.tau)
    ap.add_argument("--sbd-len", type=int, default=Config.sbd_len)
    ap.add_argument("--sbd-margin", type=int, default=Config.sbd_margin)
    ap.add_argument("--channels", default=",".join(Config.channels),
                    help="comma list from {amp_fine,sh,amp_ng}")
    ap.add_argument("--ref", default=Config.ref)
    ap.add_argument("--ref-count", type=int, default=None,
                    help="cap clean beats per cluster (N_ref); default keeps all")
    ap.add_argument("--ref-strategy", choices=["random", "medoid", "diverse"], default=None,
                    help="how to pick ref-count beats within a cluster")
    ap.add_argument("--ref-seed", type=int, default=0)
    ap.add_argument("--ng-causal", action="store_true",
                    help="AR-style past-anchored amp_ng window; saves amp_ng{i,e}{k}")
    ap.add_argument("--ng-sweep", default="",
                    help="comma list of amp_ng widths to compute in one pass, e.g. 3,5,7,11")
    ap.add_argument("--no-edge-relax", action="store_true")
    ap.add_argument("--clip-model", default=Config.clip_model)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval", action="store_true", help="fuse + score each series")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        clip_model=args.clip_model, batch_size=args.batch_size,
        grid_cols=args.grid_cols, phase_tol=args.phase_tol,
        channels=tuple(args.channels.split(",")), ng_window=args.ng_window,
        ng_causal=args.ng_causal,
        ng_sweep=tuple(int(x) for x in args.ng_sweep.split(",")) if args.ng_sweep else (),
        vscale=args.vscale, tau=args.tau, sbd_len=args.sbd_len,
        sbd_margin=args.sbd_margin, ref=args.ref, edge_relax=not args.no_edge_relax,
        ref_count=args.ref_count, ref_strategy=args.ref_strategy, ref_seed=args.ref_seed,
    )
    device = args.device or pick_device()
    print(f"[TFVision-AD] device={device} channels={cfg.channels}", flush=True)
    model = load_clip(cfg.clip_model, device)
    loader = data_mod.LOADERS[args.dataset]
    ids = args.series or data_mod.list_series(args.seg_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    evaluate = None
    if args.eval:
        from evaluation.metrics import evaluate  # noqa: E402

    summary = []
    for sid in ids:
        loaded = loader(args.data_dir, args.seg_dir, sid)
        if loaded is None:
            continue
        sig, te, y, rows = loaded
        if y.sum() == 0:
            continue
        out = detect(sig, te, rows, cfg, model, device)
        if out is None:
            continue
        scores, ul, T, tm, ref_spans = out
        np.savez(os.path.join(args.out_dir, f"{sid}.npz"),
                 y=y.astype(np.int8), ul=ul,
                 **{k: v.astype(np.float32) for k, v in scores.items()})
        with open(os.path.join(args.out_dir, f"{sid}.timing.json"), "w") as fh:
            json.dump(tm, fh, indent=2)
        row = {"sid": sid, "T": T, "t_total": round(tm["total"], 3),
               "t_ref": round(tm["ref_total"], 3), "t_detect": round(tm["detect_total"], 3),
               "nref_used": tm.get("nref_used", 0)}
        if evaluate is not None:
            fused = fuse(scores, ul, sig, te, rows, T, ref_spans)
            row.update({k: round(v, 4) for k, v in evaluate(y, fused).items()})
        summary.append(row)
        print(row, flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if summary and evaluate is not None:
        keys = [k for k in summary[0] if k not in ("sid", "T")]
        print("\n== mean over %d series ==" % len(summary), flush=True)
        for k in keys:
            vals = [r[k] for r in summary if k in r]
            print(f"  {k:10s} {np.mean(vals):.4f}", flush=True)


if __name__ == "__main__":
    main()
