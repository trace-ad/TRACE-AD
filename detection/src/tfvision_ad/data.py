"""Dataset adapters.

Both return ``(signal, train_end, y, seg_rows)``:
  signal    : 1-D float array (full series)
  train_end : index where the test region starts (train = signal[:train_end])
  y         : int label array over the test region (length len(signal)-train_end)
  seg_rows  : rows of ``{sid}_segments.csv`` (region/clean/cluster/start/end)
"""
import csv
import glob
import os

import numpy as np


def _seg_rows(seg_dir, sid):
    with open(os.path.join(seg_dir, f"{sid}_segments.csv")) as fh:
        return list(csv.DictReader(fh))


def load_ucr(data_dir, seg_dir, sid):
    """UCR Anomaly Archive: ``{idx}_UCR_Anomaly_{name}_{te}_{a0}_{a1}.txt``."""
    hits = glob.glob(os.path.join(data_dir, f"{sid}_*.txt"))
    if not hits:
        return None
    parts = os.path.basename(hits[0]).replace(".txt", "").split("_")
    try:
        te, a0, a1 = int(parts[-3]), int(parts[-2]), int(parts[-1])
    except ValueError:
        return None
    sig = np.loadtxt(hits[0])
    T = len(sig) - te
    if T < 50:
        return None
    y = np.zeros(T, int)
    y[max(0, a0 - te):min(T, a1 - te + 1)] = 1
    return sig, te, y, _seg_rows(seg_dir, sid)


def load_tsb(data_dir, seg_dir, sid):
    """TSB-AD-U CSV: columns ``Data``/``Label``; train_end in the filename."""
    import pandas as pd
    hits = glob.glob(os.path.join(data_dir, f"{sid}_*.csv"))
    if not hits:
        return None
    path = hits[0]
    te = int(os.path.basename(path).replace(".csv", "").split("_")[-3])
    df = pd.read_csv(path).dropna()
    sig = df["Data"].values.astype(float)
    lab = df["Label"].astype(int).values
    T = len(sig) - te
    if T < 50:
        return None
    y = lab[te:te + T].astype(int)
    return sig, te, y, _seg_rows(seg_dir, sid)


LOADERS = {"UCR": load_ucr, "TSB": load_tsb}


def list_series(seg_dir):
    """Series ids that have a ``*_segments.csv`` under ``seg_dir``."""
    return sorted(os.path.basename(p).replace("_segments.csv", "")
                  for p in glob.glob(os.path.join(seg_dir, "*_segments.csv")))
