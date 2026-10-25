#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESP basic-waveform segmentation + clustering engine (core) -- built from scratch
================================================================

A shared engine that splits a respiration (Respiration Rate) waveform into
"basic waveform patterns = 1 breath" and clusters them by shape. Per-series
execution is handled by a thin `<id>.py` (which just calls core.run).

RESP signal characteristics (differences from ABP)
--------------------------------
- 1 unit = 1 breath. A smooth quasi-periodic signal, a **unimodal asymmetric
  wave** (a sharp inhalation peak + a gentle exhalation trough). In this data
  the period is ~470 steps.
- There is no morphology assumption such as ABP's "systolic peak located toward
  the front" or "dicrotic notch". The peak position can vary by series, so no
  phase-position filter is imposed.
- Some series (NOISE type) have high-frequency noise superimposed. The clean
  decision uses the smoothed shape, so normal breaths are not wrongly removed
  even in noise-contaminated series.

Segmentation policy
--------
- Learn the basic waveform only from the training region (UCR: the leading
  train_end is normal).
- Treat foot (trough) to next foot as 1 breath, including the signal endpoints
  as boundaries so the entire range is covered with no gaps. To avoid
  over-segmentation, a lower bound (~0.55*period) is imposed on peak spacing.
- To be robust against amplitude drift, detect peaks on a locally
  amplitude-normalized signal so that low-amplitude breaths are not missed
  (preventing under-segmentation).
- clean filter: exclude segments whose length deviates from the period /
  saturated trapezoids / multi-peak noise fragments.

Clustering (structure, operations, and metrics are fixed; only parameters vary per series)
----------------------------------------------------------------
- Stage1 = shape (ignoring amplitude): robust z-norm + common-length resample,
  full-shift SBD + complete-linkage, K_s1 by relative silhouette comparison over {1..KMAX_S1}.
- Stage2 = amplitude/scale + fine-shape statistics: globally standardize D3
  (5 features, RAW) over all units -> Mahalanobis whiten -> Ward. Subdivide each
  Stage1 cluster with K_sub {1..KMAX_S2}.
  Acceptance guard: adopt only when there is a shape difference (medoid
  full-SBD>STAGE2_SBD) or a real amplitude submode (range relative diff>STAGE2_AMP_RANGE
  and activity/curv relative diff>threshold).
- Artifact clusters: group non-clean segments by robust z-norm shape SBD
  complete-linkage (SBD<ARTIFACT_TAU = same shape), and number them after the
  pulse clusters.

Usage
------
  python 076.py            # process series 076 -> output/076.pdf etc.
  python core.py --all     # batch-process all RESP series
  python core.py --series 076 186
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths / subset (TSB-AD-U)
# The TSB-AD-U subset this folder handles = TODS (synthetic sines, 15 series).
# For another subset, duplicate the folder and change SUBSET. Data is CSV (header: Data,Label).
# Filename <gid>_<SUBSET>_id_<n>_<domain>_tr_<train_end>_1st_<x>.csv. The leading train_end is normal (semi-supervised).
# TODS (287..301) are synthetic smooth sines (period ~100, some 300). Amplitude is per-series (value ranges [-2,2] to [-18,-4] etc.).
#   Anomalies are short spikes toward amplitude ~2 (not present in train). train_end is short, 500 or 1250; with period 100 that is 5-12
#   cycles. The period is auto-detected by autocorrelation (dominant_period). Being a single sine pattern, K_pulse=1 is the default.
# Label is used only as an evaluation reference (red-band display of anomaly regions); it is not used for clustering.
SUBSET = "SRW"
DATA_DIR = "./data/SRW"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "output")

# ----------------------------------------------------------------------------- defaults (can be overridden per series via PARAMS)
ROW_STEPS = 2000          # number of time steps per PDF row
ROWS_PER_PAGE = 12        # number of rows per PDF page
RESAMPLE_LEN = 256        # resample each breath to this length for shape comparison

# --- segmentation (for RESP) ---
PERIOD = None             # A2 is a sine but the period is per-series (119-237). Auto-detect per series by autocorrelation.
                          #   For trend-dominated series (TS50/100 etc.), set PERIOD manually in the <id>.py PARAMS.
PEAK_DIST_FRAC = 0.55     # minimum peak spacing = this ratio * period (suppresses over-segmentation)
PEAK_PROM = 0.25          # lower bound on peak prominence on the locally normalized (0..1) signal
LOCALNORM_FRAC = 1.5      # window for local amplitude normalization = this ratio * period
SMOOTH_FRAC = 0.06        # savgol window before peak detection = this ratio * period
LEN_LO_FRAC = 0.55        # clean length lower bound = this ratio * period
LEN_HI_FRAC = 1.7         # clean length upper bound = this ratio * period
FLATTOP_MAX = 0.35        # a flat fraction near the top at or above this = flat trapezoid (excluded)
                          #   The A2 sine has a rounded apex so flat naturally becomes 0.25-0.30; at 0.22 it
                          #     misclassifies normal cycles as trapezoids and wipes them all out. Relaxed to 0.35 (true saturation handled by is_clipped).
NOISE_PEAKS_MAX = 4       # peak count in the smoothed shape above this = multi-peak noise fragment (excluded)
SHAPE_SMOOTH_WIN = 31     # savgol window for peak counting in is_breath_shape (on the 256-resample).
                          #   A2 is a noisy sine, so widen it (11->31) to flatten small noise peaks.
SHAPE_PK_PROM = 0.40      # prominence lower bound for that peak counting (0.2->0.4, noise robustness).
# non-periodic fallback: for series where periodic segmentation yields 0 clean cycles (= trend+noise dominated, no sine),
#   re-split into fixed-length windows with no gaps and treat "1 window = 1 unit" (sliding fixed-window scheme).
#   Noise/trend slices are the normal reference; windows containing spikes are anomalous (train excludes by label).
FIXED_WIN = 140           # window length for the fallback (train~=710 -> about 5 windows; smaller means a richer reference).
MIN_PERIOD = 12           # The true period of real traffic is daily ~24. A detected period below this is noise misdetection
                          #   (e.g. period~4 on dense spike trains) -> treat as non-periodic and fall back to fixed windows (prevents runaway over-segmentation).
MIN_SEGS_ND = 2           # NAB: for a non-daily (autocorr) period, if the train split count is below this it is under-segmented (spurious long lag,
                          #   normal reference depleted) -> fixed windows. Exception when a daily period was significant and selected.
MAX_SEGS = 100000         # Allow many-cycle splitting for NeurIPS-TS type (period25, long train) (22 would wrongly fall back to fixed windows). Original NAB: for a non-daily period, if the train split count exceeds this the window is too fine, a pseudo short period
                          #   (e.g. period36->56 splits) -> fixed windows. Daily periods are an exception.
# Detection of saturated (measurement-range ceiling clip) beats: RESP contains clip beats that stick to the ADC
# ceiling (~+/-2000). These are "saturated trapezoids" = non-clean (a clean-exclusion condition in the spec). The flat
# fraction after z-norm cannot catch narrow clips, so detect directly on the raw waveform by "proximity to the global
# ceiling x length of the consecutive same-value run at the top" and consistently send them to artifact across all
# series (cross-cutting note from evaluation AI 2026-06-24). Normal breaths do not touch the ceiling (run 1-3);
# saturated beats have a run of 10-50.
CLIP_CEIL_FRAC = 0.02     # if the segment peak is at or above global (max - this*amplitude), it is near the ceiling
CLIP_RUN_MIN = 8          # near the ceiling and a near-peak consecutive run at least this long = saturation clip (excluded)

# --- clustering K selection / merging (structure fixed, only thresholds tuned) ---
KMAX_S1 = 3               # Stage1 K_s1 candidate upper bound
KMAX_S2 = 3               # Stage2 K_sub candidate upper bound
MAXLAG_FRAC = 0.10        # lag upper bound for the shift-limited SBD used in nearest-neighbor assignment of non-clean units
MERGE_TAU = 0.10          # if medoid-to-medoid SBD is below this, merge as the same shape
SIL_MIN = 0.12            # if the best silhouette is below this, do not split (K=1)
MIN_PATTERN = 4           # below this size it is a one-off outlier, not a recurring pattern -> absorb
MAHAL_RIDGE = 1e-3        # Stage2 covariance ridge (stabilizes Mahalanobis whiten)
STAGE2_SBD = 0.12         # if sub-cluster medoid-to-medoid full-SBD exceeds this, "shapes differ"
                          # In A3, tiny phase/noise differences of synthetic bimodals land at SBD 0.09-0.11 and cause false splits.
                          # Assuming a single submode, raise to 0.12 to merge tiny differences (clear differences above 0.12 are kept).
STAGE2_AMP_RANGE = 0.28   # if the amplitude range relative diff exceeds this, and ->, it is an "amplitude submode"
                          # A1 has real multimodes in real traffic (busy/idle), so unlike A4 do not fully
                          #   disable the amplitude branch; keep a gentle relaxation 0.18->0.28. This absorbs false splits
                          #   that cut the same z-norm shape by amplitude alone (TS12/53/67 etc.) while keeping real
                          #   distinct-shape modes (TS16/26, or the real spike mode of TS21) via the shape route. By consensus of evaluation agents.
STAGE2_AMP_ACT = 0.15     #   activity relative diff exceeds this
STAGE2_AMP_CURV = 0.35    #   or curv relative diff exceeds this
ARTIFACT_TAU = 0.10       # non-clean segments: if SBD is below this, same-shape artifact
# clean/artifact is a per-beat decision (2026-08-13, user-specified): regardless of K_pulse, decided solely by
#   whether each beat resembles the medoid (representative beat) of its cluster. An incomplete beat mixed into a consistent cluster floats off the medoid -> artifact.
CLEAN_BY_MEDOID = True     # enable per-beat clean decision
CLEAN_SBD = 0.4            # a beat is clean if it is within this SBD of the cluster medoid (~=correlation >=0.6)
CLEAN_MIN_N = 2            # if the cluster size is below this (a non-recurring singleton), all its beats are artifact
# --- singleton re-review rule (decide whether to merge/re-split a size-1 cluster) ---
SINGLE_FACTOR = 1.5       # allowed multiple of the target cluster's internal spread (medoid SBD p90) (relative comparison)
SINGLE_FLOOR = 0.12       # lower bound of the allowed merge SBD (even if the target is small, within this range treat as the same shape)
# --- semi-supervised: granularity for excluding known anomalies (label==1) in the train region from the normal reference ---
# TODS scatters point-spike anomalies (1-2 point excursions to the amplitude ends) across every cycle of the "normal sine train".
#   With the Yahoo-derived default (exclude an entire cycle if even 1 point is anomalous), all train cycles get excluded and
#   the normal reference is depleted. So decide by "the fraction of anomalous points in the cycle length": for point spikes (a few %) do not
#   drop the cycle, and move only regions that are mostly anomalous (>ANOM_EXCL_FRAC) to artifact.
#   The robust z-norm + SBD medoid is robust to a few spike points, so the normal-sine reference is preserved.
ANOM_EXCL_FRAC = 0.30     # if the fraction of anomalous points in a cycle exceeds this, exclude from the normal reference (0.0 = legacy behavior of excluding on even 1 point)


# ============================================================================= IO
def find_file(series_id):
    """Find the CSV from the series ID (= the group id at the head of the filename, e.g. "260")."""
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{series_id}_*.csv")))
    if not hits:
        raise FileNotFoundError(f"series {series_id} not found in {DATA_DIR}")
    return hits[0]


def domain_series_ids():
    """List of series IDs belonging to this SUBSET (the group id at the head of the filename), ascending numeric."""
    files = glob.glob(os.path.join(DATA_DIR, "*_SRW_*.csv"))
    return sorted((os.path.basename(f).split("_")[0] for f in files), key=int)


def parse_meta(path):
    """From the TSB-AD-U filename <gid>_<SUBSET>_id_<n>_<domain>_tr_<train_end>_1st_<x>.csv,
    get train_end (the length of the leading normal region)."""
    base = os.path.basename(path).replace(".csv", "")
    try:
        return int(base.split("_")[-3])
    except ValueError:
        return 0


def _acf_at(x, lag):
    """Autocorrelation at lag `lag` after linear detrending. Used to judge the significance of a daily period."""
    if lag <= 0 or lag >= len(x) - 1:
        return 0.0
    t = np.arange(len(x), dtype=float)
    z = x - np.polyval(np.polyfit(t, x, 1), t)
    z = z - z.mean()
    denom = float((z * z).sum()) + 1e-12
    return float((z[:-lag] * z[lag:]).sum() / denom)


def choose_period(xt):
    """Period decision for TSB-AD-U (IOPS) + non-periodic fallback decision. Returns (period, force_fixed).
    TSB-AD-U CSVs have no timestamp column (only Data,Label), so NAB's daily period
    (daily_period) cannot be used. The period is decided solely by autocorrelation (dominant_period).
    Non-periodic fallback (force_fixed=True):
      - very small period (<MIN_PERIOD): misdetection of a short noise lag.
      - weak autocorrelation at the period (<0.20): effectively non-periodic (gid 265/272 etc.).
      - split count below MIN_SEGS_ND (too few, reference depleted) / above MAX_SEGS (pseudo short period, windows too fine).
    There is no daily-priority exception (because there is no daily_period)."""
    if PERIOD:
        return PERIOD, False
    period = dominant_period(xt)
    est_segs = len(xt) / max(period, 1)
    force_fixed = (period < MIN_PERIOD) or (
        _acf_at(xt, period) < 0.20 or est_segs < MIN_SEGS_ND or est_segs > MAX_SEGS)
    return period, force_fixed


def load_series(path):
    """Read the CSV (header Data,Label) and return the min-max (train-region statistics) normalized values, Label, and train_end
    (following the TSB-AD-U preprocessing in CoAD toolkit/utils.py)."""
    df = pd.read_csv(path).dropna()
    val = df["Data"].values.astype(float)
    lab = df["Label"].astype(int).values
    train_end = parse_meta(path)
    tr = val[:train_end] if train_end > 10 else val
    lo, hi = float(tr.min()), float(tr.max())
    val = (val - lo) / (hi - lo + 1e-12)
    return val, lab, train_end


def label_spans(lab):
    """Return the contiguous Label==1 regions as a list [(s,e), ...] (e exclusive) (for red-band display)."""
    lab = np.asarray(lab).astype(int)
    if lab.sum() == 0:
        return []
    d = np.diff(np.concatenate([[0], (lab == 1).astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))




abp_series_ids = domain_series_ids   # backward-compat alias for cluster_diag etc.


# ============================================================================= segmentation
def dominant_period(x):
    """Estimate the dominant period (samples per cycle) by autocorrelation.
    Among the autocorrelation peaks after the zero crossing, take the lag with the
    highest correlation as the fundamental period (so a half-period small peak is not
    mistaken for the fundamental).
    A2 has trend-dominated series where the raw autocorrelation buries the period, so
      for detection, compute autocorrelation on a **linearly detrended copy** (the signal body stays min-max only, unchanged)."""
    t = np.arange(len(x), dtype=float)
    a, b = np.polyfit(t, x, 1)
    z = x - (a * t + b)                     # detection only: a copy with the linear trend removed
    z = z - z.mean()
    ac = np.correlate(z, z, mode="full")[len(z) - 1:]
    ac /= ac[0] + 1e-12
    zero = np.where(ac < 0)[0]
    lo = zero[0] if len(zero) else 10
    seg = ac[lo:min(len(ac), lo + 5000)]
    if len(seg) < 3:
        return 100
    peaks, props = find_peaks(seg, height=0.0)
    if len(peaks) == 0:
        return int(lo + np.argmax(seg))
    return int(lo + peaks[int(np.argmax(props["peak_heights"]))])


def _local_normalize(xs, win):
    """Normalize to 0..1 locally (min,max) to remove amplitude drift.
    This lets low-amplitude breaths be peak-detected with the same salience as high-amplitude ones."""
    win = max(5, int(win) | 1)
    lo = minimum_filter1d(xs, win, mode="nearest")
    hi = maximum_filter1d(xs, win, mode="nearest")
    return (xs - lo) / (hi - lo + 1e-9)


def segment_cycles(x, period):
    """Split the whole signal into one breath each, foot (trough) to next foot, with no gaps.

    Returns:
      spans  : [(s,e), ...] breaths covering the whole range (including partial breaths at the ends)
      clean  : whether each span is "breath-like (usable for cluster definition)" (bool list)
      peaks  : the detected inhalation peaks
    """
    n = len(x)
    w = max(5, int(period * SMOOTH_FRAC) | 1)
    xs = savgol_filter(x, window_length=w, polyorder=2)
    norm = _local_normalize(xs, win=int(period * LOCALNORM_FRAC))
    peaks, _ = find_peaks(norm, distance=max(3, int(period * PEAK_DIST_FRAC)),
                          prominence=PEAK_PROM)
    if len(peaks) < 3:
        return [], [], peaks
    # take the minimum point (trough) between consecutive peaks as the cut foot
    feet = [peaks[i] + int(np.argmin(xs[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)]
    # add the signal endpoints to the boundaries and cover [0, n] with no gaps
    bounds = sorted(set([0] + feet + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    lo_len, hi_len = int(period * LEN_LO_FRAC), int(period * LEN_HI_FRAC)
    ceiling = x.max() - CLIP_CEIL_FRAC * (x.max() - x.min())   # ceiling-proximity threshold for saturation detection
    clean = [bool(lo_len <= e - s <= hi_len) and is_breath_shape(x[s:e])
             and not is_clipped(x[s:e + 1], ceiling)
             for s, e in spans]
    return spans, clean, peaks


def fixed_windows(n, win):
    """For non-periodic (trend-dominated) series: split the whole range [0,n) into fixed-length `win` with no gaps.
    Returns the same (spans, clean, peaks) form as segment_cycles. clean is all True (no shape filter is
    applied, because the sine assumption does not hold). Label-anomaly exclusion is done by the caller."""
    win = max(20, int(win))
    bounds = list(range(0, n, win))
    if not bounds or bounds[-1] != n:
        bounds.append(n)
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    return spans, [True] * len(spans), np.array([], dtype=int)


def is_clipped(seg, ceiling):
    """Whether it is a saturated (measurement-range ceiling clip) beat. If the segment peak is near the
    global ceiling and there is a long consecutive same-value run near the top (near-peak run), treat it
    as saturation (non-clean). Normal breaths do not reach the ceiling and have a rounded top, so the run is short."""
    if len(seg) < 3 or seg.max() < ceiling:
        return False
    rng = seg.max() - seg.min() + 1e-9
    near = np.abs(seg - seg.max()) < 0.01 * rng
    run = mx = 0
    for v in near:
        run = run + 1 if v else 0
        mx = max(mx, run)
    return mx >= CLIP_RUN_MIN


def is_breath_shape(seg):
    """Whether the shape looks like one cycle (rejects saturated trapezoids and multi-peak noise fragments).
    Features are computed on the smoothed shape so normal cycles in noise-contaminated series are not wrongly removed.
    A2 is a unimodal sine with noise; with weak smoothing (win11/prom0.2) noise's small peaks
      balloon to npk=5-18 and wipe out normal cycles. With win31/prom0.4 the unimodal counts as npk~=1."""
    if len(seg) < 5:
        return False
    m = znorm(resample(seg))
    ms = savgol_filter(m, SHAPE_SMOOTH_WIN, 2)
    rng = ms.max() - ms.min() + 1e-9
    flattop = float((ms > ms.max() - 0.1 * rng).mean())          # flat fraction near the top
    npk = len(find_peaks(ms, prominence=SHAPE_PK_PROM)[0])        # number of true shape peaks
    return flattop < FLATTOP_MAX and npk <= NOISE_PEAKS_MAX


# ============================================================================= shape distance (SBD) / features
def znorm(v):
    s = v.std()
    return (v - v.mean()) / (s + 1e-12)


def robust_znorm(v):
    """Robust z-normalization that discards amplitude: (v - median)/(1.4826*MAD). For Stage1."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)


def resample(v, n=None):
    n = n or RESAMPLE_LEN
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


_ZERO_NORM = 1e-9         # a norm below this is treated as degenerate (flat = all-zero after robust_znorm)


def sbd_full_pair(a, b):
    """Full-shift SBD = 1 - max(normalized cross-correlation). Phase-tolerant (compares by waveform type).
    Degenerate (flat = zero vector) handling: if both are flat, 0 (identical); if only one is flat, 1 (different).
      Without this, two flats would get SBD=1 (max distance), each becomes a singleton, and K explodes."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _ZERO_NORM and nb < _ZERO_NORM:
        return 0.0
    if na < _ZERO_NORM or nb < _ZERO_NORM:
        return 1.0
    cc = np.correlate(a, b, mode="full")
    return float(1.0 - cc.max() / (na * nb))


def sbd_full_matrix(mat):
    """Full-shift SBD distance matrix (Stage1: broadly group by waveform type). Degenerate (flat) follows the same rule as sbd_full_pair."""
    n = len(mat)
    norms = np.linalg.norm(mat, axis=1)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] < _ZERO_NORM and norms[j] < _ZERO_NORM:
                d = 0.0
            elif norms[i] < _ZERO_NORM or norms[j] < _ZERO_NORM:
                d = 1.0
            else:
                cc = np.correlate(mat[i], mat[j], mode="full")
                d = float(1.0 - cc.max() / (norms[i] * norms[j]))
            D[i, j] = D[j, i] = d
    return D


def d3_features(w):
    """Stage2 D3 (5 features, RAW values, segment = the unit itself, no smoothing).
    [0]range=max-min (raw amplitude), [1]activity=mean(|diff|), [2]curv=max(|diff2|),
    [3]n_extrema=#find_peaks(w,prom=0.10*range)+#find_peaks(-w, same),
    [4]skew=mean(z**3) (z=(w-mean)/std, ddof=0). Length <3 gives a zero vector."""
    if len(w) < 3:
        return np.zeros(5)
    rng = float(w.max() - w.min())
    activity = float(np.mean(np.abs(np.diff(w))))
    curvature = float(np.max(np.abs(np.diff(w, n=2))))
    if rng <= 1e-9:
        n_ext, skew = 0, 0.0
    else:
        n_ext = (len(find_peaks(w, prominence=0.10 * rng)[0])
                 + len(find_peaks(-w, prominence=0.10 * rng)[0]))
        z = (w - w.mean()) / (w.std() + 1e-12)
        skew = float(np.mean(z ** 3))
    return np.array([rng, activity, curvature, float(n_ext), skew])


# ============================================================================= K selection / merging
def medoid(idx, D):
    sub = D[np.ix_(idx, idx)]
    return idx[int(sub.sum(1).argmin())]


def relabel(lab):
    order = {c: i for i, c in enumerate(sorted(set(lab), key=lambda c: -(lab == c).sum()))}
    return np.array([order[c] for c in lab])


def select_k_precomputed(D, kmax, linkage_method):
    """K selection by relative silhouette comparison on a precomputed distance matrix (Stage1 SBD) + merge/absorb."""
    n = len(D)
    if n < 2 * MIN_PATTERN:
        return np.zeros(n, dtype=int), {"K": 1, "reason": "too few"}
    Z = linkage(D[np.triu_indices(n, 1)], method=linkage_method)
    best, sils = (1, np.zeros(n, dtype=int), -1.0), {}
    for k in range(2, min(kmax, n - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust") - 1
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(D, lab, metric="precomputed")
        sils[k] = round(float(s), 3)
        if s > best[2]:
            best = (k, lab, s)
    k, lab, sil = best
    if k == 1 or sil < SIL_MIN:
        return np.zeros(n, dtype=int), {"sils": sils, "best_sil": round(sil, 3), "K": 1}
    lab = relabel(absorb_singletons(merge_close(lab, D), D))
    return lab, {"sils": sils, "best_sil": round(sil, 3), "K": len(set(lab))}


def select_k_features(F, kmax):
    """K selection by Ward + relative silhouette comparison on feature vectors (Stage2 whitened D3)."""
    n = len(F)
    if n < 2 * MIN_PATTERN:
        return np.zeros(n, dtype=int), {"K": 1, "reason": "too few"}
    Z = linkage(F, method="ward")
    best, sils = (1, np.zeros(n, dtype=int), -1.0), {}
    for k in range(2, min(kmax, n - 1) + 1):
        lab = fcluster(Z, k, criterion="maxclust") - 1
        if len(set(lab)) < 2:
            continue
        s = silhouette_score(F, lab)
        sils[k] = round(float(s), 3)
        if s > best[2]:
            best = (k, lab, s)
    k, lab, sil = best
    if k == 1 or sil < SIL_MIN:
        return np.zeros(n, dtype=int), {"sils": sils, "best_sil": round(sil, 3), "K": 1}
    lab = relabel(absorb_singletons_feat(lab, F))
    return lab, {"sils": sils, "best_sil": round(sil, 3), "K": len(set(lab))}


def merge_close(lab, D):
    """Merge cluster pairs whose medoid-to-medoid SBD < MERGE_TAU."""
    lab = lab.copy()
    while len(set(lab)) >= 2:
        ids = sorted(set(lab))
        meds = {c: medoid(np.where(lab == c)[0], D) for c in ids}
        closest, best_d = None, 1e9
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = D[meds[ids[i]], meds[ids[j]]]
                if d < best_d:
                    best_d, closest = d, (ids[i], ids[j])
        if best_d < MERGE_TAU:
            lab[lab == closest[1]] = closest[0]
        else:
            break
    return lab


def absorb_singletons(lab, D):
    """Absorb clusters of size < MIN_PATTERN into the cluster of the nearest medoid (SBD space)."""
    lab = lab.copy()
    while True:
        ids = sorted(set(lab))
        if len(ids) < 2:
            break
        sizes = {c: int((lab == c).sum()) for c in ids}
        small = [c for c in ids if sizes[c] < MIN_PATTERN]
        if not small:
            break
        c = min(small, key=lambda c: sizes[c])
        cm = medoid(np.where(lab == c)[0], D)
        target = min([d for d in ids if d != c],
                     key=lambda d: D[cm, medoid(np.where(lab == d)[0], D)])
        lab[lab == c] = target
    return lab


def absorb_singletons_feat(lab, F):
    """Absorb clusters of size < MIN_PATTERN into the nearest centroid (feature space)."""
    lab = lab.copy()
    while True:
        ids = sorted(set(lab))
        if len(ids) < 2:
            break
        sizes = {c: int((lab == c).sum()) for c in ids}
        small = [c for c in ids if sizes[c] < MIN_PATTERN]
        if not small:
            break
        c = min(small, key=lambda c: sizes[c])
        cc = F[lab == c].mean(0)
        target = min([d for d in ids if d != c],
                     key=lambda d: np.linalg.norm(cc - F[lab == d].mean(0)))
        lab[lab == c] = target
    return lab


def _reldiff(a, b):
    return abs(a - b) / (0.5 * (abs(a) + abs(b)) + 1e-9)


def guard_stage2(lab, shapes, F):
    """Stage2 acceptance guard: merge sub-cluster pairs that are "essentially the same in both shape and amplitude submode".
    The keep condition is (a) shape difference: medoid full-SBD > STAGE2_SBD, or
    (b) a real amplitude submode: range relative diff > STAGE2_AMP_RANGE and
       (activity relative diff > STAGE2_AMP_ACT or curv relative diff > STAGE2_AMP_CURV).
    -> merges false splits from crossing the n_extrema threshold or cutting a continuous gradient that only moves in range."""
    lab = lab.copy()
    while len(set(lab)) > 1:
        ids = sorted(set(lab))
        meds = {c: shapes[np.where(lab == c)[0]][
            int(sbd_full_matrix(shapes[np.where(lab == c)[0]]).sum(1).argmin())]
            for c in ids}
        merged = False
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                ca, cb = ids[a], ids[b]
                fa, fb = F[lab == ca].mean(0), F[lab == cb].mean(0)
                shape_d = sbd_full_pair(meds[ca], meds[cb]) > STAGE2_SBD
                amp_d = (_reldiff(fa[0], fb[0]) > STAGE2_AMP_RANGE and
                         (_reldiff(fa[1], fb[1]) > STAGE2_AMP_ACT or
                          _reldiff(fa[2], fb[2]) > STAGE2_AMP_CURV))
                if not (shape_d or amp_d):
                    lab[lab == cb] = ca
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    return relabel(lab)


# ============================================================================= 2-stage clustering
def cluster_two_stage(beats_raw, shapes_robust):
    """Two-stage clustering (structure, operations, and metrics are fixed)."""
    # If cycles are too few (<4), Stage2's covariance/whiten does not hold (A2's long-period, few-cycle series).
    # Return K=1 treating it as a single submode (a pure sine is a valid reference even in small numbers).
    n = len(beats_raw)
    if n < 4:
        info = {"K_s1": 1, "K_sub": {0: 1}, "K": 1, "stage1_sils": {},
                "K_pulse": 1, "K_artifact": 0}
        return np.zeros(n, dtype=int), 1, info, None
    # ---- Stage 1: shape (ignoring amplitude) ----
    D1 = sbd_full_matrix(shapes_robust)
    lab1, info1 = select_k_precomputed(D1, KMAX_S1, "complete")

    # ---- Stage 2: amplitude/scale + fine-shape statistics ----
    F = np.array([d3_features(w) for w in beats_raw])
    Fs = (F - F.mean(0)) / (F.std(0) + 1e-9)                  # global standardization
    cov = np.cov(Fs, rowvar=False) + MAHAL_RIDGE * np.eye(Fs.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    Wmat = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T        # whiten matrix
    Fw = Fs @ Wmat

    final = np.full(len(beats_raw), -1, dtype=int)
    nxt, k_sub = 0, {}
    for c1 in sorted(set(lab1)):
        idx = np.where(lab1 == c1)[0]
        lab2, _ = select_k_features(Fw[idx], KMAX_S2)
        lab2 = guard_stage2(lab2, shapes_robust[idx], F[idx])  # acceptance guard
        k_sub[int(c1)] = len(set(lab2))
        for sub in sorted(set(lab2)):
            final[idx[lab2 == sub]] = nxt
            nxt += 1
    final = relabel(final)
    info = {"K_s1": int(info1["K"]), "K_sub": k_sub, "K": int(len(set(final))),
            "stage1_sils": info1.get("sils", {})}
    return final, len(set(final)), info, D1


def cluster_artifacts(shapes):
    """Group non-clean segments by robust z-norm shape SBD complete-linkage
    (SBD < ARTIFACT_TAU = same shape). Unusual ones are their own cluster even at n=1."""
    n = len(shapes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    D = sbd_full_matrix(shapes)
    Z = linkage(D[np.triu_indices(n, 1)], method="complete")
    return relabel(fcluster(Z, t=ARTIFACT_TAU, criterion="distance") - 1)



# ============================================================================= singleton re-review rule
def _medoid_spread(shapes_rz, idx):
    """The medoid shape of cluster idx and its internal spread (90th percentile of SBD to the medoid).
    For large clusters, evenly subsample to <=80 to keep the O(n^2) SBD down (speedup)."""
    idx = np.asarray(idx)
    if len(idx) == 1:
        return shapes_rz[idx[0]], 0.0
    sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
    D = sbd_full_matrix(shapes_rz[sub])
    mi = int(D.sum(1).argmin())
    return shapes_rz[sub[mi]], float(np.percentile(D[mi], 90))


def _nearest_cluster(shape, labels, shapes_rz, exclude):
    """Return the cluster nearest to `shape` (other than exclude): (SBD, cluster, acceptance threshold).
    Acceptance threshold = max(SINGLE_FLOOR, SINGLE_FACTOR * that cluster's internal spread) <- relative comparison."""
    best = None
    for d in sorted(set(labels)):
        if d == exclude:
            continue
        didx = np.where(labels == d)[0]
        med, spread = _medoid_spread(shapes_rz, didx)
        dist = sbd_full_pair(shape, med)
        thr = max(SINGLE_FLOOR, SINGLE_FACTOR * spread)
        if best is None or dist < best[0]:
            best = (dist, int(d), thr)
    return best


def _split_span(xt, s, e, period):
    """Re-split the span [s,e] (a long under-segmented unit) at its internal troughs and return the sub-spans."""
    seg = xt[s:e]
    if len(seg) < int(2 * period * LEN_LO_FRAC):
        return [(s, e)]
    w = max(5, int(period * SMOOTH_FRAC) | 1)
    xs = savgol_filter(seg, window_length=min(w, len(seg) - (1 - len(seg) % 2)), polyorder=2)
    norm = _local_normalize(xs, win=int(period * LOCALNORM_FRAC))
    feet, _ = find_peaks(-norm, distance=max(3, int(period * PEAK_DIST_FRAC)), prominence=PEAK_PROM)
    if len(feet) < 1:
        return [(s, e)]
    bounds = sorted(set([0] + [int(f) for f in feet] + [len(seg)]))
    # Keep full coverage: do not discard too-short fragments; merge them into the adjacent (next) fragment to tile [s,e] with no gaps.
    subs, i = [], 0
    while i < len(bounds) - 1:
        a, b = bounds[i], bounds[i + 1]
        while (b - a) < int(period * LEN_LO_FRAC) and (i + 2) < len(bounds):
            i += 1
            b = bounds[i + 1]
        subs.append((s + a, s + b))
        i += 1
    return subs


def reconcile_singletons(xt, spans, labels, period, k_orig):
    """singleton/small-cluster re-review rule (added 2026-06-25, generalizing a user request):
    (a) **same-shape recovery**: if an artifact cluster's medoid (label >= k_orig) is within the nearest pulse
        cluster's internal spread (medoid SBD p90) x SINGLE_FACTOR, merge that entire cluster into pulse
        (recovers normal cycles that were judged non-clean due to noise/distortion into c0 etc. Relative comparison).
    (b) **re-split**: an artifact of size 1 and length ~2 periods or more (an under-segmented multi-cycle) is re-split
        at its internal troughs, and each piece is merged into the nearest cluster (standalone artifact if no merge target).
    (c) **keep standalone**: truly distinct shapes (partial/double/unusual) are left as is.
    Returns: the updated spans, labels. (Speedup: cluster medoids are computed only once per iteration.)"""
    spans = [tuple(sp) for sp in spans]
    labels = list(int(l) for l in labels)
    for _ in range(100):
        ids = sorted(set(labels))
        if len(ids) < 2:
            break
        shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
        labarr = np.array(labels)
        med = {c: _medoid_spread(shapes_rz, np.where(labarr == c)[0]) for c in ids}
        pulse_ids = [c for c in ids if c < k_orig]
        acted = False
        # (a) same-shape recovery of an entire artifact cluster into pulse
        for c in [c for c in ids if c >= k_orig]:
            cmed = med[c][0]
            best = None
            for p in pulse_ids:
                pmed, pspread = med[p]
                dist = sbd_full_pair(cmed, pmed)
                thr = max(SINGLE_FLOOR, SINGLE_FACTOR * pspread)
                if best is None or dist < best[0]:
                    best = (dist, p, thr)
            if best and best[0] <= best[2]:
                for i in np.where(labarr == c)[0]:
                    labels[i] = best[1]
                acted = True
                break
        if acted:
            continue
        # (b) re-split a size-1 long artifact -> merge each piece
        for c in [c for c in ids if c >= k_orig and labels.count(c) == 1]:
            i = labels.index(c)
            s, e = spans[i]
            if (e - s) > 1.5 * period:
                sub = _split_span(xt, s, e, period)
                if len(sub) >= 2:
                    sub_lab = []
                    for a, b in sub:
                        ssh = resample(robust_znorm(xt[a:b + 1]))
                        best = None
                        for d in ids:
                            if d == c:
                                continue
                            dmed, dspread = med[d]
                            dist = sbd_full_pair(ssh, dmed)
                            thr = max(SINGLE_FLOOR, SINGLE_FACTOR * dspread)
                            if best is None or dist < best[0]:
                                best = (dist, d, thr)
                        sub_lab.append(best[1] if (best and best[0] <= best[2]) else None)
                    if any(l is not None for l in sub_lab):
                        nxt = max(labels) + 1
                        del spans[i]
                        del labels[i]
                        for (a, b), nl in zip(sub, sub_lab):
                            spans.append((a, b))
                            labels.append(nl if nl is not None else nxt)
                        acted = True
                        break
        if not acted:
            break
    return spans, labels


def _apply_reconcile(xt, spans, labels, period, k_orig, clean_set):
    """Apply reconcile_singletons, re-decide pulse/artifact, and renumber with pulse first.
    A pulse cluster = one where the original pulse label (< k_orig) remained. New labels created by split are artifacts."""
    spans, labels = reconcile_singletons(xt, spans, labels, period, k_orig)
    present = sorted(set(labels))
    pulse_ids = [c for c in present if c < k_orig]
    art_ids = [c for c in present if c >= k_orig]
    order = {}
    for k, c in enumerate(sorted(pulse_ids, key=lambda c: -labels.count(c))):
        order[c] = k
    for k, c in enumerate(sorted(art_ids, key=lambda c: -labels.count(c))):
        order[c] = len(pulse_ids) + k
    labels = [order[c] for c in labels]
    # sort spans by start position (readability: PDF/CSV in chronological order)
    sidx = sorted(range(len(spans)), key=lambda i: spans[i][0])
    spans = [spans[i] for i in sidx]
    labels = [labels[i] for i in sidx]
    return spans, labels, len(pulse_ids), len(art_ids), None
# ============================================================================= driver
def process(series_id, save=True):
    path = find_file(series_id)
    x, label, train_end = load_series(path)         # min-max (train statistics) normalized, Label, train_end
    xt = x[:train_end] if train_end > 200 else x    # learn the basic waveform from the normal training region
    anom_spans = label_spans(label)                 # contiguous Label==1 regions (red-band display, evaluation reference)

    period, short_period = choose_period(xt)
    spans, clean, peaks = segment_cycles(xt, period)
    clean_idx = [i for i, c in enumerate(clean) if c]
    # Yahoo (semi-supervised): exclude cycles overlapping label anomalies from the basic-waveform learning pool.
    #   Excluded cycles become non-clean and go to the artifact cluster (do not put a few train anomalies into the normal reference).
    def _anom_frac(s, e):
        seg = label[s:min(e + 1, len(label))]
        return float(seg.sum()) / max(len(seg), 1)
    n_pool0 = len(clean_idx)
    clean_idx = [i for i in clean_idx
                 if _anom_frac(spans[i][0], spans[i][1]) <= ANOM_EXCL_FRAC]
    n_excl = n_pool0 - len(clean_idx)
    # non-periodic fallback: if periodic segmentation yields 0 clean cycles (= trend+noise dominated, no sine),
    #   or a very-small-period misdetection (short_period), re-split into fixed-length windows (sliding fixed window).
    use_fixed = len(clean_idx) < 1 or short_period
    if use_fixed:
        spans, clean, peaks = fixed_windows(len(xt), FIXED_WIN)
        clean_idx = [i for i, (s, e) in enumerate(spans)
                     if label[s:min(e + 1, len(label))].sum() == 0]   # non-anomaly windows become the normal reference
    if len(clean_idx) < 1:
        print(f"[{series_id}] segmentation failed (period={period}, spans={len(spans)}) "
              f"-- cannot segment")
        return None

    shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    beats_raw = [xt[s:e + 1] for s, e in spans]

    # two-stage clustering on clean units only (not misled by artifacts)
    ci = clean_idx
    labels_c, K, info, _ = cluster_two_stage([beats_raw[i] for i in ci], shapes_rz[ci])

    labels_plot = np.full(len(spans), -1, dtype=int)
    for k, i in enumerate(ci):
        labels_plot[i] = int(labels_c[k])

    # non-clean segments = artifact clusters (numbered after the pulse clusters)
    nc = [i for i in range(len(spans)) if labels_plot[i] < 0]
    n_art = 0
    if nc:
        art = cluster_artifacts(shapes_rz[nc])
        for j, i in enumerate(nc):
            labels_plot[i] = K + int(art[j])
        n_art = int(art.max()) + 1

    # singleton re-review rule: merge/re-split size-1/small clusters (relative comparison). Keep pulse-derived cycles.
    clean_set = set(clean_idx)
    spans, labels_plot, K, n_art, clean_set = _apply_reconcile(
        xt, spans, list(labels_plot), period, K, clean_set)
    labels_plot = np.array(labels_plot)
    K_total = K + n_art
    info["K_pulse"], info["K_artifact"] = K, n_art

    # semi-supervised guarantee: train segments overlapping label anomalies must never enter the normal-reference
    #   medoid, even if reconcile returns them to pulse (normal reference). Move such segments to a dedicated artifact cluster (prevents known anomalies from mixing in).
    anom_train = [i for i, (s, e) in enumerate(spans)
                  if labels_plot[i] < K and _anom_frac(s, e) > ANOM_EXCL_FRAC]
    if anom_train:
        for i in anom_train:
            labels_plot[i] = K_total          # all into a new artifact id
        n_art += 1
        K_total = K + n_art
        info["K_artifact"] = n_art

    # overlay drawing uses std-based z-norm (MAD diverges for saturated units and is hard to read)
    shapes_plot = np.array([znorm(resample(xt[s:e + 1])) for s, e in spans])

    # ---- Also apply the learned segmentation to the test region (from train_end on) and assign each unit to the nearest train cluster medoid ----
    #      (the basic waveform is learned on train only; test is classified by nearest medoid without re-clustering = same structure as VLM-AD).
    #      Assign every test segment to the nearest cluster (do not set -1 whether clean or non-clean).
    #        Candidates are all learned clusters (pulse 0..K-1 + artifact K..K_total-1); take the nearest medoid.
    final_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    medoids = {}
    for c in range(K_total):
        idx = np.where(labels_plot == c)[0]
        if len(idx):
            sub = final_rz[idx]
            medoids[c] = sub[sbd_full_matrix(sub).sum(1).argmin()] if len(idx) > 1 else sub[0]
    # per-beat clean decision: a beat is clean if it is within SBD<=CLEAN_SBD of its cluster medoid and the cluster size >=CLEAN_MIN_N.
    clu_size = {c: int(np.sum(np.asarray(labels_plot) == c)) for c in range(K_total)}
    def _clean_by_medoid(shape, lab):
        if not CLEAN_BY_MEDOID:
            return lab < K
        return clu_size.get(lab, 0) >= CLEAN_MIN_N and \
            float(sbd_full_pair(medoids[lab], shape)) <= CLEAN_SBD
    clean_train = [bool(_clean_by_medoid(final_rz[i], int(labels_plot[i]))) for i in range(len(spans))]
    test_spans, test_labels, clean_test = [], [], []
    x_disp = xt
    if 200 < train_end < len(x) and medoids:
        xtt = x[train_end:].astype(float)
        ts, _tc, _ = (fixed_windows(len(xtt), FIXED_WIN) if use_fixed
                      else segment_cycles(xtt, period))
        # test coverage guarantee: strong-trend series fail trough detection on the test side and ts becomes empty/sparse
        #   (test uncovered, anomalies out of scope). In that case, split with fixed windows using the train period as a fixed pitch to always cover the whole test.
        covered = sum(e - s for s, e in ts)
        if not ts or covered < 0.5 * len(xtt):
            ts, _tc, _ = fixed_windows(len(xtt), period if period and period > 10 else FIXED_WIN)
        for (s, e) in ts:
            shp = resample(robust_znorm(xtt[s:e + 1]))
            lab = min(medoids, key=lambda k: float(sbd_full_pair(medoids[k], shp)))
            test_spans.append((int(s + train_end), int(e + train_end)))
            test_labels.append(int(lab))
            clean_test.append(bool(_clean_by_medoid(shp, int(lab))))
        x_disp = np.concatenate([xt, xtt])

    print(f"[{series_id}] train_len={train_end:6d} period={period:4d}  "
          f"clean={len(ci)} K_pulse={K} K_artifact={n_art}  K={K_total}  "
          f"test_seg={len(test_spans)}  {info}")
    def _anom(s, e):                                 # whether the span overlaps a Label anomaly
        return int(label[s:e + 1].sum() > 0)
    if save:
        plot(series_id, x_disp, spans, labels_plot, K_total, anom_spans, shapes_plot,
             train_end, k_pulse=K, test_spans=test_spans, test_labels=test_labels)
    return {"series": series_id, "train_len": int(train_end), "period": int(period),
            "n_segments": len(spans), "n_clean": len(ci), "K": int(K_total), "info": info,
            "segments": [{"i": i, "start": int(s), "end": int(e), "cluster": int(l),
                          "clean": bool(clean_train[i]), "region": "train", "anom": _anom(s, e)}
                         for i, ((s, e), l) in enumerate(zip(spans, labels_plot))]
                        + [{"i": len(spans) + j, "start": int(s), "end": int(e),
                            "cluster": int(l), "clean": bool(clean_test[j]), "region": "test",
                            "anom": _anom(s, e)}
                           for j, ((s, e), l) in enumerate(zip(test_spans, test_labels))]}


def plot(series_id, x, spans, labels, K, anom_spans, shapes, train_end, k_pulse=None,
         test_spans=None, test_labels=None):
    """PDF output: wrap at ROW_STEPS steps per row, vertical dashed lines at split points, and
    background-highlight each span with its cluster color (full coverage). The last page overlays cluster shapes + mean.
    Passing test_spans/test_labels also draws the region from train_end on (test region), colored by the assigned
    train cluster medoid. train/test boundary = solid blue; Label anomaly regions (anom_spans, may be multiple) = red band + red lines."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    n = len(x)
    anom_spans = list(anom_spans or [])
    anom_max = max((e for _, e in anom_spans), default=0)
    all_spans = list(spans) + list(test_spans or [])
    all_labels = list(labels) + list(test_labels or [])
    out = os.path.join(OUT_DIR, f"{series_id}.pdf")
    with PdfPages(out) as pdf:
        nrows = int(np.ceil(n / ROW_STEPS))
        # for long series, cap the max page count to avoid PDF bloat. But always show up to the last Label anomaly region.
        max_pages = 8
        if anom_max:
            max_pages = max(max_pages, int(np.ceil((anom_max + ROW_STEPS) / ROW_STEPS / ROWS_PER_PAGE)) + 1)
        max_pages = min(max_pages, 20)
        rows_cap = max_pages * ROWS_PER_PAGE
        # unify the vertical-axis scale across all rows (no per-row auto-scaling, feedback 2026-08-09).
        #   fix ylim by the robust range (1-99 percentile) of all displayed data + margin.
        disp = x[:min(n, rows_cap * ROW_STEPS)]
        dlo, dhi = float(np.percentile(disp, 1)), float(np.percentile(disp, 99))
        if dhi - dlo < 1e-9:
            dlo, dhi = float(disp.min()), float(disp.max())
        ypad = 0.08 * (dhi - dlo + 1e-9)
        ylo, yhi = dlo - ypad, dhi + ypad
        for pg in range(0, min(nrows, rows_cap), ROWS_PER_PAGE):
            rows = list(range(pg, min(pg + ROWS_PER_PAGE, nrows)))
            fig, axs = plt.subplots(len(rows), 1, figsize=(16, 1.5 * len(rows)),
                                    squeeze=False)
            for ax, r in zip(axs[:, 0], rows):
                w0, w1 = r * ROW_STEPS, min((r + 1) * ROW_STEPS, n)
                # light-red band for Label anomaly regions (bottom background layer, multiple regions)
                for (as_, ae_) in anom_spans:
                    if ae_ <= w0 or as_ >= w1:
                        continue
                    ax.axvspan(max(as_, w0), min(ae_, w1), color="red", alpha=0.10, lw=0, zorder=0)
                    if w0 <= as_ < w1:
                        ax.axvline(as_, color="red", lw=0.7, ls="-", alpha=0.35, zorder=1.5)
                    if w0 <= ae_ < w1:
                        ax.axvline(ae_, color="red", lw=0.7, ls="-", alpha=0.35, zorder=1.5)
                ax.plot(range(w0, w1), x[w0:w1], color="black", lw=0.5, zorder=3)
                for (s, e), lab in zip(all_spans, all_labels):
                    if e <= w0 or s >= w1:
                        continue
                    col = "lightgray" if lab < 0 else cmap(lab % 10)
                    ax.axvspan(max(s, w0), min(e, w1), color=col, alpha=0.30, lw=0, zorder=1)
                    if w0 <= s < w1:
                        ax.axvline(s, color="dimgray", lw=0.5, ls="--", zorder=2)
                if w0 <= train_end < w1:                   # train/test boundary
                    ax.axvline(train_end, color="blue", lw=1.6, ls="-", zorder=5)
                ax.set_xlim(w0, w0 + ROW_STEPS)
                ax.set_ylim(ylo, yhi)                  # unify vertical-axis scale across all rows
                ax.set_ylabel(f"{w0}", fontsize=7, rotation=0, ha="right", va="center")
                ax.tick_params(labelsize=6)
            kp = "" if k_pulse is None else f" (pulse 0..{k_pulse-1}, artifact {k_pulse}..)"
            extra = "" if nrows <= rows_cap else f"  [showing first {rows_cap} rows of {nrows}]"
            axs[0, 0].set_title(
                f"{series_id}  train(len={train_end})+test applied  |  "
                f"train_seg={len(spans)} test_seg={len(test_spans or [])}  K={K}{kp}   "
                f"[blue=train/test boundary, red=injected anomaly, bg=cluster color]{extra}",
                fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- last page: per-cluster overlay + mean ----
        ncol = min(7, max(K, 1))
        nrow = int(np.ceil(max(K, 1) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow), squeeze=False)
        for c in range(K):
            ax = axs[c // ncol, c % ncol]
            members = shapes[labels == c]
            for m in members:
                ax.plot(m, color=cmap(c % 10), alpha=0.25, lw=0.6)
            if len(members):
                ax.plot(members.mean(0), color="black", lw=1.2)
            kind = "" if k_pulse is None else ("pulse" if c < k_pulse else "artifact")
            ax.set_title(f"c{c} {kind} (n={int((labels == c).sum())})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        for c in range(K, nrow * ncol):
            axs[c // ncol, c % ncol].axis("off")
        fig.suptitle(f"{series_id}  cluster shapes (z-norm, resampled)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)


# ----------------------------------------------------------------------------- run API
# keys that may be overridden by the <id>.py PARAMS (does not change the clustering structure or metrics)
TUNABLE = ("PERIOD", "RESAMPLE_LEN", "PEAK_DIST_FRAC", "PEAK_PROM", "LOCALNORM_FRAC",
           "SMOOTH_FRAC", "LEN_LO_FRAC", "LEN_HI_FRAC", "FLATTOP_MAX", "NOISE_PEAKS_MAX",
           "CLIP_CEIL_FRAC", "CLIP_RUN_MIN",
           "KMAX_S1", "KMAX_S2", "MAXLAG_FRAC", "MERGE_TAU", "SIL_MIN", "MIN_PATTERN",
           "MAHAL_RIDGE", "STAGE2_SBD", "STAGE2_AMP_RANGE", "STAGE2_AMP_ACT",
           "STAGE2_AMP_CURV", "ARTIFACT_TAU", "ANOM_EXCL_FRAC",
           "MIN_PERIOD", "MIN_SEGS_ND", "MAX_SEGS", "FIXED_WIN")


def run(series_id, params=None, save=True):
    """Public API to process a single series. Called from `<id>.py`.
    `params` overrides the core defaults (TUNABLE) (per-series tuning)."""
    g = globals()
    saved = {k: g[k] for k in TUNABLE}
    try:
        for k, v in (params or {}).items():
            if k not in TUNABLE:
                raise KeyError(f"unknown param '{k}'. valid: {TUNABLE}")
            g[k] = v
        r = process(series_id, save=save)
        if r:
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f"{series_id}_segments.csv"), "w") as f:
                f.write("seg_index,start,end,cluster,clean,region,anom\n")
                for s in r["segments"]:
                    f.write(f"{s['i']},{s['start']},{s['end']},{s['cluster']},"
                            f"{int(s['clean'])},{s.get('region', 'train')},{s.get('anom', 0)}\n")
        return r
    finally:
        for k, v in saved.items():
            g[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", help="series id(s) e.g. 225 226")
    ap.add_argument("--all", action="store_true", help=f"all {SUBSET} series")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ids = domain_series_ids() if args.all else (args.series or domain_series_ids()[:3])
    results = []
    for sid in ids:
        r = process(sid, save=not args.no_save)
        if r:
            results.append(r)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write("series,train_len,period,n_segments,n_clean,K\n")
        for r in results:
            f.write(f"{r['series']},{r['train_len']},{r['period']},"
                    f"{r['n_segments']},{r['n_clean']},{r['K']}\n")
    with open(os.path.join(OUT_DIR, "segments.csv"), "w") as f:
        f.write("series,seg_index,start,end,cluster,clean,region,anom\n")
        for r in results:
            for s in r["segments"]:
                f.write(f"{r['series']},{s['i']},{s['start']},{s['end']},"
                        f"{s['cluster']},{int(s['clean'])},{s.get('region', 'train')},{s.get('anom', 0)}\n")
    print(f"\n-> {len(results)} series. output in {OUT_DIR}")


if __name__ == "__main__":
    main()
