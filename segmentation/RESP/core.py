#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESP base-waveform segmentation + clustering engine (core) -- implemented from scratch
=====================================================================================

A shared engine that splits respiration waveforms (Respiration Rate) into "base-waveform
patterns = one breath" and clusters them by shape. Per-series runs are handled by thin
`<id>.py` files (which just call core.run).

Characteristics of RESP signals (differences from ABP)
------------------------------------------------------
- One unit = one breath. A smooth quasi-periodic signal, an **asymmetric single peak**
  (a sharp inspiratory peak + a gentle expiratory valley). The period is ~470 steps in this data.
- There is no morphology assumption like ABP's "systolic peak position is toward the front" or
  "dicrotic notch". The peak position can vary by series, so no phase-position filter is applied.
- Some series (the NOISE family) carry superimposed high-frequency noise. The clean decision is
  made on the smoothed shape, so normal breaths are not mistakenly removed even in noisy series.

Segmentation policy
-------------------
- Learn the base waveform only from the training region (UCR: the leading train_end is normal).
- Treat foot (valley) to next foot as one breath, include the signal ends as boundaries, and cover
  the whole range with no gaps. To avoid over-segmentation, impose a lower bound on peak spacing (~0.55*period).
- To be robust to amplitude drift, detect peaks on a locally amplitude-normalized signal so that
  low-amplitude breaths are not missed (prevents under-segmentation).
- clean filter: exclude units whose length is off-period / saturation trapezoid / multi-peak noise fragment.

Clustering (* structure, operations, and metrics are fixed; only parameters vary per series)
--------------------------------------------------------------------------------------------
- Stage1 = shape (ignore amplitude): robust z-norm + common-length resample, all-shift SBD +
  complete-linkage, K_s1 chosen by relative silhouette comparison over {1..KMAX_S1}.
- Stage2 = amplitude/scale + fine shape statistics: globally standardize D3 (5 features, RAW) over all units
  -> Mahalanobis whiten -> Ward. Subdivide each Stage1 cluster by K_sub {1..KMAX_S2}.
  Acceptance guard: keep only when there is a shape difference (medoid full-SBD>STAGE2_SBD) or a real
  amplitude sub-mode (range relative diff>STAGE2_AMP_RANGE and activity/curv relative diff>threshold).
- Artifact clusters: group non-clean units by robust z-norm shape SBD complete-linkage
  (SBD<ARTIFACT_TAU = same shape), numbered after the pulse clusters.

Usage
-----
  python 076.py            # process series 076 -> output/076.pdf, etc.
  python core.py --all     # batch-process all RESP series
  python core.py --series 076 186
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths / domain
# The domain this folder handles (a key of domains_dict.json). The shared json lives in the parent.
DOMAIN = "Respiration Rate (RESP)"
DATA_DIR = "./data/UCR_Anomaly_FullData"
_HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_JSON = os.path.join(os.path.dirname(_HERE), "domains_dict.json")
OUT_DIR = os.path.join(_HERE, "output")

# ----------------------------------------------------------------------------- defaults (can be overridden per series via PARAMS)
ROW_STEPS = 2000          # time steps per PDF row
ROWS_PER_PAGE = 12        # rows per PDF page
RESAMPLE_LEN = 256        # resample each breath to this length for shape comparison

# --- segmentation (for RESP) ---
PERIOD = None             # manual period (None = auto-estimate by autocorrelation)
PEAK_DIST_FRAC = 0.55     # min peak spacing = this fraction * period (suppresses over-segmentation)
PEAK_PROM = 0.25          # lower bound on peak prominence on the locally normalized (0..1) signal
LOCALNORM_FRAC = 1.5      # local amplitude-normalization window = this fraction * period
SMOOTH_FRAC = 0.06        # savgol window before peak detection = this fraction * period
LEN_LO_FRAC = 0.55        # clean length lower bound = this fraction * period
LEN_HI_FRAC = 1.7         # clean length upper bound = this fraction * period
FLATTOP_MAX = 0.22        # flat fraction near the top at or above this = flat trapezoid (excluded)
NOISE_PEAKS_MAX = 4       # peak count in the smoothed shape above this = multi-peak noise fragment (excluded)
# Detection of saturated (measurement-range ceiling clip) beats: RESP contains clip beats that peg to the
# ADC ceiling (~+/-2000). These are "saturation trapezoids" = non-clean (a spec clean-exclusion condition).
# Because the flat fraction after z-norm cannot catch a narrow clip, detect directly on the raw waveform by
# "proximity to the global ceiling x the length of the continuous same-value run at the top", and consistently
# send them to artifact across all series (cross-evaluation-AI feedback 2026-06-24). Normal breaths do not
# touch the ceiling (run 1-3); saturated beats have run 10-50.
CLIP_CEIL_FRAC = 0.02     # near the ceiling if the unit's peak is at or above global (max - this*amplitude)
CLIP_RUN_MIN = 8          # near the ceiling and a near-peak continuous run at or above this = saturation clip (excluded)

# --- clustering K selection / merging (structure fixed; only thresholds tuned) ---
KMAX_S1 = 3               # upper bound on Stage1 K_s1 candidates
KMAX_S2 = 3               # upper bound on Stage2 K_sub candidates
MAXLAG_FRAC = 0.10        # lag upper bound for the shift-limited SBD used to assign non-clean units to nearest
MERGE_TAU = 0.10          # merge as same shape if the SBD between medoids is below this
SIL_MIN = 0.12            # do not split if the best silhouette is below this (K=1)
MIN_PATTERN = 4           # below this size it is a one-off outlier, not a recurring pattern -> absorbed
MAHAL_RIDGE = 1e-3        # Stage2 covariance ridge (stabilizes Mahalanobis whitening)
STAGE2_SBD = 0.08         # if full-SBD between sub-cluster medoids exceeds this, "shapes differ"
STAGE2_AMP_RANGE = 0.12   # if amplitude range relative diff exceeds this, and below, then "amplitude sub-mode"
                          # RESP has large natural amplitude fluctuation per breath, so a ~8% range
                          # splits easily. Set to 0.12 to merge false splits driven only by a tiny range difference.
STAGE2_AMP_ACT = 0.15     #   activity relative diff exceeds this
STAGE2_AMP_CURV = 0.25    #   or curv relative diff exceeds this
ARTIFACT_TAU = 0.10       # non-clean units: same-shape artifact if SBD is below this
# clean/artifact is a per-beat decision (2026-08-13, user-specified): rather than an absolute criterion
#   (is_cycle_shape/K_pulse), it is decided solely by "whether each beat resembles its cluster's medoid
#   (representative beat)". An incomplete beat mixed into a consistent cluster deviates from the medoid -> artifact.
#   Scattered clusters / singletons are all-beat artifact. SBD is length-normalized + all-shift max-correlation.
CLEAN_BY_MEDOID = True     # enable per-beat clean decision
CLEAN_SBD = 0.4            # a beat is clean if its SBD to the cluster medoid is at most this (~correlation >= 0.6)
CLEAN_MIN_N = 2            # if the cluster size is below this (a non-repeating singleton), all beats are artifact
# --- singleton reconsideration rule (decide whether to merge/re-split a size-1 cluster) ---
SINGLE_FACTOR = 1.5       # tolerance multiplier relative to the receiving cluster's internal spread (medoid SBD p90) (relative comparison)
SINGLE_FLOOR = 0.12       # floor on the merge-tolerance SBD (treat as same shape within this range even if the receiver is small)


# ============================================================================= IO
def find_file(series_id):
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{series_id}_*.txt")))
    if not hits:
        raise FileNotFoundError(f"series {series_id} not found in {DATA_DIR}")
    return hits[0]


def parse_meta(path):
    """Get the UCR filename suffix _<train_end>_<anom_start>_<anom_end>."""
    base = os.path.basename(path).replace(".txt", "")
    nums = base.split("_")
    try:
        return int(nums[-3]), int(nums[-2]), int(nums[-1])
    except ValueError:
        return 0, 0, 0


def domain_series_ids():
    """List of series IDs belonging to this folder's DOMAIN (ascending)."""
    d = json.load(open(DOMAINS_JSON))
    return sorted(n.split("_")[0] for n in d[DOMAIN])


abp_series_ids = domain_series_ids   # backward-compat alias for cluster_diag etc.


# ============================================================================= segmentation
def dominant_period(x):
    """Estimate the dominant period (samples per breath) by autocorrelation.
    Among autocorrelation peaks after the zero crossing, take the lag with the highest correlation
    as the base period (so a half-period minor peak is not mistaken for the base period)."""
    z = x - x.mean()
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
    """Normalize to 0..1 with local (min,max) to remove amplitude drift.
    So low-amplitude breaths can be peak-detected with the same prominence as high-amplitude ones."""
    win = max(5, int(win) | 1)
    lo = minimum_filter1d(xs, win, mode="nearest")
    hi = maximum_filter1d(xs, win, mode="nearest")
    return (xs - lo) / (hi - lo + 1e-9)


def segment_cycles(x, period):
    """Split the whole signal into breaths, foot (valley) to next foot, with no gaps.

    Returns:
      spans  : [(s,e), ...] breaths covering the whole range (including partial breaths at the ends)
      clean  : whether each span "looks like a breath (usable for cluster definition)" (bool list)
      peaks  : the detected inspiratory peaks
    """
    n = len(x)
    w = max(5, int(period * SMOOTH_FRAC) | 1)
    xs = savgol_filter(x, window_length=w, polyorder=2)
    norm = _local_normalize(xs, win=int(period * LOCALNORM_FRAC))
    peaks, _ = find_peaks(norm, distance=max(3, int(period * PEAK_DIST_FRAC)),
                          prominence=PEAK_PROM)
    if len(peaks) < 3:
        return [], [], peaks
    # Use the minimum point (valley) between consecutive peaks as the cut foot
    feet = [peaks[i] + int(np.argmin(xs[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)]
    # add the signal ends as boundaries too, covering [0, n] with no gaps
    bounds = sorted(set([0] + feet + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    lo_len, hi_len = int(period * LEN_LO_FRAC), int(period * LEN_HI_FRAC)
    ceiling = x.max() - CLIP_CEIL_FRAC * (x.max() - x.min())   # ceiling-proximity threshold for saturation detection
    clean = [bool(lo_len <= e - s <= hi_len) and is_breath_shape(x[s:e])
             and not is_clipped(x[s:e + 1], ceiling)
             for s, e in spans]
    return spans, clean, peaks


def is_clipped(seg, ceiling):
    """Whether this is a saturated (measurement-range ceiling clip) beat. If the unit's peak is near the global ceiling and
    there is a long continuous same-value run (near-peak run) at the top, treat it as saturated (non-clean).
    Normal breaths do not reach the ceiling and have a rounded top, so the run is short."""
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
    """Whether the shape looks like one breath (rejects saturation trapezoids and multi-peak noise fragments).
    Features are computed on the smoothed shape, so normal breaths in noisy series are not mistakenly removed.
    RESP does not care about peak position (no ABP-style phase-position filter)."""
    if len(seg) < 5:
        return False
    m = znorm(resample(seg))
    ms = savgol_filter(m, 11, 2)
    rng = ms.max() - ms.min() + 1e-9
    flattop = float((ms > ms.max() - 0.1 * rng).mean())   # flat fraction near the top
    npk = len(find_peaks(ms, prominence=0.2)[0])          # number of true shape peaks
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


def sbd_full_pair(a, b):
    """All-shift SBD = 1 - max(normalized cross-correlation). Phase-tolerant (compares by waveform type)."""
    cc = np.correlate(a, b, mode="full")
    return float(1.0 - cc.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def sbd_full_matrix(mat):
    """All-shift SBD distance matrix (Stage1: broad grouping by waveform type)."""
    n = len(mat)
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cc = np.correlate(mat[i], mat[j], mode="full")
            D[i, j] = D[j, i] = float(1.0 - cc.max() / (norms[i] * norms[j]))
    return D


def d3_features(w):
    """Stage2 D3 (5 features, RAW values, interval = the unit itself, no smoothing).
    [0]range=max-min (raw amplitude), [1]activity=mean(|diff|), [2]curv=max(|diff2|),
    [3]n_extrema=#find_peaks(w,prom=0.10*range)+#find_peaks(-w, same),
    [4]skew=mean(z**3) (z=(w-mean)/std, ddof=0). Length<3 returns a zero vector."""
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
    """Merge cluster pairs whose inter-medoid SBD < MERGE_TAU."""
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
    """Absorb clusters with size < MIN_PATTERN into the nearest-medoid cluster (SBD space)."""
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
    """Absorb clusters with size < MIN_PATTERN into the nearest centroid (feature space)."""
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
    """Stage2 acceptance guard: merge sub-cluster pairs that are "effectively the same in both shape and amplitude sub-mode".
    They are kept only if (a) shape difference: medoid full-SBD > STAGE2_SBD, or
    (b) a real amplitude sub-mode: range relative diff > STAGE2_AMP_RANGE and
       (activity relative diff > STAGE2_AMP_ACT or curv relative diff > STAGE2_AMP_CURV).
    -> merges false splits caused by crossing the n_extrema threshold or cutting a continuous gradient where only range moves."""
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
    # ---- Stage 1: shape (ignore amplitude) ----
    D1 = sbd_full_matrix(shapes_robust)
    lab1, info1 = select_k_precomputed(D1, KMAX_S1, "complete")

    # ---- Stage 2: amplitude/scale + fine shape statistics ----
    F = np.array([d3_features(w) for w in beats_raw])
    Fs = (F - F.mean(0)) / (F.std(0) + 1e-9)                  # global standardization
    cov = np.cov(Fs, rowvar=False) + MAHAL_RIDGE * np.eye(Fs.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    Wmat = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T        # whitening matrix
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
    """Group non-clean units by robust z-norm shape SBD complete-linkage
    (SBD < ARTIFACT_TAU = same shape). A distinctive one is its own cluster even at n=1."""
    n = len(shapes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    D = sbd_full_matrix(shapes)
    Z = linkage(D[np.triu_indices(n, 1)], method="complete")
    return relabel(fcluster(Z, t=ARTIFACT_TAU, criterion="distance") - 1)



# ============================================================================= singleton reconsideration rule
def _medoid_spread(shapes_rz, idx):
    """The medoid shape of cluster idx and its internal spread (90th percentile of SBD to the medoid).
    Large clusters are evenly subsampled to <=80 items to limit the O(n^2) SBD (speedup)."""
    idx = np.asarray(idx)
    if len(idx) == 1:
        return shapes_rz[idx[0]], 0.0
    sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
    D = sbd_full_matrix(shapes_rz[sub])
    mi = int(D.sum(1).argmin())
    return shapes_rz[sub[mi]], float(np.percentile(D[mi], 90))


def _nearest_cluster(shape, labels, shapes_rz, exclude):
    """Return the cluster nearest to shape (other than exclude): (SBD, cluster, acceptance threshold).
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
    """Re-split interval [s,e] (a long under-segmented unit) at internal valleys and return the sub-spans."""
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
    # Keep full coverage: do not drop too-short fragments; merge them into the next fragment to tile [s,e] with no gaps.
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
    """singleton / small-cluster reconsideration rule (added 2026-06-25, generalizing a user request):
    (a) **same-shape reclaim**: if an artifact cluster's (label >= k_orig) medoid is within the nearest pulse cluster's
        internal spread (medoid SBD p90) x SINGLE_FACTOR, merge that whole cluster into the pulse
        (reclaim normal cycles judged non-clean due to noise/distortion into c0 etc. Relative comparison).
    (b) **re-split**: a size-1 artifact of length ~2 periods or more (an under-segmented multi-cycle) is re-split at its internal
        valleys, and each piece is merged into the nearest cluster (if there is no target, it stays an independent artifact).
    (c) **keep independent**: genuinely different shapes (partial/double/distinctive) are left as-is.
    Returns: updated spans, labels. (Speedup: cluster medoids are computed only once per iteration.)"""
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
        # (a) same-shape reclaim of a whole artifact cluster into a pulse
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
    Pulse clusters = original pulse labels (< k_orig) that remained. New labels created by splitting are artifacts."""
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
    # Sort intervals by start position (readability: PDF/CSV in chronological order)
    sidx = sorted(range(len(spans)), key=lambda i: spans[i][0])
    spans = [spans[i] for i in sidx]
    labels = [labels[i] for i in sidx]
    return spans, labels, len(pulse_ids), len(art_ids), None
# ============================================================================= driver
def process(series_id, save=True):
    path = find_file(series_id)
    x = np.loadtxt(path).astype(float)
    train_end, anom_s, anom_e = parse_meta(path)
    xt = x[:train_end] if train_end > 200 else x   # learn the base waveform on the normal training region

    period = PERIOD if PERIOD else dominant_period(xt)
    spans, clean, peaks = segment_cycles(xt, period)
    clean_idx = [i for i, c in enumerate(clean) if c]
    if len(clean_idx) < 3:
        print(f"[{series_id}] segmentation failed (period={period}, clean={len(clean_idx)})")
        return None

    shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    beats_raw = [xt[s:e + 1] for s, e in spans]

    # two-stage clustering on clean units only (not misled by artifacts)
    ci = clean_idx
    labels_c, K, info, _ = cluster_two_stage([beats_raw[i] for i in ci], shapes_rz[ci])

    labels_plot = np.full(len(spans), -1, dtype=int)
    for k, i in enumerate(ci):
        labels_plot[i] = int(labels_c[k])

    # non-clean units = artifact clusters (numbered after the pulse clusters)
    nc = [i for i in range(len(spans)) if labels_plot[i] < 0]
    n_art = 0
    if nc:
        art = cluster_artifacts(shapes_rz[nc])
        for j, i in enumerate(nc):
            labels_plot[i] = K + int(art[j])
        n_art = int(art.max()) + 1

    # singleton reconsideration rule: merge/re-split size-1 / small clusters (relative comparison). Preserve pulse-derived cycles.
    clean_set = set(clean_idx)
    spans, labels_plot, K, n_art, clean_set = _apply_reconcile(
        xt, spans, list(labels_plot), period, K, clean_set)
    labels_plot = np.array(labels_plot)
    K_total = K + n_art
    info["K_pulse"], info["K_artifact"] = K, n_art

    # std-based z-norm for the overlay plot (MAD diverges on saturated units and is hard to read)
    shapes_plot = np.array([znorm(resample(xt[s:e + 1])) for s, e in spans])

    # ---- apply the learned segmentation to the test region (from train_end on) and assign each unit to the nearest train cluster medoid ----
    #      (the base waveform is learned only on train. The test region is not re-clustered but classified by nearest medoid = same structure as VLM-AD).
    #      Assign all test segments to the nearest cluster (clean or non-clean, never -1).
    #        Candidates are all learned clusters (pulse 0..K-1 + artifact K..K_total-1); take the nearest medoid.
    final_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    medoids = {}
    clu_size = {}
    for c in range(K_total):
        idx = np.where(labels_plot == c)[0]
        clu_size[c] = len(idx)
        if len(idx):
            sub = final_rz[idx]
            medoids[c] = sub[sbd_full_matrix(sub).sum(1).argmin()] if len(idx) > 1 else sub[0]

    def _clean_by_medoid(shape, lab):
        """per-beat clean decision: clean if SBD to the cluster medoid <= CLEAN_SBD and cluster size >= CLEAN_MIN_N."""
        if not CLEAN_BY_MEDOID:
            return lab < K                                   # legacy: pulse cluster (< K_pulse) = clean
        return clu_size.get(lab, 0) >= CLEAN_MIN_N and \
            float(sbd_full_pair(medoids[lab], shape)) <= CLEAN_SBD
    clean_train = [bool(_clean_by_medoid(final_rz[i], int(labels_plot[i]))) for i in range(len(spans))]
    test_spans, test_labels, clean_test = [], [], []
    x_disp = xt
    if 200 < train_end < len(x) and medoids:
        xtt = x[train_end:].astype(float)
        ts, _tc, _ = segment_cycles(xtt, period)
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
    if save:
        plot(series_id, x_disp, spans, labels_plot, K_total, anom_s, anom_e, shapes_plot,
             train_end, k_pulse=K, test_spans=test_spans, test_labels=test_labels)
    return {"series": series_id, "train_len": int(train_end), "period": int(period),
            "n_segments": len(spans), "n_clean": len(ci), "K": int(K_total), "info": info,
            "segments": [{"i": i, "start": int(s), "end": int(e), "cluster": int(l),
                          "clean": bool(clean_train[i]), "region": "train"}
                         for i, ((s, e), l) in enumerate(zip(spans, labels_plot))]
                        + [{"i": len(spans) + j, "start": int(s), "end": int(e),
                            "cluster": int(l), "clean": bool(clean_test[j]), "region": "test"}
                           for j, ((s, e), l) in enumerate(zip(test_spans, test_labels))]}


def plot(series_id, x, spans, labels, K, anom_s, anom_e, shapes, train_end, k_pulse=None,
         test_spans=None, test_labels=None):
    """PDF output: wrap every ROW_STEPS steps per row, dashed vertical lines at split points, and each interval
    highlighted in its cluster color (full coverage). The last page overlays cluster shapes + mean.
    If test_spans/test_labels are given, also draw from train_end on (the test region), colored by assignment to the
    train pulse cluster medoids. train/test boundary = solid blue line; injected anomaly region (anom_s..anom_e) = solid red line + light red band."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    n = len(x)
    all_spans = list(spans) + list(test_spans or [])
    all_labels = list(labels) + list(test_labels or [])
    out = os.path.join(OUT_DIR, f"{series_id}.pdf")
    with PdfPages(out) as pdf:
        nrows = int(np.ceil(n / ROW_STEPS))
        # For long series, cap the max page count to avoid PDF bloat. But always show up to the injected anomaly region anom_e.
        max_pages = 8
        if anom_e:
            max_pages = max(max_pages, int(np.ceil((anom_e + ROW_STEPS) / ROW_STEPS / ROWS_PER_PAGE)) + 1)
        max_pages = min(max_pages, 20)
        rows_cap = max_pages * ROWS_PER_PAGE
        for pg in range(0, min(nrows, rows_cap), ROWS_PER_PAGE):
            rows = list(range(pg, min(pg + ROWS_PER_PAGE, nrows)))
            fig, axs = plt.subplots(len(rows), 1, figsize=(16, 1.5 * len(rows)),
                                    squeeze=False)
            for ax, r in zip(axs[:, 0], rows):
                w0, w1 = r * ROW_STEPS, min((r + 1) * ROW_STEPS, n)
                # light red band for the injected anomaly region (bottom background layer)
                if anom_e and anom_s and not (anom_e <= w0 or anom_s >= w1):
                    ax.axvspan(max(anom_s, w0), min(anom_e, w1), color="red", alpha=0.10, lw=0, zorder=0)
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
                for ab in (anom_s, anom_e):                # anomaly region boundaries
                    if ab and w0 <= ab < w1:
                        ax.axvline(ab, color="red", lw=1.4, ls="-", zorder=5)
                ax.set_xlim(w0, w0 + ROW_STEPS)
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
# Keys that <id>.py PARAMS may override (do not change the clustering structure or metrics)
TUNABLE = ("PERIOD", "RESAMPLE_LEN", "PEAK_DIST_FRAC", "PEAK_PROM", "LOCALNORM_FRAC",
           "SMOOTH_FRAC", "LEN_LO_FRAC", "LEN_HI_FRAC", "FLATTOP_MAX", "NOISE_PEAKS_MAX",
           "CLIP_CEIL_FRAC", "CLIP_RUN_MIN",
           "KMAX_S1", "KMAX_S2", "MAXLAG_FRAC", "MERGE_TAU", "SIL_MIN", "MIN_PATTERN",
           "MAHAL_RIDGE", "STAGE2_SBD", "STAGE2_AMP_RANGE", "STAGE2_AMP_ACT",
           "STAGE2_AMP_CURV", "ARTIFACT_TAU")


def run(series_id, params=None, save=True):
    """Public API that processes a single series. Called from `<id>.py`.
    params overrides the core defaults (TUNABLE) (series-specific tuning)."""
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
                f.write("seg_index,start,end,cluster,clean,region\n")
                for s in r["segments"]:
                    f.write(f"{s['i']},{s['start']},{s['end']},{s['cluster']},"
                            f"{int(s['clean'])},{s.get('region', 'train')}\n")
        return r
    finally:
        for k, v in saved.items():
            g[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", help="series id(s) e.g. 076 186")
    ap.add_argument("--all", action="store_true", help="all RESP series")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ids = domain_series_ids() if args.all else (args.series or ["076", "184", "186", "187", "077"])
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
        f.write("series,seg_index,start,end,cluster,clean,region\n")
        for r in results:
            for s in r["segments"]:
                f.write(f"{r['series']},{s['i']},{s['start']},{s['end']},"
                        f"{s['cluster']},{int(s['clean'])},{s.get('region', 'train')}\n")
    print(f"\n-> {len(results)} series. output in {OUT_DIR}")


if __name__ == "__main__":
    main()
