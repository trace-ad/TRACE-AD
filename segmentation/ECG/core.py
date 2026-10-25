#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG (Electrocardiogram Arrhythmia) base-waveform segmentation + clustering engine (core) -- built from scratch
==============================================================================================

Shared engine that segments ECG periodic waveforms into "base-waveform patterns = 1 beat
(1 P-QRS-T complex)" and clusters them by shape. Per-series execution is handled by a thin
`<id>.py` (which just calls core.run).

Characteristics of ECG signals (differences from other domains)
--------------------------------------
- Many recording sources (91 series): ECG1-4 / sddb / apneaecg / mit14xxx longtermecg / CHARIS / STAFFIII /
  BIDMC / qtdb / ltstdb / sel840m / Fantasia etc. **Beat period, amplitude, polarity, and noise vary greatly by source.**
- One beat is a multi-feature waveform centered on a sharp QRS (R-wave spike) with P-wave and T-wave. The R-wave is the sharpest, most consistent landmark.
- The period is auto-estimated as the median R-R interval (PERIOD=None). Autocorrelation is
  unusable because it picks up the slow envelope of respiration / apnea / baseline drift and
  misses the true beat (166 apneaecg->996, 194 sddb49->3954). See dominant_period.
- Some series contain flat / very-low-amplitude spans from baseline drift (wander), EMG noise, or
  electrode detachment. These are separated as non-clean = artifacts (full-span coverage is kept).
- Because arrhythmia can make RR vary beat to beat, the clean length band is wide (0.45-1.9*period).

Segmentation policy
--------
- Learn base waveforms only on the training span (UCR: the leading train_end is normal).
- Place beat boundaries at the low point between R-waves (TP baseline), include the signal ends as
  boundaries, and cover the whole span with no gaps.
  Cut points are placed at low points, not at peaks (R-wave tops). The R-wave is used as the phase-alignment "reference" (per PROMPT.md).
  To avoid over-segmentation, impose a lower bound on peak-to-peak distance (~0.55*period).
- To be robust to baseline drift / noise, detect peaks on a locally amplitude-normalized smoothed signal
  (SMOOTH_FRAC is kept small at 0.03 so the QRS is not crushed).
- clean filter: exclude fragments whose length falls outside the period band, fragments lacking shape coherence, and very-low-amplitude spans.

Clustering (structure, operations, and metrics are fixed; only parameters are per-series)
----------------------------------------------------------------
- Stage1 = shape (amplitude ignored): robust z-norm + common-length resample, full-shift SBD +
  complete-linkage, K_s1 chosen by relative silhouette comparison over {1..KMAX_S1}.
- Stage2 = amplitude/scale + fine shape statistics: globally standardize D3 (5 features, RAW) over all units
  -> Mahalanobis whiten -> Ward. Subdivide each Stage1 cluster with K_sub {1..KMAX_S2}.
  Acceptance guard: adopt only when there is a shape difference (medoid full-SBD>STAGE2_SBD) or a real
  amplitude submode (range relative diff>STAGE2_AMP_RANGE and activity/curv relative diff>threshold).
- Artifact clusters: group non-clean spans by robust z-norm shape SBD complete
  (SBD<ARTIFACT_TAU = same shape), numbered after the pulse clusters.

Usage
------
  python 119.py            # process series 119 -> output/119.pdf etc.
  python core.py --all     # batch-process all ECG series
  python core.py --series 119 125
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
from scipy.ndimage import minimum_filter1d, maximum_filter1d, median_filter
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths / domain
# The domain this folder handles (key in domains_dict.json). The shared json lives in the parent.
DOMAIN = "Electrocardiogram (ECG) Arrhythmia"
DATA_DIR = "./data/UCR_Anomaly_FullData"
_HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_JSON = os.path.join(os.path.dirname(_HERE), "domains_dict.json")
OUT_DIR = os.path.join(_HERE, "output")

# ----------------------------------------------------------------------------- defaults (can be overridden per-series via PARAMS)
ROW_STEPS = 2000          # time steps per PDF row
ROWS_PER_PAGE = 12        # rows per PDF page
RESAMPLE_LEN = 128        # resample each cycle to this length for shape comparison

# --- segmentation (for ECG) ---
PERIOD = None             # manual period (None = auto-estimate the median R-R interval; varies greatly by series)
PEAK_DIST_FRAC = 0.55     # min peak-to-peak distance = this ratio * period (suppresses over-segmentation)
PEAK_PROM = 0.20          # lower bound on peak prominence on the locally normalized (0..1) signal
LOCALNORM_FRAC = 1.5      # window for local amplitude normalization = this ratio * period (removes amplitude drift)
SMOOTH_FRAC = 0.03        # savgol window before peak detection = this ratio * period
                          # ECG-specific: the QRS is sharp and narrow, so smaller than Gait (0.08), i.e. 0.03 (so the R-wave is not crushed).
LEN_LO_FRAC = 0.45        # clean length lower bound = this ratio * period (allows beats shortened by arrhythmia)
LEN_HI_FRAC = 1.9         # clean length upper bound = this ratio * period (allows beats lengthened by arrhythmia)
SHORT_SPAN_FRAC = 0.45    # sub-period fragments shorter than this ratio*period are merged into the adjacent segment
RR_CUT_BEFORE_QRS = False # True: cut just before each QRS (PQ junction) (each segment = QRS then T-wave; the first is a lead-in partial)
TEST_SPLIT_MULTIBEAT = False  # True: re-split 2-QRS fused spans in the test region at the R-waves too (default False = keep long anomalous beats).
                              # Enable for series where normal beats fuse via a missed R (2 peaks in 1 beat) on smooth low-amplitude waves (e.g. 202).
                              # Only fusions with 2+ R are split; long anomalous beats with a single R are kept (preserves the anomaly signal).
QRS_SHARP_DETECT = False  # True: detect R-waves by a derivative-energy criterion (sharp QRS only; does not mis-detect large T-waves)
RPEAK_LOCALNORM = False   # True: detect R-waves with local (min,max) normalization (detects low-amplitude peaks on high-amplitude-variation smooth waves = 202)
NOISE_ROUGH_TAU = 0.0     # >0: merge artifacts whose roughness (2nd-diff RMS/range) exceeds this into one noise cluster (degraded region)
PRE_SMOOTH_WN = 0.0       # >0: apply a Butterworth low-pass (normalized cutoff) at start. Removes high-frequency ripple (209)
ISOLATE_SBD = 0.0         # >0: separate beats with outlying medoid SBD (ectopic/inverted) within the normal cluster into a separate cluster
QRS_HEIGHT = 0.15         # derivative-energy peak height threshold in detect_qrs_sharp (lower it to pick up weak QRS)
ABNORMAL_SPANS = []       # [(start,end),...] force beats in these time ranges into an anomaly cluster (expert-specified episode)
ABNORMAL_TAU = 0.45       # shape sub-clustering threshold within ABNORMAL_SPANS (SBD complete distance)
PVC_SHAPE_K = 0           # >0: re-split anomalous pulse beats into K by shape SBD (corrects shape inconsistency of the D3 amplitude split)
VALLEY_RECLAIM = 0        # >0: move cut points that are N*MAD below the baseline back to the onset of the downstroke (captures valleys of PVCs without a sharp QRS)
RR_CAP_FRAC = 0.0         # >0: cap the rr-anchor cut-window RR at period*this value (avoids mislocated cuts on pauses/dropped beats)
FORCE_SHAPE_K = 0         # >0: fix Stage1 K from shape-SBD complete instead of silhouette (forces a continuous type into the desired count)
DEEP_DIP_FRAC = 0.0       # >0: put beats with raw dip > median*this value into a separate pulse cluster (amplitude-feature type of deep T-wave undershoot)
PVC_MERGE_TAU = 0.0       # >0: after PVC_SHAPE, merge cluster pairs with medoid SBD<tau (fold same-shape PVCs into the main type)
PVC_TEMPLATE = ""         # series-id string. When set, assign PVCs to the nearest medoid of a reference series (cross-series consistent classification)
PVC_TEMPLATE_SPLIT_TAU = 0.0  # >0: sub-split template types whose 2-way-split medoid SBD>tau (subtypes absent from the reference series)
_QRS_SIGNAL = None  # process temporarily stores the raw signal here under PRE_SMOOTH (for R-wave detection). Not a TUNABLE
SHARP_QRS_MIN = 3.0       # ECG-specific: whether a beat has a sharp QRS = max deviation/MAD at least this (otherwise treat as a
                          # flat dropout / no-signal span = non-clean). Recognizes beats by QRS sharpness even for long periods (STAFFIII).
NOISE_PEAKS_MAX = 9       # if the number of prominent peaks (both polarities) in the smoothed shape exceeds this = multi-peak noise fragment (excluded)
                          # ECG: one beat has P,Q,R,S,T (up/down) = up to ~6-8 extrema, so lenient at 9.
FLATTOP_MAX = 0.30        # (unused by is_cycle_shape for ECG; kept for compatibility)
AMP_FLOOR_FRAC = 0.30     # if a span's raw amplitude (range) is below "median of all-span range * this", it is non-clean
                          # = artifact. Excludes ECG flat / very-low-amplitude dropout / no-signal spans.

# --- clustering K selection/merging (structure fixed; only thresholds tuned) ---
KMAX_S1 = 3               # Stage1 K_s1 candidate upper bound
KMAX_S2 = 3               # Stage2 K_sub candidate upper bound
MAXLAG_FRAC = 0.10        # lag upper bound of the shift-limited SBD used for nearest-neighbor assignment of non-clean units
MERGE_TAU = 0.10          # merge as the same shape if medoid-to-medoid SBD is below this
SIL_MIN = 0.12            # do not split (K=1) if the best silhouette is below this
MIN_PATTERN = 8           # smaller than this size = a single outlier rather than a recurring pattern -> absorb
                          # ECG: beats are usually numerous (tens to hundreds). Absorbs the few false splits from phase-cut
                          # variability while retaining real submodes of ectopic beats (PVC etc.), hence a moderate 8 (tuned per series in evaluation).
MAHAL_RIDGE = 1e-3        # Stage2 covariance ridge (stabilizes the Mahalanobis whiten)
STAGE2_SBD = 0.08         # if medoid-to-medoid full-SBD of sub-clusters exceeds this, "shapes differ"
STAGE2_AMP_RANGE = 0.12   # if amplitude range relative diff exceeds this, and the below holds, it is an "amplitude submode"
STAGE2_AMP_ACT = 0.15     #   activity relative diff exceeds this
STAGE2_AMP_CURV = 0.25    #   or curv relative diff exceeds this
ARTIFACT_TAU = 0.10       # non-clean span: same-shape artifact if SBD is below this
# clean/artifact is a per-beat decision (2026-08-13, user-specified): rather than an absolute criterion
#   (is_cycle_shape/K_pulse), it is decided solely by "whether each beat resembles the medoid (representative beat) of its cluster".
#   An incomplete beat mixed into a consistent cluster floats off the medoid -> artifact. A scattered cluster/singleton is all artifact.
#   SBD is length-normalized + full-shift max-correlation.
CLEAN_BY_MEDOID = True     # enable per-beat clean decision
CLEAN_SBD = 0.4            # clean if the beat is within this SBD of the cluster medoid (~correlation >= 0.6)
CLEAN_MIN_N = 2            # if cluster size is below this (a non-recurring singleton), all beats are artifact
# --- singleton reconsideration rule (decide whether to merge/re-split a size-1 cluster) ---
SINGLE_FACTOR = 1.5       # tolerance multiplier relative to the target cluster's internal spread (medoid SBD p90) (relative comparison)
SINGLE_FLOOR = 0.12       # lower bound on the mergeable SBD (treat as same shape within this range even if the target is small)
# --- stricter test assignment (opt-in; default False = current nearest-of-all-clusters assignment) ---
ASSIGN_STRICT = False     # True: use only "inside each pulse cluster's train SBD-spread radius" as pulse candidates, and
                          # route outlying beats outside the radius to artifact. Prevents biphasic PVCs from being pulled
                          # into a tight normal c0 (e.g. 001/002 of sddb40). Enable per-series (making it global breaks other series).
ASSIGN_RADIUS_FAC = 2.5   # radius = p90 of within-train-cluster SBD x this
ASSIGN_RADIUS_FLOOR = 0.12 # lower bound on the radius (allow at least this even for a tight cluster so normal micro-variation is not rejected).
                          # 0.12 = boundary that separates normal beats (c0 full-SBD ~0.02-0.05) inside from biphasic PVCs (~0.14-0.20)
                          # outside (0.15 would pull in PVCs such as 67000).
AMP_GATE_ASSIGN = False   # True: amplitude gate on test assignment: if a beat's raw amplitude (ptp) is below AMP_GATE_FRAC times the
                          # candidate cluster's train median amplitude, exclude that (large-amplitude) cluster from candidates. Since SBD
                          # discards amplitude, this uses amplitude to pull back short normal beats crammed just before a PVC (large beat)
                          # (distorted by resampling, so SBD is ambiguous) that get mis-assigned to the large-amplitude c1 (057) to normal c0. Enable per-series.
AMP_GATE_FRAC = 0.5       # if a beat's amplitude is below this ratio of the cluster median amplitude, drop that cluster from candidates.


# ============================================================================= IO
def find_file(series_id):
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{series_id}_*.txt")))
    if not hits:
        raise FileNotFoundError(f"series {series_id} not found in {DATA_DIR}")
    return hits[0]


def parse_meta(path):
    """Read the _<train_end>_<anom_start>_<anom_end> suffix of a UCR filename."""
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
def _autocorr_period(x):
    """Estimate the dominant period by autocorrelation (fallback)."""
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


def detect_rpeaks(x, min_dist=15, height=0.30):
    """Detect R-waves (the sharpest QRS spike; polarity-agnostic via |x-baseline|) on the
    baseline-removed rectified signal. Used for both period estimation and the ECG anchor.
    RPEAK_LOCALNORM (opt-in): detect peaks with local (min,max) normalization rather than global
      99th-percentile normalization. Prevents low-amplitude peaks from falling below height and being
      missed (merging two peaks) on series with large amplitude variation (e.g. 202's smooth sine-like
      oscillation). Default False keeps the legacy behavior."""
    if RPEAK_LOCALNORM:
        win = max(31, int(min_dist * 3) | 1)
        lo = minimum_filter1d(x, win, mode="nearest")
        hi = maximum_filter1d(x, win, mode="nearest")
        rng = hi - lo
        norm = (x - lo) / (rng + 1e-9)              # local 0..1 (absorbs amplitude variation)
        # Flat-region gate: where local amplitude is below a fixed fraction of the median (= straight line/
        #   dropout), normalization approaches 0/0 and mass-produces false peaks, so do not detect there
        #   (avoids fragmenting flat signal-break regions).
        medr = np.median(rng[rng > 0]) if np.any(rng > 0) else 1.0
        norm = np.where(rng > 0.10 * medr, norm, 0.0)
        pk, _ = find_peaks(norm, distance=max(5, int(min_dist)), prominence=PEAK_PROM)
        return pk
    n = len(x)
    win = max(31, (n // 200) | 1)
    base = median_filter(x, size=min(win, 801))
    hp = np.abs(x - base)
    hp = hp / (np.percentile(hp, 99) + 1e-9)
    pk, _ = find_peaks(hp, height=height, distance=max(5, int(min_dist)))
    return pk


def detect_qrs_sharp(x, period):
    """Detect sharp QRS via derivative energy (does not mis-detect a wide T-wave/hump as an R-wave:
    large-T-wave series such as 193). The QRS has a steep slope (high |d/dt|), the T-wave is gentle. Take peaks of dxs^2."""
    xs = savgol_filter(x, max(5, int(period * 0.03) | 1), 2)
    energy = np.gradient(xs) ** 2
    energy = savgol_filter(energy, max(5, int(period * 0.05) | 1), 2)
    energy = energy / (np.percentile(energy, 99) + 1e-9)
    pk, _ = find_peaks(energy, height=QRS_HEIGHT, distance=max(5, int(PEAK_DIST_FRAC * period)))
    return pk


def dominant_period(x):
    """ECG period = median R-R interval. Estimate from R-wave intervals because autocorrelation picks up
    the slow envelope of respiration / apnea / baseline drift and misses the true beat period (166 apneaecg->996,
    194 sddb49->3954 etc.). Fall back to autocorrelation for series where R-waves cannot be detected well."""
    pk = detect_rpeaks(x, min_dist=15)
    if len(pk) >= 5:
        rr = np.diff(pk)
        rr = rr[(rr > np.percentile(rr, 10)) & (rr < np.percentile(rr, 90))]
        if len(rr):
            return int(max(15, np.median(rr)))
    return _autocorr_period(x)


def _local_normalize(xs, win):
    """Normalize to 0..1 locally (min,max) to remove amplitude drift.
    Lets low-amplitude cycles be peak-detected with the same prominence as high-amplitude ones."""
    win = max(5, int(win) | 1)
    lo = minimum_filter1d(xs, win, mode="nearest")
    hi = maximum_filter1d(xs, win, mode="nearest")
    return (xs - lo) / (hi - lo + 1e-9)


def _candidate_feet(xs, norm, period, mode):
    """Return cut-point (foot) candidates per anchor mode. Cut points are always placed at a "low point
    (cycle boundary)", keeping the whole waveform (hump) inside the segment. 'peak' cutting at the hump top is
    not used because it bisects the waveform (user note 127: a single peak was split at its top, 2026-06-25).
      'trough'   : cut at the valley of the local normalization (deepest valley = cycle boundary)
      'peakmin'  : detect humps and cut at the minimum point between them
      'upstroke' : cut at the trough just before each rise (upstroke) (consistent even at a flat baseline valley)
      'downstroke': cut at the low point just after a sharp drop (the falling edge of a square wave/plateau; phase is stable)
      'rr'       : ECG-only. Detect R-waves (QRS) and cut at the TP baseline (the latter-half valley) between adjacent R-waves.
                   Avoids just-after the S-wave and keeps one whole beat (P-QRS-T) inside the segment."""
    dist = max(3, int(period * PEAK_DIST_FRAC))
    if mode == "rr":
        # Detect R-waves on the raw signal (_QRS_SIGNAL) (avoids crushing QRS of weak/anomalous beats under PRE_SMOOTH).
        #   Look for the cut point (TP-baseline valley) on the smoothed xs (so ripple does not jitter the cut-point phase).
        qsig = _QRS_SIGNAL if _QRS_SIGNAL is not None else xs
        R = detect_qrs_sharp(qsig, period) if QRS_SHARP_DETECT else detect_rpeaks(xs, min_dist=dist)
        if len(R) < 2:
            return np.array([], dtype=int)
        feet = []
        if RR_CUT_BEFORE_QRS:
            # Cut just before each QRS (PQ junction). Each segment then begins with "sharp QRS spike -> wide
            #   T-wave hump" (when viewing QRS+T as one unit, as in 193; the first becomes a pre-QRS lead-in partial).
            for r in R:
                a = max(0, r - int(0.20 * period))
                b = max(a + 1, r - int(0.03 * period))
                feet.append(a + int(np.argmin(xs[a:b])))
            return np.array(sorted(set(feet)), dtype=int)
        for i in range(len(R) - 1):
            rr = R[i + 1] - R[i]                  # based on the actual RR interval (tracks RR even when arrhythmia/distortion vary it)
            # When RR is long on a pause/dropped beat (>RR_CAP_FRAC*period), cap the cut window at period-based.
            #   Otherwise the window crosses the first beat's T-wave into the pause region, mislocating the cut (229's RR=254).
            rr_eff = min(rr, int(RR_CAP_FRAC * period)) if RR_CAP_FRAC > 0 else rr
            a = R[i] + int(0.45 * rr_eff)         # after the T-wave (avoid too-early S-wave/T-wave)
            b = R[i] + int(0.78 * rr_eff)         # before the next P-wave/Q-wave (do not grab a too-late pre-QRS valley)
            if b > a:
                feet.append(a + int(np.argmin(xs[a:b])))   # cut at the TP baseline (the central minimum)
        return np.array(feet, dtype=int)
    if mode == "trough":
        f, _ = find_peaks(-norm, distance=dist, prominence=PEAK_PROM)
        return f
    if mode == "upstroke":
        d = np.gradient(xs)
        drng = float(np.percentile(np.abs(d), 99)) + 1e-9
        rises, _ = find_peaks(d, distance=dist, prominence=0.2 * drng)
        return np.array([max(0, r - int(0.5 * period))
                         + int(np.argmin(xs[max(0, r - int(0.5 * period)):r + 1]))
                         for r in rises])
    if mode == "downstroke":
        d = np.gradient(xs)
        drng = float(np.percentile(np.abs(d), 99)) + 1e-9
        drops, _ = find_peaks(-d, distance=dist, prominence=0.2 * drng)
        return np.array([dd + int(np.argmin(xs[dd:min(len(xs), dd + int(0.3 * period))]))
                         for dd in drops])
    pk, _ = find_peaks(norm, distance=dist, prominence=PEAK_PROM)
    if len(pk) < 2:
        return np.array([], dtype=int)
    return np.array([pk[i] + int(np.argmin(xs[pk[i]:pk[i + 1]]))
                     for i in range(len(pk) - 1)])


def _spans_clean(x, feet, period):
    n = len(x)
    bounds = sorted(set([0] + list(feet) + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    lo_len, hi_len = int(period * LEN_LO_FRAC), int(period * LEN_HI_FRAC)
    # Relative amplitude floor: mark flat / very-low-amplitude fragments (standing/calibration phase) as non-clean (= artifact).
    ranges = np.array([float(x[s:e + 1].max() - x[s:e + 1].min()) for s, e in spans])
    amp_floor = AMP_FLOOR_FRAC * float(np.median(ranges)) if len(ranges) else 0.0
    clean = [bool(lo_len <= e - s <= hi_len) and is_cycle_shape(x[s:e]) and (rng >= amp_floor)
             for (s, e), rng in zip(spans, ranges)]
    return spans, clean


def _align_snr(x, spans, clean):
    """Phase-alignment degree of clean cycles = range(mean shape)/mean per-point std (higher = better phase alignment)."""
    cl = [(s, e) for (s, e), c in zip(spans, clean) if c]
    if len(cl) < 3:
        return -1.0
    sh = np.array([znorm(resample(x[s:e + 1])) for s, e in cl])
    m = sh.mean(0)
    return float((m.max() - m.min()) / (sh.std(0).mean() + 1e-9))


def _merge_short_spans(x, feet, period):
    """Merge sub-period fragments much shorter than the period (extra cut points produced by a flat baseline
    etc.) into an adjacent segment, keeping 1 waveform = 1 segment. By default merge into the "next" one
    (the fragment is often the quiet lead-in of the next cycle); only the trailing fragment goes to the
    "prev". Full-span coverage is unchanged (just removes one cut).
    Length guard: do not merge in a direction where the result would exceed the clean length upper bound
    (LEN_HI_FRAC*period) (prevents adding a fragment to a full cycle to make an over-long segment = a separate
    shape cluster). If neither direction is allowed, leave it in place."""
    n = len(x)
    short = int(period * SHORT_SPAN_FRAC)
    hi = int(period * LEN_HI_FRAC)
    bounds = sorted(set([0] + [int(f) for f in feet] + [n]))
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(len(bounds) - 1):
            if bounds[i + 1] - bounds[i] < short:
                prev_ok = i > 0 and (bounds[i + 1] - bounds[i - 1]) <= hi
                next_ok = (i + 1 < len(bounds) - 1) and (bounds[i + 2] - bounds[i]) <= hi
                if i + 1 >= len(bounds) - 1:          # last span -> prev (remove left cut)
                    if prev_ok:
                        bounds.pop(i); changed = True; break
                elif next_ok:                          # default: next (remove right cut)
                    bounds.pop(i + 1); changed = True; break
                elif prev_ok:                          # if next not allowed, prev
                    bounds.pop(i); changed = True; break
                # if both would be over-long, do not merge (leave in place)
    return np.array([b for b in bounds if b not in (0, n)], dtype=int)


def _has_qrs(seg):
    """Whether a span has a sharp QRS (max deviation/MAD >= SHARP_QRS_MIN). Does not savgol-smooth (smoothing
    dulls a narrow PVC spike so sharp drops below the threshold and is mis-judged as a T-wave: 109's PVC beat
    sharp 5.4->2.9). Decide on the raw resample znorm (wide humps like T-waves are ~1.5-2.4, a sharp QRS is ~5,
    clearly separated). QRS presence/absence check used for merging."""
    if len(seg) < 5:
        return False
    m = znorm(resample(seg))
    med = np.median(m)
    mad = np.median(np.abs(m - med)) * 1.4826 + 1e-9
    return float(np.max(np.abs(m - med)) / mad) >= SHARP_QRS_MIN


def _merge_qrs_fragments(x, feet, period):
    """In 'rr' segmentation, if a large T-wave is mis-detected as an R-wave, one beat splits into
    "QRS part + T-wave part" (093's 30000/4250). Merge a fragment without a sharp QRS (T-wave/baseline) into the
    adjacent QRS-containing beat to restore one beat. Since the T-wave follows the QRS, prefer merging into the
    previous beat. Length-guarded (do not merge in a direction where the result > LEN_HI*period). A trailing beat
    that has a QRS is not merged (prevents recurrence of the trailing 2-beat bug)."""
    n = len(x)
    hi = int(period * 2.2)        # merging a QRS-less fragment can only make 1 beat (never 2 beats), so lenient
                                  # (for slow beats QRS+T can span ~2 periods; e.g. 093's 4250)
    # Under QRS_SHARP_DETECT, judge QRS by the presence of already-detected R-waves (avoids the problem where
    #   PRE_SMOOTH dulls the sharpness ratio so _has_qrs mis-judges a real beat as QRS-less -> merges a real beat:
    #   e.g. 209's [14310:14594]).
    if QRS_SHARP_DETECT:
        Rset = detect_qrs_sharp(_QRS_SIGNAL if _QRS_SIGNAL is not None else x, period)
        has_qrs = lambda a, b: bool(np.any((Rset >= a) & (Rset < b)))
    else:
        has_qrs = lambda a, b: _has_qrs(x[a:b])
    bounds = sorted(set([0] + [int(f) for f in feet] + [n]))
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            # A leading/trailing signal-end partial (recording starts/ends mid-beat) is an independent partial,
            # not a T-wave fragment, so do not merge it (093: merging the leading partial into a beat -> over-long ->
            # reconcile re-split it at the QRS). Internal fragments only.
            if not has_qrs(a, b) and a != 0 and b != n:  # internal QRS-less fragment (T-wave part)
                prev_ok = i > 0 and has_qrs(bounds[i - 1], a) and (b - bounds[i - 1]) <= hi
                next_ok = (i + 1 < len(bounds) - 1) and has_qrs(b, bounds[i + 2]) and (bounds[i + 2] - a) <= hi
                if prev_ok:                                # into the previous QRS beat (T-wave preference)
                    bounds.pop(i); changed = True; break
                elif next_ok:                              # into the next QRS beat
                    bounds.pop(i + 1); changed = True; break
    return np.array([b for b in bounds if b not in (0, n)], dtype=int)


def segment_cycles(x, period, split_multibeat=True):
    """Segment the whole signal into one beat at a time with no gaps.
    split_multibeat: whether to re-split 2-beat fused spans at the R-waves. True for train (normal) only; False
      for test: a length-outlier segment in test may be not under-segmentation but "one long anomalous beat", and
      splitting it erases the anomaly signal (dt = segment duration etc.) and breaks detection (001's DISTORTED
      anomaly). train is known normal, so re-splitting fusion artifacts is safe.

    For ECG the R-wave (QRS) is the sharpest, most consistent landmark, so prefer the R-wave-based 'rr' anchor
    (cut at the TP baseline between adjacent R-waves, keeping one whole beat). Comparing by phase-alignment SNR,
    'trough' (which cuts at the S-wave, the sharp valley right after R) scores higher and puts the QRS at the beat
    boundary, splitting the T-wave into the next beat; so for ECG prefer 'rr' over the SNR comparison (PROMPT.md:
    place cuts at low points not peaks, keeping 1 whole waveform / same phase). Only for series where R-waves
    cannot be detected well, fall back to the generic self-tuning anchor (max phase-alignment SNR among
    trough/peakmin/upstroke/downstroke).

    Returns: spans, clean (bool list), feet (cut points of the adopted anchor)
    """
    n = len(x)
    w = max(5, int(period * SMOOTH_FRAC) | 1)
    xs = savgol_filter(x, window_length=w, polyorder=2)
    norm = _local_normalize(xs, win=int(period * LOCALNORM_FRAC))
    # ECG preference: R-wave-based 'rr' anchor. If enough beats are obtained, adopt without the SNR comparison.
    used_rr = False
    rr_feet = _candidate_feet(xs, norm, period, "rr")
    if len(rr_feet) >= 3:
        spans, clean = _spans_clean(x, rr_feet, period)
        if sum(clean) >= max(3, 0.5 * len(spans)):     # adopt 'rr' if the majority are clean beats
            best = (_align_snr(x, spans, clean), spans, clean, rr_feet)
            used_rr = True
        else:
            best = None
    else:
        best = None
    if best is None:                                    # fallback: generic self-tuning anchor
        for mode in ("trough", "peakmin", "upstroke", "downstroke"):
            feet = _candidate_feet(xs, norm, period, mode)
            if len(feet) < 3:
                continue
            spans, clean = _spans_clean(x, feet, period)
            snr = _align_snr(x, spans, clean)
            if best is None or snr > best[0]:
                best = (snr, spans, clean, feet)
    if best is None:
        return [], [], np.array([], dtype=int)
    # The 'rr' anchor cuts between adjacent R-waves, so each span is already 1 beat. To avoid the bug where
    #   _merge_short_spans absorbs a short trailing beat (a true beat whose T-wave is clipped at the signal end to
    #   under 0.45*period) into the previous one, making 2 beats in 1 segment (017/086 trailing end), skip fragment
    #   merging when rr is adopted. Merge only for the generic anchor.
    if used_rr:
        feet = _merge_qrs_fragments(x, best[3], period)    # merge QRS-less fragments (T-wave parts) into QRS beats
    else:
        feet = _merge_short_spans(x, best[3], period)      # merge sub-period fragments into neighbors
    if VALLEY_RECLAIM > 0:                                  # move boundaries cut at a deep valley back to the downstroke onset (TP baseline)
        feet = _reclaim_valleys(x, feet, period)
    if split_multibeat:                                    # train only: re-split 2-beat fusions at the R-waves
        feet = _split_multibeat_feet(x, feet, period)      #   (test does not split long anomalous segments)
    spans, clean = _spans_clean(x, feet, period)
    return spans, clean, feet


def _split_multibeat_feet(x, feet, period):
    """When a span has 2+ QRS (R-waves) (= 2+ beats fused via a missed R-wave), split at the TP baseline
    (latter-half valley) between adjacent R-waves to make 1 beat per span. Fixes ECG under-segmentation.
    A long single-QRS beat (arrhythmia/pause) has one R-wave, so it is not split (avoids over-segmentation).
    R-wave detection uses the same raw signal (_QRS_SIGNAL) and the same min_dist as the 'rr' anchor, binding
    multiple detections within one QRS into one via the period*PEAK_DIST_FRAC distance constraint."""
    qsig = _QRS_SIGNAL if _QRS_SIGNAL is not None else x
    md = max(3, int(period * PEAK_DIST_FRAC))
    R = detect_qrs_sharp(qsig, period) if QRS_SHARP_DETECT else detect_rpeaks(qsig, min_dist=md)
    if len(R) < 2:
        return feet
    n = len(x)
    bounds = sorted(set([0] + [int(f) for f in feet] + [n]))
    extra = []
    for a, b in zip(bounds, bounds[1:]):
        Rin = [int(r) for r in R if a < r < b]
        for r0, r1 in zip(Rin, Rin[1:]):                   # cut between adjacent R-waves within the same span
            lo = r0 + int(0.5 * (r1 - r0))
            hi = max(lo + 1, r1 - int(0.05 * period))       # TP baseline just before the next QRS (latter-half minimum)
            if hi > lo:
                extra.append(lo + int(np.argmin(x[lo:hi])))
    if not extra:
        return feet
    return np.array(sorted(set([int(f) for f in feet] + extra)), dtype=int)


def _reclaim_valleys(x, feet, period):
    """Fix the problem where a cut is placed at the bottom of a PVC's deep valley so the descent into the valley
    falls into the previous segment (e.g. 224 bigeminy). A PVC without a sharp QRS (valley -> wide hump) is not
    R-wave-detected and is cut at the minimum of the previous beat's cut window = the valley bottom. Move cuts
    deeper than VALLEY_RECLAIM*MAD below the baseline leftward to the point where it returns near the baseline
    (the downstroke onset)."""
    base = float(np.median(x))
    mad = float(np.median(np.abs(x - base))) * 1.4826 + 1e-9
    deep = base - VALLEY_RECLAIM * mad
    near = base - 0.5 * mad                                 # baseline band (the downstroke onset once it returns here)
    out = []
    peak = base + 1.5 * mad                                # QRS/peak level (if it crosses this = a separate beat, abort reclaim)
    for f in feet:
        f = int(f)
        if 0 < f < len(x) and x[f] < deep:
            lo = max(0, f - int(0.5 * period))
            i = f
            while i > lo and x[i] < near:
                i -= 1
            # If [i:f] contains a QRS (peak), moving would cross another beat's S-wave and place the cut on the
            #   QRS (224 bigeminy's normal-beat S-wave). In that case keep the original cut without moving.
            if float(np.max(x[i:f + 1])) > peak:
                out.append(f)
            else:
                out.append(i)
        else:
            out.append(f)
    return np.array(sorted(set(out)), dtype=int)


def is_cycle_shape(seg):
    """Whether the shape looks like one beat (P-QRS-T). An ECG beat is "a long baseline + a momentary sharp QRS",
    so Gait's "reject too-flat (flattop)" idea does not apply (for long periods like STAFFIII the QRS occupies a
    small fraction of the beat and everything gets killed as flat). Instead, use "whether a sharply protruding QRS
    exists" as the main condition, and reject flat dropout / no-signal spans (no protrusion) and multi-peak noise
    fragments.
      - QRS protrusion: max deviation is at least SHARP_QRS_MIN times the shape's robust scale (MAD) (a sharp spike).
      - Multi-peak noise: the prominent-peak count of the smoothed shape is NOISE_PEAKS_MAX or fewer."""
    if len(seg) < 5:
        return False
    m = znorm(resample(seg))
    ms = savgol_filter(m, 11, 2)
    # QRS sharpness presence: max deviation from the median / MAD (robust scale). Small for flat spans, large for QRS beats.
    med = np.median(ms)
    mad = np.median(np.abs(ms - med)) * 1.4826 + 1e-9
    sharp = float(np.max(np.abs(ms - med)) / mad)
    npk = len(find_peaks(ms, prominence=0.2)[0]) + len(find_peaks(-ms, prominence=0.2)[0])
    return sharp >= SHARP_QRS_MIN and npk <= NOISE_PEAKS_MAX


# ============================================================================= shape distance (SBD) / features
def sbd_limited(a, b):
    """Shift-limited SBD = 1 - max(normalized cross-correlation), restricted to lags +-MAXLAG_FRAC*L.
    For ECG test assignment. With the 'rr' anchor each beat is already phase-aligned to the R-wave, so this
    prevents full-shift SBD from circular-shifting biphasic PVCs etc. toward the normal QRS and mis-assigning them
    (pulled into cluster0 on 001/002's test). Allows only small lags that preserve QRS position, absorbing just the
    tiny offset of foot detection."""
    L = len(a)
    ml = max(1, int(L * MAXLAG_FRAC))
    cc = np.correlate(a, b, mode="full")
    mid = L - 1
    cc = cc[mid - ml: mid + ml + 1]
    return float(1.0 - cc.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


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
    """Full-shift SBD = 1 - max(normalized cross-correlation). Phase-tolerant (compares by waveform type).
    On a constant (nearly flat) span the cross-correlation can numerically exceed the norm product slightly and
    go negative, so clip the lower bound to 0 with max(0,.) (SBD is mathematically in [0,2])."""
    cc = np.correlate(a, b, mode="full")
    return float(max(0.0, 1.0 - cc.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))


def sbd_full_matrix(mat):
    """Full-shift SBD distance matrix (Stage1: broad grouping by waveform type). Clip negatives to 0 (numerical stability)."""
    n = len(mat)
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cc = np.correlate(mat[i], mat[j], mode="full")
            D[i, j] = D[j, i] = float(max(0.0, 1.0 - cc.max() / (norms[i] * norms[j])))
    return D


def d3_features(w):
    """Stage2 D3 (5 features, RAW values, span = the unit itself, no smoothing).
    [0]range=max-min (raw amplitude), [1]activity=mean(|diff|), [2]curv=max(|diff2|),
    [3]n_extrema=#find_peaks(w,prom=0.10*range)+#find_peaks(-w,same),
    [4]skew=mean(z**3) (z=(w-mean)/std, ddof=0). Length<3 is a zero vector."""
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


# ============================================================================= K selection / merge
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
    """Merge cluster pairs with medoid-to-medoid SBD < MERGE_TAU."""
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
    """Stage2 acceptance guard: merge sub-cluster pairs that are "essentially the same in both shape and
    amplitude submode". The conditions to keep them split are (a) shape difference: medoid full-SBD > STAGE2_SBD, or
    (b) a real amplitude submode: range relative diff > STAGE2_AMP_RANGE and
       (activity relative diff > STAGE2_AMP_ACT or curv relative diff > STAGE2_AMP_CURV).
    -> merges false splits from crossing the n_extrema threshold or cutting a continuous gradient where only range moves."""
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
def _merge_clusters_by_shape(labels, shapes, tau=None):
    """Iteratively merge final-cluster pairs with medoid-to-medoid full-SBD < tau (default MERGE_TAU). Corrects,
    by relative shape comparison, mis-groupings where complete-linkage splits the same morphology into separate
    groups (055: c2 is same-shape as c0 = SBD0.19 yet on the c1 side; 093: same-shape artifacts c3/c4 = SBD0.085
    split apart). Artifacts use ARTIFACT_TAU."""
    tau = MERGE_TAU if tau is None else tau
    labels = np.asarray(labels).copy()
    while len(set(labels)) >= 2:
        cls = sorted(set(labels))
        meds = {}
        for c in cls:
            idx = np.where(labels == c)[0]
            sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
            D = sbd_full_matrix(shapes[sub])
            meds[c] = shapes[sub[int(D.sum(1).argmin())]]
        best = None
        for i, a in enumerate(cls):
            for b in cls[i + 1:]:
                d = sbd_full_pair(meds[a], meds[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None or best[0] >= tau:
            break
        labels[labels == best[2]] = best[1]               # merge the closest pair
    return relabel(labels)


def _reassign_shape_distinct(labels, shapes, ranges=None):
    """Correct clean beats to their SBD-nearest cluster. Because Stage1 complete-linkage uses a max-distance
    criterion, a beat can be assigned to a cluster it is medoid-wise farther from (057: the leading beat is near c0
    yet in c1). But move only between clusters whose shapes clearly differ (medoid SBD > 0.25), and do not touch
    amplitude submodes (Stage2 subdivisions with close shape) (so a shape criterion does not break amplitude assignment).
    Also consider amplitude (when ranges given): when amplitude differs greatly between clusters (median ratio>1.8)
    and a beat's raw amplitude deviates greatly from the current cluster's median (<0.5x or >2x) while matching
    another cluster's median (0.6-1.7x), move it to that cluster by amplitude even if shape SBD is misled by a short
    truncated beat etc. (057 [5644:5698]: amplitude 352 = c0 yet it was in the deep c1)."""
    labels = np.asarray(labels).copy()
    cls = sorted(set(labels))
    if len(cls) < 2:
        return labels
    meds, amed = {}, {}
    for c in cls:
        idx = np.where(labels == c)[0]
        sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
        D = sbd_full_matrix(shapes[sub])
        meds[c] = shapes[sub[int(D.sum(1).argmin())]]
        if ranges is not None:
            amed[c] = float(np.median(ranges[idx]))
    out = labels.copy()
    for j in range(len(labels)):
        cur = labels[j]
        dn = {c: sbd_full_pair(shapes[j], meds[c]) for c in cls}
        near = min(dn, key=dn.get)
        if near != cur and dn[near] < dn[cur] and sbd_full_pair(meds[cur], meds[near]) > 0.25:
            out[j] = near
            continue
        if ranges is not None:                       # correction by amplitude
            r = ranges[j]
            for c in cls:
                if c == cur:
                    continue
                amp_distinct = max(amed[c], amed[cur]) > 1.8 * min(amed[c], amed[cur])
                off_cur = ranges[j] < 0.5 * amed[cur] or ranges[j] > 2.0 * amed[cur]
                match_c = 0.6 * amed[c] <= r <= 1.7 * amed[c]
                if amp_distinct and off_cur and match_c:
                    out[j] = c
                    break
    return relabel(out)


def cluster_two_stage(beats_raw, shapes_robust):
    """Two-stage clustering (structure, operations, and metrics are fixed)."""
    # ---- Stage 1: shape (amplitude ignored) ----
    D1 = sbd_full_matrix(shapes_robust)
    # FORCE_SHAPE_K>0: bypass silhouette and cut shape-SBD complete at a fixed K (forces a morphology that is
    #   continuous but that an expert regards as a type into the desired count: 234's deep-dip type is continuous
    #   with the main so silhouette does not separate it, but K=3 isolates it into 25 beats).
    if FORCE_SHAPE_K > 0 and len(shapes_robust) > FORCE_SHAPE_K:
        Z = linkage(D1[np.triu_indices(len(shapes_robust), 1)], method="complete")
        final = relabel(np.asarray(fcluster(Z, FORCE_SHAPE_K, criterion="maxclust")) - 1)
        info = {"K_s1": FORCE_SHAPE_K, "K_sub": {}, "K": int(len(set(final))), "stage1_sils": {}}
        return final, len(set(final)), info, D1
    lab1, info1 = select_k_precomputed(D1, KMAX_S1, "complete")

    # ---- Stage 2: amplitude/scale + fine shape statistics ----
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
        # Shape-first reassignment (2026-06-26): since subdivisions pass the acceptance guard as "shapes differ",
        #   align beat assignment to a shape criterion too. The Stage2 D3 (amplitude/scale) assignment leaves
        #   boundary beats out of step with shape (003's seg@15000 has mid-range amplitude so it enters c1 but its
        #   shape is c2). For accepted subdivisions (>1), reassign each beat to the SBD-nearest sub-medoid (limited
        #   to between child clusters of the same Stage1 parent = local).
        if len(set(lab2)) > 1:
            subs = sorted(set(lab2))
            Dsub = sbd_full_matrix(shapes_robust[idx])
            medoid = {sub: np.where(lab2 == sub)[0][
                Dsub[np.ix_(np.where(lab2 == sub)[0], np.where(lab2 == sub)[0])].sum(1).argmin()]
                for sub in subs}
            lab2 = np.array([min(subs, key=lambda sub: Dsub[j, medoid[sub]])
                             for j in range(len(idx))])
        k_sub[int(c1)] = len(set(lab2))
        for sub in sorted(set(lab2)):
            final[idx[lab2 == sub]] = nxt
            nxt += 1
    final = relabel(final)
    final = _merge_clusters_by_shape(final, shapes_robust)   # merge same-shape clusters across parents
    ranges = np.array([float(np.ptp(w)) for w in beats_raw])  # raw amplitude (range)
    final = _reassign_shape_distinct(final, shapes_robust, ranges)  # correct misassignment by shape + amplitude
    info = {"K_s1": int(info1["K"]), "K_sub": k_sub, "K": int(len(set(final))),
            "stage1_sils": info1.get("sils", {})}
    return final, len(set(final)), info, D1


def cluster_artifacts(shapes):
    """Group non-clean spans by robust z-norm shape SBD complete
    (SBD < ARTIFACT_TAU = same shape). Singular ones form their own cluster even at n=1."""
    n = len(shapes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    D = sbd_full_matrix(shapes)
    Z = linkage(D[np.triu_indices(n, 1)], method="complete")
    lab = relabel(fcluster(Z, t=ARTIFACT_TAU, criterion="distance") - 1)
    return _merge_clusters_by_shape(lab, shapes, tau=ARTIFACT_TAU)  # medoid merge of same-shape artifacts



# ============================================================================= singleton reconsideration rule
def _medoid_spread(shapes_rz, idx):
    """The medoid shape of cluster idx and its internal spread (90th percentile of SBD to the medoid).
    For large clusters, subsample evenly to <=80 items to cap the O(n^2) SBD (speedup)."""
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
    """Re-split span [s,e] (a long under-segmented unit) at its internal valleys and return the sub-spans."""
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
    # Keep full-span coverage: do not discard too-short fragments; merge them into the neighboring (next) fragment to tile [s,e] with no gaps.
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
    """Singleton/small-cluster reconsideration rule (added 2026-06-25, generalizing a user request):
    (a) **Same-shape reclaim**: if an artifact cluster's medoid (label >= k_orig) is within the nearest pulse
        cluster's internal spread (medoid SBD p90) x SINGLE_FACTOR, merge that whole cluster into pulse
        (reclaims a normal cycle judged non-clean due to noise/distortion into c0 etc. Relative comparison).
    (b) **Re-split**: a size-1 artifact of length ~2 periods or more (under-segmented multi-cycle) is re-split at
        its internal valleys and each piece merged into the nearest cluster (an independent artifact if there is no target).
    (c) **Keep independent**: truly distinct shapes (partial/double/singular) are left as-is.
    Returns: updated spans, labels. (Speedup: cluster medoids are computed once per iteration.)"""
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
        # (a) same-shape reclaim of a whole artifact cluster into pulse
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
        #     Under QRS_SHARP_DETECT, re-split only when there are 2+ QRS (truly under-segmented). Re-splitting one
        #       beat (a signal-start lead-in partial etc.) that slightly exceeds 1.5*period would fragment it (237's [0:160]).
        Rall = detect_qrs_sharp(_QRS_SIGNAL if _QRS_SIGNAL is not None else xt, period) \
            if QRS_SHARP_DETECT else None
        for c in [c for c in ids if c >= k_orig and labels.count(c) == 1]:
            i = labels.index(c)
            s, e = spans[i]
            nqrs = int(np.sum((Rall >= s) & (Rall < e))) if Rall is not None else 2
            if (e - s) > 1.5 * period and nqrs >= 2:
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


_TEMPLATE_CACHE = {}


def _template_pvc_medoids(template_sid):
    """Run the reference series (template_sid) and return the medoid shapes of its PVC (non-normal pulse)
    clusters in descending count order. For cross-series consistent PVC type classification (classify 226 using
    225 as reference). The result is cached."""
    if template_sid in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[template_sid]
    import importlib
    tmod = importlib.import_module(str(template_sid))
    tparams = {k: v for k, v in getattr(tmod, "PARAMS", {}).items() if k != "PVC_TEMPLATE"}
    tparams["PVC_TEMPLATE"] = ""                             # recursion guard: disable it for the template series
    tres = run(template_sid, tparams, save=False)            # run saves/restores globals
    tpath = find_file(template_sid)
    tx = np.loadtxt(tpath).astype(float)
    tend, _, _ = parse_meta(tpath)
    txt = tx[:tend] if tend > 200 else tx
    segs = sorted([s for s in tres["segments"] if s.get("region", "train") == "train"],
                  key=lambda s: s["start"])     # exclude test segments (full coordinates), train only
    kp = tres["info"]["K_pulse"]
    cnt = {}
    for s in segs:
        cnt[s["cluster"]] = cnt.get(s["cluster"], 0) + 1
    pulse = sorted([c for c in cnt if c < kp], key=lambda c: -cnt[c])
    meds = []
    for c in pulse[1:]:                                      # PVC clusters, excluding pulse[0] = normal
        cl = [s for s in segs if s["cluster"] == c]
        A = np.array([resample(robust_znorm(txt[s["start"]:s["end"] + 1])) for s in cl])
        meds.append(A[sbd_full_matrix(A).sum(1).argmin()])
    _TEMPLATE_CACHE[template_sid] = meds
    return meds


def _recluster_pvc_template(xt, spans, labels, k_orig, template_sid):
    """Assign anomalous pulse beats to the nearest reference-series PVC medoid (template) (cross-series consistent
    classification). pulse count = 1 normal + the number of template types used. Returns: (labels, new_k_orig)."""
    meds = _template_pvc_medoids(template_sid)
    if not meds:
        return labels, k_orig
    labels = list(int(l) for l in labels)
    pulse_ids = sorted(c for c in set(labels) if c < k_orig)
    if len(pulse_ids) < 2:
        return labels, k_orig
    normal = max(pulse_ids, key=lambda c: labels.count(c))
    pvc_idx = [i for i in range(len(labels)) if labels[i] < k_orig and labels[i] != normal]
    art_labels = sorted(set(labels[i] for i in range(len(labels)) if labels[i] >= k_orig))
    assign = {}
    for i in pvc_idx:
        v = resample(robust_znorm(xt[spans[i][0]:spans[i][1] + 1]))
        assign[i] = int(np.argmin([sbd_full_pair(m, v) for m in meds]))   # 0..len(meds)-1
    # Sub-split template types whose shape divides (2-way-split medoid SBD > tau)
    #   (subtypes absent from the reference series: 226's initial-hump type has 2 subtypes, deep pre-dip in the first half / biphasic in the second).
    if PVC_TEMPLATE_SPLIT_TAU > 0:
        nextk = len(meds)
        for t in sorted(set(assign.values())):
            members = [i for i in pvc_idx if assign[i] == t]
            if len(members) < 6:
                continue
            sub = np.array([resample(robust_znorm(xt[spans[i][0]:spans[i][1] + 1])) for i in members])
            Ds = sbd_full_matrix(sub)
            # Split by distance threshold (not misled by a lone outlier group). Only groups of size>=2 are true subtypes.
            sl2 = fcluster(linkage(Ds[np.triu_indices(len(members), 1)], "complete"),
                           t=PVC_TEMPLATE_SPLIT_TAU, criterion="distance")
            groups = {g: np.where(sl2 == g)[0] for g in set(sl2)}
            big = sorted([g for g in groups if len(groups[g]) >= 2], key=lambda g: -len(groups[g]))
            # Split only when it cleanly divides into 2-3 subtypes that cover the majority (>=70%)
            # (do not split something that shatters into many groups like the main PVC = diverse but one type).
            if not (2 <= len(big) <= 3) or sum(len(groups[g]) for g in big) < 0.7 * len(members):
                continue
            med_big = {g: groups[g][int(Ds[np.ix_(groups[g], groups[g])].sum(1).argmin())] for g in big}
            glabel = {big[0]: t}                         # the largest group keeps the original label
            for g in big[1:]:
                glabel[g] = nextk
                nextk += 1
            for jpos in range(len(members)):             # assign each beat to the nearest big-medoid (absorbs outliers too)
                g = min(big, key=lambda gg: Ds[jpos][med_big[gg]])
                assign[members[jpos]] = glabel[g]
    used = sorted(set(assign.values()), key=lambda t: -list(assign.values()).count(t))
    remap = {t: r + 1 for r, t in enumerate(used)}          # 1..K' (pulse) in descending count order
    Kf = len(used)
    new = [0] * len(labels)
    for i in pvc_idx:
        new[i] = remap[assign[i]]
    art_map = {c: Kf + 1 + j for j, c in enumerate(art_labels)}
    for i in range(len(labels)):
        if labels[i] >= k_orig:
            new[i] = art_map[labels[i]]
    return new, Kf + 1


def _recluster_pvc_shape(xt, spans, labels, k_orig, K):
    """Group anomalous pulse beats other than normal (the largest pulse cluster) and re-split into K by full-shift
    SBD complete. When the Stage2 (D3 = amplitude-leaning) split conflicts with shape, re-cut PVC types by pure
    shape (225: amplitude {31,24} -> 3 shape types {41 sharp, 12 wide, 2 biphasic}). Reset the pulse cluster count
    to 1 normal + K and return (K_pul=K+1). Returns: (labels, new_k_orig)."""
    labels = list(int(l) for l in labels)
    pulse_ids = sorted(c for c in set(labels) if c < k_orig)
    if len(pulse_ids) < 2:
        return labels, k_orig
    normal = max(pulse_ids, key=lambda c: labels.count(c))    # largest pulse = normal
    pvc_idx = [i for i in range(len(labels)) if labels[i] < k_orig and labels[i] != normal]
    art_labels = sorted(set(labels[i] for i in range(len(labels)) if labels[i] >= k_orig))
    if len(pvc_idx) < K:
        return labels, k_orig
    sh = np.array([resample(robust_znorm(xt[spans[i][0]:spans[i][1] + 1])) for i in pvc_idx])
    D = sbd_full_matrix(sh)
    Z = linkage(D[np.triu_indices(len(pvc_idx), 1)], method="complete")
    sl = np.asarray(fcluster(Z, K, criterion="maxclust"))
    # k-medoids reassignment: iteratively reassign each beat to the nearest medoid from the complete-linkage initial split (resolves mixing).
    for _ in range(20):
        med_j = {}
        for g in range(1, K + 1):
            mem = np.where(sl == g)[0]
            if len(mem):
                med_j[g] = mem[int(D[np.ix_(mem, mem)].sum(1).argmin())]
        new_sl = np.array([min(med_j, key=lambda g: D[j][med_j[g]]) for j in range(len(sl))])
        if np.array_equal(new_sl, sl):
            break
        sl = new_sl
    # Merge cluster pairs with medoid SBD < PVC_MERGE_TAU (fold same-shape PVCs like sharp/wide into the main type: 225).
    if PVC_MERGE_TAU > 0:
        while len(set(sl)) > 1:
            groups = sorted(set(sl))
            med = {g: np.where(sl == g)[0][int(D[np.ix_(np.where(sl == g)[0], np.where(sl == g)[0])].sum(1).argmin())]
                   for g in groups}
            pair = None
            for ai in range(len(groups)):
                for bi in range(ai + 1, len(groups)):
                    d = D[med[groups[ai]]][med[groups[bi]]]
                    if d < PVC_MERGE_TAU and (pair is None or d < pair[2]):
                        pair = (groups[ai], groups[bi], d)
            if pair is None:
                break
            sl[sl == pair[1]] = pair[0]              # merge the larger label into the smaller
    Kf = len(set(sl))
    remap = {g: i + 1 for i, g in enumerate(sorted(set(sl), key=lambda g: -(sl == g).sum()))}
    sl = np.array([remap[s] for s in sl])            # renumber to 1..Kf in descending count order
    # renumber: normal -> 0, PVC groups -> 1..Kf (pulse), artifact -> Kf+1.. (pulse count = Kf+1)
    new = [0] * len(labels)
    for i, s in zip(pvc_idx, sl):
        new[i] = int(s)
    art_map = {c: Kf + 1 + j for j, c in enumerate(art_labels)}
    for i in range(len(labels)):
        if labels[i] >= k_orig:
            new[i] = art_map[labels[i]]
    return new, Kf + 1


def _isolate_deep_dip(xt, spans, labels, k_orig, frac):
    """Move beats whose raw dip depth (median - minimum) exceeds global median*frac into a separate pulse cluster.
    Groups, from raw waveforms, the "deep T-wave undershoot (amplitude feature)" type that znorm shape cannot
    capture (235's 4411/4490/18732). Returns: (labels, new_k_orig)."""
    labels = list(int(l) for l in labels)
    dips = np.array([float(np.median(xt[s:e + 1]) - xt[s:e + 1].min()) for s, e in spans])
    thresh = frac * float(np.median(dips))
    moved = set(i for i in range(len(labels)) if dips[i] > thresh)  # regardless of pulse/artifact
    if not moved:
        return labels, k_orig
    out = [(l + 1 if l >= k_orig else l) for l in labels]   # shift existing artifacts up by one
    for i in moved:
        out[i] = k_orig                                       # new deep-dip pulse cluster
    return out, k_orig + 1


def _force_abnormal(xt, spans, labels, k_orig):
    """Force clean beats whose start falls in an ABNORMAL_SPANS time range out of c0 into an anomaly cluster.
    Carves out a disturbed episode (209's 12076-13800: a continuous span with transient amplitude drop/inversion
    that SBD alone cannot fully capture) by an expert-specified time range. Beats within the range are sub-clustered
    by shape (full-shift SBD complete), separating differently-shaped beats (large variation/inversion etc.) into a
    separate cluster (granularity tuned by ABNORMAL_TAU)."""
    labels = list(int(l) for l in labels)
    moved = [i for i, (s, e) in enumerate(spans)
             if labels[i] < k_orig and any(a <= s < b for a, b in ABNORMAL_SPANS)]
    if not moved:
        return labels
    base = max(labels) + 1
    if len(moved) == 1:
        labels[moved[0]] = base
        return labels
    sub = np.array([resample(robust_znorm(xt[spans[i][0]:spans[i][1] + 1])) for i in moved])
    D = sbd_full_matrix(sub)
    Z = linkage(D[np.triu_indices(len(moved), 1)], method="complete")
    sl = fcluster(Z, t=ABNORMAL_TAU, criterion="distance")
    for i, s in zip(moved, sl):
        labels[i] = base + int(s) - 1
    return labels


def _isolate_distinct(xt, spans, labels, k_orig, tau):
    """Separate clean beats whose medoid SBD protrudes beyond tau, within the normal cluster (pulse), into a
    separate cluster. Carves out of c0 ectopic beats / disturbed spans that arise naturally during recording
    (209's inverted beat @13209 etc.). Full-shift SBD captures inversion/morphology differences. Leading/trailing
    partial beats are expected to stay below tau and remain."""
    labels = list(int(l) for l in labels)
    shapes = [resample(robust_znorm(xt[s:e + 1])) for s, e in spans]
    moved = []
    for c in [c for c in set(labels) if c < k_orig]:
        idx = [i for i in range(len(labels)) if labels[i] == c]
        if len(idx) < 3:
            continue
        sub = np.array([shapes[i] for i in idx])
        med = sub[sbd_full_matrix(sub).sum(1).argmin()]
        for i in idx:
            if sbd_full_pair(med, shapes[i]) > tau:
                moved.append(i)
    if moved:
        new_label = max(labels) + 1
        for i in moved:
            labels[i] = new_label
    return labels


def _consolidate_noise(xt, spans, labels, k_orig):
    """Group high-frequency-noise (signal degradation / electrode detachment) artifact segments into one noise
    cluster. Roughness = 2nd-diff RMS / range. Noise is large (~0.2-0.9) versus a smooth pulse (~0.07) (206's
    trailing degraded region). Merges mutually-incoherent noise that would otherwise form individual singletons
    (keeps other artifacts with roughness < threshold)."""
    labels = list(int(l) for l in labels)

    def rough(s, e):
        seg = xt[s:e + 1].astype(float)
        rng = float(seg.max() - seg.min()) + 1e-9
        return float(np.sqrt(np.mean(np.diff(seg, 2) ** 2)) / rng) if (e - s) > 3 else 0.0
    noise_i = [i for i, (s, e) in enumerate(spans)
               if labels[i] >= k_orig and rough(s, e) > NOISE_ROUGH_TAU]
    if len(noise_i) >= 2:
        target = min(labels[i] for i in noise_i)
        for i in noise_i:
            labels[i] = target
    return labels


def _apply_reconcile(xt, spans, labels, period, k_orig, clean_set):
    """Apply reconcile_singletons, re-judge pulse/artifact, and renumber with pulse first.
    pulse cluster = original pulse labels (< k_orig) that remain. New labels produced by splitting are artifacts."""
    spans, labels = reconcile_singletons(xt, spans, labels, period, k_orig)
    if NOISE_ROUGH_TAU > 0:                          # merge high-frequency-noise artifacts into one
        labels = _consolidate_noise(xt, spans, labels, k_orig)
    if ISOLATE_SBD > 0:                              # separate protruding beats (ectopic/inverted) within the normal cluster
        labels = _isolate_distinct(xt, spans, labels, k_orig, ISOLATE_SBD)
    if ABNORMAL_SPANS:                               # move beats in the specified time range into an anomaly cluster (sub-split by shape)
        labels = _force_abnormal(xt, spans, labels, k_orig)
    if PVC_TEMPLATE:                                 # classify PVCs by a reference-series template (cross-series consistent)
        labels, k_orig = _recluster_pvc_template(xt, spans, labels, k_orig, PVC_TEMPLATE)
    elif PVC_SHAPE_K > 0:                            # re-split the anomalous pulse cluster into K by shape SBD
        labels, k_orig = _recluster_pvc_shape(xt, spans, labels, k_orig, PVC_SHAPE_K)
    if DEEP_DIP_FRAC > 0:                            # move deep-dip beats into a separate pulse cluster by raw dip depth (amplitude-feature-based)
        labels, k_orig = _isolate_deep_dip(xt, spans, labels, k_orig, DEEP_DIP_FRAC)
    present = sorted(set(labels))
    pulse_ids = [c for c in present if c < k_orig]
    art_ids = [c for c in present if c >= k_orig]
    order = {}
    for k, c in enumerate(sorted(pulse_ids, key=lambda c: -labels.count(c))):
        order[c] = k
    for k, c in enumerate(sorted(art_ids, key=lambda c: -labels.count(c))):
        order[c] = len(pulse_ids) + k
    labels = [order[c] for c in labels]
    # Sort spans by start position (readability: PDF/CSV in chronological order)
    sidx = sorted(range(len(spans)), key=lambda i: spans[i][0])
    spans = [spans[i] for i in sidx]
    labels = [labels[i] for i in sidx]
    return spans, labels, len(pulse_ids), len(art_ids), None
# ============================================================================= driver
def process(series_id, save=True):
    path = find_file(series_id)
    x = np.loadtxt(path).astype(float)
    train_end, anom_s, anom_e = parse_meta(path)
    xt = x[:train_end] if train_end > 200 else x   # learn base waveforms on the normal training span
    global _QRS_SIGNAL
    # detect_qrs_sharp (R-wave detection) always uses the raw signal. segment_cycles passes smoothed xs, but
    # detect_qrs_sharp smooths internally too, so double smoothing crushes weak/near QRS (223's PVC-adjacent normal beat 51768 etc.) -> single smoothing.
    _QRS_SIGNAL = xt
    if PRE_SMOOTH_WN > 0:                           # low-pass to remove high-frequency ripple (measurement noise) (209 Fantasia etc.)
        bb, aa = butter(3, PRE_SMOOTH_WN)            # _QRS_SIGNAL stays the raw signal (before ripple removal)
        xt = filtfilt(bb, aa, xt)

    period = PERIOD if PERIOD else dominant_period(xt)
    spans, clean, peaks = segment_cycles(xt, period)
    clean_idx = [i for i, c in enumerate(clean) if c]
    if len(clean_idx) < 3:
        print(f"[{series_id}] segmentation failed (period={period}, clean={len(clean_idx)})")
        return None

    shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    beats_raw = [xt[s:e + 1] for s, e in spans]

    # Two-stage clustering on clean units only (not misled by artifacts)
    ci = clean_idx
    labels_c, K, info, _ = cluster_two_stage([beats_raw[i] for i in ci], shapes_rz[ci])

    labels_plot = np.full(len(spans), -1, dtype=int)
    for k, i in enumerate(ci):
        labels_plot[i] = int(labels_c[k])

    # Non-clean spans = artifact clusters (numbered after the pulse clusters)
    nc = [i for i in range(len(spans)) if labels_plot[i] < 0]
    n_art = 0
    if nc:
        art = cluster_artifacts(shapes_rz[nc])
        for j, i in enumerate(nc):
            labels_plot[i] = K + int(art[j])
        n_art = int(art.max()) + 1

    # Singleton reconsideration rule: merge/re-split size-1 / small clusters (relative comparison). Keep pulse-derived cycles.
    clean_set = set(clean_idx)
    spans, labels_plot, K, n_art, clean_set = _apply_reconcile(
        xt, spans, list(labels_plot), period, K, clean_set)
    labels_plot = np.array(labels_plot)
    K_total = K + n_art
    info["K_pulse"], info["K_artifact"] = K, n_art

    # For the overlay plot, use std-based z-norm
    shapes_plot = np.array([znorm(resample(xt[s:e + 1])) for s, e in spans])

    # ---- Apply the learned segmentation to the test span (after train_end) too, assigning each beat to the nearest train pulse cluster medoid ----
    #      (base waveforms are learned on train only. test is not re-clustered but classified by nearest medoid = same structure as VLM-AD).
    final_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    pulse_medoids = {}
    clu_size = {}
    for c in range(K_total):       # medoids of all pulse + artifact clusters
        idx = np.where(labels_plot == c)[0]
        clu_size[c] = len(idx)
        if len(idx):
            sub = final_rz[idx]
            pulse_medoids[c] = sub[sbd_full_matrix(sub).sum(1).argmin()] if len(idx) > 1 else sub[0]

    def _clean_by_medoid(shape, lab):
        """Per-beat clean decision: clean if within SBD<=CLEAN_SBD of its cluster medoid and cluster size>=CLEAN_MIN_N."""
        if not CLEAN_BY_MEDOID:
            return lab < K                                   # legacy: pulse cluster (< K_pulse) = clean
        return clu_size.get(lab, 0) >= CLEAN_MIN_N and \
            float(sbd_full_pair(pulse_medoids[lab], shape)) <= CLEAN_SBD
    clean_train = [bool(_clean_by_medoid(final_rz[i], int(labels_plot[i]))) for i in range(len(spans))]
    # For AMP_GATE_ASSIGN: each cluster's train raw-amplitude (ptp) median
    cluster_amp = {}
    for c in range(K_total):
        idx = np.where(labels_plot == c)[0]
        if len(idx):
            cluster_amp[c] = float(np.median([np.ptp(xt[spans[i][0]:spans[i][1] + 1]) for i in idx]))
    test_spans, test_labels, clean_test = [], [], []
    x_disp = xt
    if 200 < train_end < len(x) and pulse_medoids:
        x_test_raw = x[train_end:].astype(float)
        _QRS_SIGNAL = x_test_raw                          # R-wave detection uses the raw signal for test too (matches train)
        xtt = filtfilt(bb, aa, x_test_raw) if PRE_SMOOTH_WN > 0 else x_test_raw
        ts, tc, _ = segment_cycles(xtt, period, split_multibeat=TEST_SPLIT_MULTIBEAT)   # normally test is not fusion-split (keep long anomalous beats).
        #   For series where normal beats fuse into 2 peaks in 1 beat on smooth low-amplitude waves, set TEST_SPLIT_MULTIBEAT=True to re-split only 2-QRS fusions.
        art_keys = [k for k in pulse_medoids if k >= K]    # artifact clusters (pulse is 0..K-1)
        # ASSIGN_STRICT (opt-in): use each pulse cluster's within-train SBD spread (p90xFAC, floor) as a radius,
        #   and make only pulses within the radius candidates. Route outlying beats outside the radius (biphasic PVCs
        #   etc.) to artifact / other pulses. Distance is full-shift SBD (phase-invariant). A normal beat phase-shifted
        #   by foot jitter is still near c0 under full-shift and falls within the radius (shift-limited had the problem
        #   of wrongly rejecting a phase-shifted normal beat from c0 and sending it to c1). After robust z-norm a PVC is
        #   still full-SBD ~0.14-0.20 from the c0 medoid, separable from normal beats (~0.02-0.05) by placing the radius
        #   (floor 0.12) between them (verified on 001/002/003: keep normal c0, exclude PVCs from c0). On series with a
        #   tight c0, PVCs fall outside the radius -> elsewhere; on series with a wide c0, normal beats fall inside -> keep
        #   c0, adaptively. Default (False) is the current full-shift nearest assignment (other series unchanged). Enable for the sddb40 family.
        _dist = sbd_full_pair
        radius = {}
        if ASSIGN_STRICT:
            for c in range(K):
                idx = np.where(labels_plot == c)[0]
                if len(idx) >= 3:
                    d = [_dist(pulse_medoids[c], final_rz[i]) for i in idx]
                    radius[c] = max(float(np.percentile(d, 90)) * ASSIGN_RADIUS_FAC, ASSIGN_RADIUS_FLOOR)
                else:
                    radius[c] = 0.25
        for (s, e), cln in zip(ts, tc):                    # assign all test segments to the nearest cluster (do not set -1)
            shp = resample(robust_znorm(xtt[s:e + 1]))
            # A clean (beat-like) beat goes to the nearest of all clusters. A non-clean one (length/shape outlier =
            #   fusion or pause etc.) is not allowed to be pulled into a normal pulse cluster; take the nearest only
            #   within the artifact clusters (prevents false detection).
            cand = pulse_medoids if cln or not art_keys else {k: pulse_medoids[k] for k in art_keys}
            dists = {k: float(_dist(cand[k], shp)) for k in cand}
            if ASSIGN_STRICT and cln:
                within = [k for k in cand if k < K and dists[k] <= radius.get(k, 0.25)]
                if within:
                    lab = min(within, key=lambda k: dists[k])           # nearest pulse within the radius
                elif art_keys:
                    lab = min(art_keys, key=lambda k: dists[k])         # outlier -> artifact
                else:
                    lab = min(dists, key=dists.get)
            elif AMP_GATE_ASSIGN and cln:
                # A small-amplitude beat is not assigned to a large-amplitude cluster (amplitude-matching clusters only -> nearest if none).
                ampb = float(np.ptp(xtt[s:e + 1]))
                gated = [k for k in dists if ampb >= AMP_GATE_FRAC * cluster_amp.get(k, 0.0)]
                lab = min(gated, key=lambda k: dists[k]) if gated else min(dists, key=dists.get)
            else:
                lab = min(dists, key=dists.get)            # default: nearest of all clusters (full-shift)
            test_spans.append((int(s + train_end), int(e + train_end)))
            test_labels.append(int(lab))
            clean_test.append(bool(_clean_by_medoid(resample(robust_znorm(xtt[s:e + 1])), int(lab))))
        x_disp = np.concatenate([xt, xtt])                 # for display: concatenate train (smoothed) + test (smoothed)

    print(f"[{series_id}] train_len={train_end:7d} period={period:5d}  "
          f"clean={len(ci)} K_pulse={K} K_artifact={n_art}  K={K_total} "
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
    """PDF output: wrap at ROW_STEPS steps per row, vertical dashed lines at cut points, and background-highlight
    each span in its cluster color (full-span coverage). The final page shows per-cluster overlays + mean.
    If test_spans/test_labels are given, also draw the test span (after train_end) in the color assigned to the
    train pulse cluster medoid. train/test boundary = solid blue, injected anomaly span (anom_s..anom_e) = solid red + light-red band."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    n = len(x)
    all_spans = list(spans) + list(test_spans or [])
    all_labels = list(labels) + list(test_labels or [])
    out = os.path.join(OUT_DIR, f"{series_id}.pdf")
    with PdfPages(out) as pdf:
        nrows = int(np.ceil(n / ROW_STEPS))
        # For long series, cap the max page count to avoid a bloated PDF. But always show up to the injected anomaly span anom_e.
        max_pages = 8
        if anom_e:
            max_pages = max(max_pages, int(np.ceil((anom_e + ROW_STEPS) / ROW_STEPS / ROWS_PER_PAGE)) + 1)
        max_pages = min(max_pages, 20)
        rows_cap = max_pages * ROWS_PER_PAGE
        # Unify the y-axis scale across all rows (so amplitudes are comparable between rows). Range of all displayed data + margin.
        disp = x[:min(n, rows_cap * ROW_STEPS)]
        ypad = 0.05 * (float(disp.max()) - float(disp.min()) + 1e-9)
        ylo, yhi = float(disp.min()) - ypad, float(disp.max()) + ypad
        for pg in range(0, min(nrows, rows_cap), ROWS_PER_PAGE):
            rows = list(range(pg, min(pg + ROWS_PER_PAGE, nrows)))
            fig, axs = plt.subplots(len(rows), 1, figsize=(16, 1.5 * len(rows)),
                                    squeeze=False)
            for ax, r in zip(axs[:, 0], rows):
                w0, w1 = r * ROW_STEPS, min((r + 1) * ROW_STEPS, n)
                # Light-red band of the injected anomaly span (bottom background layer)
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
                for ab in (anom_s, anom_e):                # anomaly span boundaries
                    if ab and w0 <= ab < w1:
                        ax.axvline(ab, color="red", lw=1.4, ls="-", zorder=5)
                ax.set_xlim(w0, w0 + ROW_STEPS)
                ax.set_ylim(ylo, yhi)                  # unify y-axis scale across all rows
                ax.set_ylabel(f"{w0}", fontsize=7, rotation=0, ha="right", va="center")
                ax.tick_params(labelsize=6)
            kp = "" if k_pulse is None else f" (pulse 0..{k_pulse-1}, artifact {k_pulse}..)"
            extra = "" if nrows <= rows_cap else f"  [showing first {rows_cap} rows of {nrows}]"
            ttl = (f"{series_id}  train(len={train_end})+test applied  |  "
                   f"train_seg={len(spans)} test_seg={len(test_spans or [])}  K={K}{kp}   "
                   f"[blue=train/test boundary, red=injected anomaly, bg=cluster color]{extra}")
            axs[0, 0].set_title(ttl, fontsize=10)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- Final page: per-cluster overlay + mean ----
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
           "SMOOTH_FRAC", "LEN_LO_FRAC", "LEN_HI_FRAC", "SHORT_SPAN_FRAC", "RR_CUT_BEFORE_QRS", "TEST_SPLIT_MULTIBEAT", "QRS_SHARP_DETECT", "RPEAK_LOCALNORM", "FLATTOP_MAX", "NOISE_PEAKS_MAX",
           "AMP_FLOOR_FRAC", "SHARP_QRS_MIN",
           "KMAX_S1", "KMAX_S2", "MAXLAG_FRAC", "MERGE_TAU", "SIL_MIN", "MIN_PATTERN",
           "MAHAL_RIDGE", "STAGE2_SBD", "STAGE2_AMP_RANGE", "STAGE2_AMP_ACT",
           "STAGE2_AMP_CURV", "ARTIFACT_TAU", "SINGLE_FLOOR", "SINGLE_FACTOR", "NOISE_ROUGH_TAU", "PRE_SMOOTH_WN", "ISOLATE_SBD", "QRS_HEIGHT", "ABNORMAL_SPANS", "ABNORMAL_TAU", "PVC_SHAPE_K", "VALLEY_RECLAIM", "PVC_MERGE_TAU", "PVC_TEMPLATE", "PVC_TEMPLATE_SPLIT_TAU", "RR_CAP_FRAC", "FORCE_SHAPE_K", "DEEP_DIP_FRAC",
           "ASSIGN_STRICT", "ASSIGN_RADIUS_FAC", "ASSIGN_RADIUS_FLOOR",
           "AMP_GATE_ASSIGN", "AMP_GATE_FRAC")


def run(series_id, params=None, save=True):
    """Public API to process one series. Called from `<id>.py`.
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
    ap.add_argument("--series", nargs="+", help="series id(s) e.g. 127 167")
    ap.add_argument("--all", action="store_true", help="all ECG series")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ids = domain_series_ids() if args.all else (args.series or ["127", "167", "170", "181", "248", "239"])
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
