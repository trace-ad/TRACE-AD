#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RESP basic-waveform segmentation + clustering engine (core) -- built from scratch
================================================================

Shared engine that segments the respiration waveform (Respiration Rate) into "basic-waveform patterns = 1 breath"
and clusters them by shape. Per-series execution is handled by thin `<id>.py` files (which just call core.run).

Characteristics of the RESP signal (differences from ABP)
--------------------------------
- One unit = one breath. A smooth quasi-periodic signal, a **single-peak asymmetric wave** (steep inhalation peak +
  gentle exhalation valley). The period is ~470 steps in this data.
- There are no morphology assumptions like ABP's "systolic peak is toward the front" or "dicrotic notch".
  The peak position can vary per series, so no phase-position filter is imposed.
- Some series (the NOISE family) have superimposed high-frequency noise. The clean decision uses the smoothed shape,
  so normal breaths are not wrongly removed even in noise-superimposed series.

Segmentation policy
--------
- Learn the basic waveform only from the training region (UCR: the leading train_end is normal).
- Treat foot (valley) to the next foot as one breath, and include the signal ends as boundaries so the whole range is covered with no gaps.
  To avoid over-segmentation, impose a lower bound on the inter-peak distance (~0.55*period).
- To be robust to amplitude drift, detect peaks on a locally amplitude-normalized signal so low-amplitude
  breaths are not missed (prevents under-segmentation).
- clean filter: exclude those whose length is outside the period / saturation trapezoids / multi-peak noise fragments.

Clustering (configuration, operations, and metrics are fixed. Only parameters vary per series)
----------------------------------------------------------------
- Stage1 = shape (amplitude ignored): robust z-norm + common-length resample, full-shift SBD +
  complete-linkage, K_s1 by relative silhouette comparison over {1..KMAX_S1}.
- Stage2 = amplitude/scale + fine shape statistics: globally standardize D3 (5 features, RAW) over all units
  -> Mahalanobis whiten -> Ward. Subdivide each Stage1 cluster by K_sub {1..KMAX_S2}.
  Acceptance guard: adopt only when there is a shape difference (medoid full-SBD>STAGE2_SBD) or a real amplitude submode
  (range relative difference>STAGE2_AMP_RANGE and activity/curv relative difference>threshold).
- Artifact clusters: group non-clean regions by robust z-norm shape SBD complete
  (SBD<ARTIFACT_TAU = same shape), numbered after the pulse clusters.

Usage
------
  python 076.py            # process series 076 -> output/076.pdf, etc.
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
# The TSB-AD-U subset this folder covers = SED (Medical pulse trains, 3 series).
# For a different subset, copy the folder and change SUBSET. Data is CSV (header: Data,Label).
# Filename <gid>_<SUBSET>_id_<n>_<domain>_tr_<train_end>_1st_<x>.csv. The leading train_end is normal (semi-supervised).
# SED (234/235/236) is a 3-level square-pulse train (full revision 2026-10-07). After min-max (train-statistics) normalization it has
#   three levels: **baseline 0.5 / UP pulse -> 1.0 / DOWN pulse -> 0.0**. In raw values, excursions of +/-300 about baseline 0 (UP=+300,
#   DOWN=-300, both ~32 wide). In train the excursions are perfectly regular (gap=35, pulse=32 -> pitch 67), but
#   in test the gaps become irregular at 35/64/99... due to pulse dropouts/anomalies.
# Problem with the old implementation: fixed 67-grid segmentation (GRID_ANCHOR) drifted with the irregular test gaps, and cuts
#   landed on pulse rising/falling edges or bisected a pulse. Also the old comment said "only ON pulses about baseline 0", looking only at
#   UP and ignoring the DOWN excursion (an incomplete description).
# Current policy = event/edge-based segmentation (EVENT_SEGMENT, default True): detect each excursion by threshold crossings about baseline 0.5
#   (v>up_thr=UP, v<dn_thr=DOWN) and make **1 excursion = 1 segment**. Place cuts on the baseline between adjacent
#   pulses (PULSE_MARGIN inside the edge) so a pulse is not bisected. UP and DOWN types are
#   anti-correlated under z-norm+SBD (SBD~1), so **K_pulse=2** is appropriate (both are normal repeating pulses).
#   The baseline between pulses is a separate "flat" segment, covering the whole range with no gaps (gaps=0), and flat is non-clean=artifact.
#   "1 pulse = 1 segment" is consistent even at irregular spacing. Anomalies are picked up by per-beat clean (a beat that floats off the medoid = clean0).
# Label is used only as an evaluation reference (red-band display of anomaly regions), not for clustering.
SUBSET = "SED"
DATA_DIR = "./data/TSB-AD-U"
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "output")

# ----------------------------------------------------------------------------- defaults (can be overridden per series via PARAMS)
ROW_STEPS = 2000          # time steps per PDF row
ROWS_PER_PAGE = 12        # rows per PDF page
RESAMPLE_LEN = 256        # resample each breath to this length for shape comparison

# --- segmentation (for the SED square-pulse train) ---
PERIOD = 67               # In SED, pulses stand on a 67-grid (with dropouts). Segment with a fixed 67-window aligned to the
                          #   pulse phase (GRID_ANCHOR). Auto-detection would be dominated by the long gaps and give ~268,
                          #   missing close pulses (67 spacing), so default to 67.
GRID_ANCHOR = False       # Old SED segmentation: estimate the pulse phase from the rising edge and segment with a period-wide fixed grid.
                          #   This was the cause of the regression where the test irregular gaps drift and cuts land on edges,
                          #   so default False (replaced by EVENT_SEGMENT). Toggling can reproduce the old behavior.
ONSET_CYCLE = True        # Current SED segmentation (2026-10-08): 1 cycle = 1 beat. Treat from a pulse rising edge (onset) to the next onset
                          #   as one beat = [pulse + the baseline right after it] (cycle_onset_segments). Do not make a baseline-only standalone
                          #   segment (only the leading intro baseline is an exception). -> clean beats are contiguous in train, and
                          #   TRACE-AD's Period channel tbposwl (2-beat prev->cur comparison) can form reference pairs.
                          #   UP/DOWN are anti-correlated under z-norm+SBD so K_pulse=2 (both normal beats). Dropout/irregular/amplitude
                          #   anomalies are picked up by per-beat medoid clean (the beat shape deviates from the medoid, clean=0).
EVENT_SEGMENT = False     # Old SED segmentation (event/edge-based, excursion_segments). Because it cuts pulse beats (clean1) and baseline beats
                          #   (clean0) into separate segments, clean alternates 0,1,0,1... and there are 0 contiguous clean pairs
                          #   -> tbposwl could not form reference pairs. Replaced by ONSET_CYCLE (toggling can reproduce it).
ON_THRESH_FRAC = 0.5      # excursion threshold: UP = v > base + this*(p98 - base), DOWN = v < base - this*(base - p2).
                          #   base = median(signal). At 0.5, the 3-level signal gives up_thr~0.75 / dn_thr~0.25.
PULSE_MARGIN_FRAC = 0.09  # the shoulder (margin) by which a pulse segment extends toward the baseline = this*PERIOD (~6 steps). Place cuts
                          #   margin inside the edge toward the baseline to prevent splitting a pulse (capped at half the adjacent gap).
MIN_PULSE_W = 5           # a tiny deviation whose excursion width is below this is not treated as a pulse but included in the baseline (removes 1-2 point noise).
FLAT_DEV_MIN = 0.25       # per-beat clean flat guard: a segment whose raw amplitude (max-min) is below this = a baseline-only
                          #   flat segment -> not a pulse, so not clean (flat is artifact).
PEAK_DIST_FRAC = 0.55     # minimum inter-peak distance = this ratio * period (suppresses over-segmentation)
PEAK_PROM = 0.25          # lower bound on peak prominence on the locally normalized (0..1) signal
LOCALNORM_FRAC = 1.5      # local amplitude normalization window = this ratio * period
SMOOTH_FRAC = 0.06        # savgol window before peak detection = this ratio * period
LEN_LO_FRAC = 0.5         # clean length lower bound = this ratio * period (a grid cell is roughly period long)
LEN_HI_FRAC = 1.15        # clean length upper bound = this ratio * period. For the ONSET_CYCLE length guard: against normal beats <=period(67),
                          #   an anomalous (pulse-dropout) beat is ~2*period(131). 1.15*67~77 keeps normal (<=67) and rejects dropout beats (79+)
FLATTOP_MAX = 0.7         # if the flat fraction near the top is at least this = flat trapezoid (excluded)
                          #   Loose 0.7 to keep SED's one-pulse cell (ON width ~32 / period 67 ~ 0.48) clean.
                          #     A fully flat empty cell (flattop~1.0) is excluded and becomes an artifact.
NOISE_PEAKS_MAX = 4       # if the number of peaks in the smoothed shape exceeds this = multi-peak noise fragment (excluded)
SHAPE_SMOOTH_WIN = 31     # savgol window for peak counting in is_breath_shape (on the 256-resample).
                          #   A2 is a noisy sinusoid, so wider (11->31) to crush small noise peaks.
SHAPE_PK_PROM = 0.40      # prominence lower bound for that peak count (0.2->0.4, noise hardening).
# Aperiodic fallback: for series where periodic segmentation yields 0 clean cycles (= trend+noise dominated, no sinusoid),
#   re-segment with fixed-length windows covering the whole range and treat "1 window = 1 unit" (sliding fixed-window method).
#   Noise/trend slices are the normal reference; windows containing a spike are anomalies (train excludes labels).
FIXED_WIN = 140           # the window length on fallback (train~710 -> about 5 windows. Smaller gives richer references).
MIN_PERIOD = 12           # Real traffic's true period is daily ~24. A detected period below this is a noise misdetection
                          #   (e.g. period~4 on a dense spike train) -> treat as aperiodic and fall back to fixed windows (prevents runaway over-segmentation).
MIN_SEGS_ND = 2           # NAB: for a non-daily (autocorr) period, if the train segment count is below this it is under-segmentation (spurious long lag,
                          #   normal references run out) -> fixed windows. An exception if a daily period was significant and chosen.
MAX_SEGS = 22             # NAB: for a non-daily period, if the train segment count exceeds this the windows are too fine, a pseudo short period
                          #   (e.g. period36 -> 56 splits) -> fixed windows. Daily period is an exception.
# Detection of saturated (measurement-range ceiling-clipped) beats: RESP contains clipped beats that stick to the ADC ceiling
# (~+/-2000). These are "saturation trapezoids" = non-clean (the spec's clean-exclusion condition). The post-z-norm flat fraction
# cannot catch a narrow clip, so detect it directly on the raw waveform by "proximity to the global ceiling x the length of the
# contiguous same-value run at the top" and consistently send it to artifacts across all series (cross-evaluation-AI note
# 2026-06-24). Normal breaths do not touch the ceiling (run 1-3), saturated beats have runs of 10-50.
CLIP_CEIL_FRAC = 0.02     # if a segment peak is at or above the global (max - this*amplitude) it is near the ceiling
CLIP_RUN_MIN = 8          # near the ceiling and a near-peak contiguous run of at least this = saturation clip (excluded)

# --- clustering K selection/merging (configuration fixed, only thresholds adjusted) ---
KMAX_S1 = 3               # Stage1 K_s1 candidate upper bound
KMAX_S2 = 3               # Stage2 K_sub candidate upper bound
MAXLAG_FRAC = 0.10        # lag upper bound of the shift-limited SBD used for nearest-neighbor assignment of non-clean units
MERGE_TAU = 0.10          # if medoid-to-medoid SBD is below this, merge as same shape
SIL_MIN = 0.12            # if the best silhouette is below this, do not split (K=1)
MIN_PATTERN = 4           # smaller than this size is a one-off outlier, not a recurring pattern -> absorb
MAHAL_RIDGE = 1e-3        # Stage2 covariance ridge (stabilizes Mahalanobis whiten)
STAGE2_SBD = 0.12         # if the full-SBD between subcluster medoids exceeds this, "the shape differs"
                          # In A3, tiny phase/noise differences of the synthetic bimodal ride at SBD 0.09-0.11 and cause false splits.
                          # Assuming a single submode, raise to 0.12 to merge the tiny differences (a clear difference above 0.12 is kept).
STAGE2_AMP_RANGE = 0.28   # if the amplitude range relative difference exceeds this, and ->, then "amplitude submode"
                          # A1 is real traffic with real multimodes (busy/idle), so do not fully disable the amplitude branch as in A4;
                          #   keep a mild relaxation 0.18->0.28. This absorbs false splits that divide the same z-norm
                          #   shape by amplitude alone (TS12/53/67, etc.) while keeping real distinct-shape modes
                          #   (the real spike modes of TS16/26 or TS21) via the shape route. Evaluation-agent consensus.
STAGE2_AMP_ACT = 0.15     #   activity relative difference exceeds this
STAGE2_AMP_CURV = 0.35    #   or curv relative difference exceeds this
ARTIFACT_TAU = 0.10       # non-clean regions: if SBD is below this, same-shape artifact
# clean/artifact is a per-beat decision (2026-08-13, user-specified): regardless of K_pulse, decided solely by whether each beat
#   resembles the medoid (representative beat) of its cluster. An incomplete beat mixed into a consistent cluster floats off the medoid -> artifact.
CLEAN_BY_MEDOID = True     # enable per-beat clean decision
CLEAN_SBD = 0.4            # a beat is clean if its SBD to the cluster medoid is at most this (~correlation >= 0.6)
CLEAN_MIN_N = 2            # if the cluster size is below this (a non-recurring singleton), all beats are artifacts
# --- singleton re-examination rule (decide whether to merge/re-split a size-1 cluster) ---
SINGLE_FACTOR = 1.5       # tolerance multiplier on the receiving cluster's internal spread (medoid SBD p90) (relative comparison)
SINGLE_FLOOR = 0.12       # lower bound on the merge-tolerance SBD (treat as same shape within this range even if the receiver is small)


# ============================================================================= IO
def find_file(series_id):
    """Find the CSV from the series ID (= the group id at the start of the filename, e.g. "260")."""
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{series_id}_*.csv")))
    if not hits:
        raise FileNotFoundError(f"series {series_id} not found in {DATA_DIR}")
    return hits[0]


def domain_series_ids():
    """List of series IDs (filename-leading group id) belonging to this SUBSET (ascending numeric order)."""
    files = glob.glob(os.path.join(DATA_DIR, f"*_{SUBSET}_*.csv"))
    return sorted((os.path.basename(f).split("_")[0] for f in files), key=int)


def parse_meta(path):
    """Get train_end (the length of the leading normal region) from the TSB-AD-U filename
    <gid>_<SUBSET>_id_<n>_<domain>_tr_<train_end>_1st_<x>.csv."""
    base = os.path.basename(path).replace(".csv", "")
    try:
        return int(base.split("_")[-3])
    except ValueError:
        return 0


def _acf_at(x, lag):
    """Autocorrelation after linear detrending (lag lag). Used to judge the significance of a daily period."""
    if lag <= 0 or lag >= len(x) - 1:
        return 0.0
    t = np.arange(len(x), dtype=float)
    z = x - np.polyval(np.polyfit(t, x, 1), t)
    z = z - z.mean()
    denom = float((z * z).sum()) + 1e-12
    return float((z[:-lag] * z[lag:]).sum() / denom)


def choose_period(xt):
    """Period decision for TSB-AD-U (IOPS) + aperiodic-fallback judgment. Returns (period, force_fixed).
    TSB-AD-U CSVs have no timestamp column (Data,Label only), so NAB's daily period
    (daily_period) is unavailable. The period is decided by autocorrelation (dominant_period) only.
    Aperiodic fallback (force_fixed=True):
      - tiny period (<MIN_PERIOD): a noise short-lag misdetection.
      - weak autocorrelation at the period (<0.20): effectively aperiodic (gid 265/272, etc.).
      - segment count below MIN_SEGS_ND (too few, references run out) / above MAX_SEGS (pseudo short period, windows too fine).
    There is no daily-priority exception (because there is no daily_period)."""
    if PERIOD:
        return PERIOD, False
    period = dominant_period(xt)
    est_segs = len(xt) / max(period, 1)
    force_fixed = (period < MIN_PERIOD) or (
        _acf_at(xt, period) < 0.20 or est_segs < MIN_SEGS_ND or est_segs > MAX_SEGS)
    return period, force_fixed


def load_series(path):
    """Read the CSV (header Data,Label) and return min-max (train-region statistics) normalized values, Label, and train_end
    (follows the TSB-AD-U preprocessing in CoAD toolkit/utils.py)."""
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




abp_series_ids = domain_series_ids   # backward-compatibility alias for cluster_diag, etc.


# ============================================================================= segmentation
def dominant_period(x):
    """Estimate the dominant period (samples per period) via autocorrelation.
    Among the autocorrelation peaks after the zero crossing, take the highest-correlation lag as the fundamental period
    (so a half-period small peak is not mistaken for the fundamental).
    A2 has trend-dominated series where raw autocorrelation buries the period, so for detection take autocorrelation on a
      **linearly detrended copy** (the signal itself stays min-max only, unchanged)."""
    t = np.arange(len(x), dtype=float)
    a, b = np.polyfit(t, x, 1)
    z = x - (a * t + b)                     # detection-only: a copy with the linear trend removed
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
    Lets low-amplitude breaths be peak-detected with the same prominence as high-amplitude ones."""
    win = max(5, int(win) | 1)
    lo = minimum_filter1d(xs, win, mode="nearest")
    hi = maximum_filter1d(xs, win, mode="nearest")
    return (xs - lo) / (hi - lo + 1e-9)


def _pulse_phase(x, period):
    """Estimate the rising-edge phase of the square-pulse train. ON = x > min + ON_THRESH_FRAC*(max-min).
    Return the modal phase (mod period) of the rising (0->1) positions. 0 if there are no edges."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return 0
    on = (x > lo + ON_THRESH_FRAC * (hi - lo)).astype(int)
    rises = np.where(np.diff(on) == 1)[0] + 1
    if len(rises) == 0:
        return 0
    phases = rises % period
    # Circular mode (period is small, so a simple histogram mode is enough)
    return int(np.bincount(phases, minlength=period).argmax())


def grid_segments(x, period):
    """SED: segment the whole range with no gaps on a period-wide fixed grid aligned to the pulse phase.
    Each cell either "contains a pulse (= the same square-pulse shape)" or is "empty (flat)". In the clean decision,
    only pulse cells are clean (= K_pulse references), and empty cells fall to non-clean (= artifact)."""
    n = len(x)
    ph = _pulse_phase(x, period)
    bounds = sorted(set([0] + list(range(ph, n, period)) + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    lo_len, hi_len = int(period * LEN_LO_FRAC), int(period * LEN_HI_FRAC)
    ceiling = x.max() - CLIP_CEIL_FRAC * (x.max() - x.min())
    clean = [bool(lo_len <= e - s <= hi_len) and is_breath_shape(x[s:e])
             and not is_clipped(x[s:e + 1], ceiling)
             for s, e in spans]
    return spans, clean, np.array([], dtype=int)


def excursion_segments(x, period):
    """Old SED segmentation (event/edge-based). Detect each excursion by threshold crossings about baseline 0.5
    (UP pulse v>up_thr / DOWN pulse v<dn_thr) and make **1 excursion = 1 segment**.
    Place cuts on the baseline between adjacent pulses (margin inside the edge) so a pulse is not bisected.
    The baseline left between pulses is a "flat" segment, covering the whole range with no gaps (gaps=0).

    Because it does not depend on a fixed period, "1 pulse = 1 segment" stays consistent even with the test irregular gaps (35/64/99...).

    Returns (spans, clean, peaks):
      spans : [(s,e), ...] full coverage (e exclusive)
      clean : pulse excursion segment = True (cluster reference candidate), flat segment = False (-> artifact)
      peaks : empty (for compatibility)
    """
    n = len(x)
    if n < 3:
        return [(0, n)], [False], np.array([], dtype=int)
    base = float(np.median(x))
    hip = float(np.percentile(x, 98))
    lop = float(np.percentile(x, 2))
    up_thr = base + ON_THRESH_FRAC * (hip - base)
    dn_thr = base - ON_THRESH_FRAC * (base - lop)
    state = np.where(x > up_thr, 1, np.where(x < dn_thr, -1, 0))
    nz = (state != 0).astype(int)
    d = np.diff(np.concatenate([[0], nz, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]            # the [start, end) excursion runs
    pulses = []
    for a, b in zip(starts, ends):
        if b - a < MIN_PULSE_W:            # a tiny deviation (noise) is treated as baseline
            continue
        sign = 1 if state[a:b].sum() >= 0 else -1
        pulses.append((int(a), int(b), sign))
    if not pulses:                         # no excursions at all -> the whole range is one flat segment
        return [(0, n)], [False], np.array([], dtype=int)
    margin = max(2, int(PULSE_MARGIN_FRAC * (period or 67)))
    spans, clean, cursor = [], [], 0
    for i, (a, b, sign) in enumerate(pulses):
        gl = a - (pulses[i - 1][1] if i > 0 else 0)          # left gap
        gr = (pulses[i + 1][0] if i + 1 < len(pulses) else n) - b  # right gap
        s = a - min(margin, max(0, gl // 2))                 # the shoulder is capped at half the adjacent gap (prevents overlap)
        e = b + min(margin, max(0, gr // 2))
        if s > cursor:                                       # baseline left before it = flat segment
            spans.append((cursor, s)); clean.append(False)
        spans.append((s, e)); clean.append(True)             # pulse excursion segment
        cursor = e
    if cursor < n:                                           # trailing baseline
        spans.append((cursor, n)); clean.append(False)
    return spans, clean, np.array([], dtype=int)


def cycle_onset_segments(x, period):
    """SED 1 cycle = 1 beat (onset anchor, 2026-10-08). Treat from a pulse rising edge (onset) to the next pulse onset
    as one beat = [pulse + the baseline right after it]. Do not make a baseline-only standalone segment (only the leading intro baseline is an exception).
    -> clean beats are contiguous in the train normal region, and many tbposwl (2-beat prev->cur) reference pairs are formed.
    UP/DOWN pulses are anti-correlated under z-norm+SBD so K_pulse=2 (both are normal repeating beats). Dropout/irregular gaps/
    amplitude anomalies are picked up by per-beat medoid clean (a dropout beat has a longer onset interval, so its shape deviates from the medoid, clean=0).

    Returns (spans, clean, peaks): spans full coverage (e exclusive), clean = whether it is "a beat containing a pulse = a reference candidate".
    """
    n = len(x)
    if n < 3:
        return [(0, n)], [False], np.array([], dtype=int)
    base = float(np.median(x))
    hip = float(np.percentile(x, 98))
    lop = float(np.percentile(x, 2))
    up_thr = base + ON_THRESH_FRAC * (hip - base)
    dn_thr = base - ON_THRESH_FRAC * (base - lop)
    state = np.where(x > up_thr, 1, np.where(x < dn_thr, -1, 0))
    nz = (state != 0).astype(int)
    d = np.diff(np.concatenate([[0], nz, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    onsets = [int(a) for a, b in zip(starts, ends) if b - a >= MIN_PULSE_W]  # rising edges of valid pulses
    if not onsets:                              # no pulses at all -> the whole range is one flat segment
        return [(0, n)], [False], np.array([], dtype=int)
    bounds = sorted(set([0] + onsets + [n]))    # cut at each onset (onset -> next onset is one beat)
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    # A pulse-containing beat (s is an onset) is a reference candidate True. Only the leading intro baseline [0, onset0) is False.
    onset_set = set(onsets)
    clean = [bool(s in onset_set) for (s, e) in spans]
    return spans, clean, np.array([], dtype=int)


def segment_cycles(x, period):
    """Segment the whole signal one unit at a time with no gaps.
    When ONSET_CYCLE (current SED), delegate to the 1-cycle=1-beat (onset anchor) segmentation.
    When EVENT_SEGMENT (old SED), delegate to the excursion (deviation from baseline) based segmentation.
    When GRID_ANCHOR (old SED), delegate to the pulse-phase-aligned fixed-grid segmentation.
    Otherwise, foot (valley) to the next foot (one breath) segmentation.

    Returns:
      spans  : [(s,e), ...] units covering the whole range (including partial units at the ends)
      clean  : whether each span "looks like a basic waveform (usable to define a cluster)" (bool list)
      peaks  : the detected peaks (empty for grid)
    """
    if ONSET_CYCLE:
        return cycle_onset_segments(x, period)
    if EVENT_SEGMENT:
        return excursion_segments(x, period)
    if GRID_ANCHOR:
        return grid_segments(x, period)
    n = len(x)
    w = max(5, int(period * SMOOTH_FRAC) | 1)
    xs = savgol_filter(x, window_length=w, polyorder=2)
    norm = _local_normalize(xs, win=int(period * LOCALNORM_FRAC))
    peaks, _ = find_peaks(norm, distance=max(3, int(period * PEAK_DIST_FRAC)),
                          prominence=PEAK_PROM)
    if len(peaks) < 3:
        return [], [], peaks
    # The minimum point (valley) between consecutive peaks is the cut foot
    feet = [peaks[i] + int(np.argmin(xs[peaks[i]:peaks[i + 1]]))
            for i in range(len(peaks) - 1)]
    # Add the signal ends as boundaries too and cover [0, n] with no gaps
    bounds = sorted(set([0] + feet + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    lo_len, hi_len = int(period * LEN_LO_FRAC), int(period * LEN_HI_FRAC)
    ceiling = x.max() - CLIP_CEIL_FRAC * (x.max() - x.min())   # the near-ceiling threshold for the saturation decision
    clean = [bool(lo_len <= e - s <= hi_len) and is_breath_shape(x[s:e])
             and not is_clipped(x[s:e + 1], ceiling)
             for s, e in spans]
    return spans, clean, peaks


def fixed_windows(n, win):
    """For aperiodic (trend-dominated) series: segment the whole range [0,n) into fixed-length win with no gaps.
    Returns the same (spans, clean, peaks) form as segment_cycles. clean is all True (no shape filter is applied,
    because the sinusoid assumption does not hold). Exclusion of label anomalies is done by the caller."""
    win = max(20, int(win))
    bounds = list(range(0, n, win))
    if not bounds or bounds[-1] != n:
        bounds.append(n)
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    return spans, [True] * len(spans), np.array([], dtype=int)


def is_clipped(seg, ceiling):
    """Whether this is a saturated (measurement-range ceiling-clipped) beat. Treat as saturated (non-clean) if the segment peak is
    near the global ceiling and there is a long contiguous same-value run (near-peak run) near the top.
    A normal breath does not reach the ceiling and its top is rounded, so its run is short."""
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
    """Whether the shape looks like one cycle (rejects saturation trapezoids and multi-peak noise fragments).
    Features are computed on the smoothed shape so normal cycles in noise-superimposed series are not wrongly removed.
    A2 is a noisy single-peak sinusoid; with weak smoothing (win11/prom0.2) the small noise peaks swell to
      npk=5-18 and wipe out normal cycles. With win31/prom0.4 a single peak is counted as npk~1."""
    if len(seg) < 5:
        return False
    m = znorm(resample(seg))
    ms = savgol_filter(m, SHAPE_SMOOTH_WIN, 2)
    rng = ms.max() - ms.min() + 1e-9
    flattop = float((ms > ms.max() - 0.1 * rng).mean())          # flat fraction near the top
    npk = len(find_peaks(ms, prominence=SHAPE_PK_PROM)[0])        # number of real shape peaks
    return flattop < FLATTOP_MAX and npk <= NOISE_PEAKS_MAX


# ============================================================================= shape distance (SBD) / features
def znorm(v):
    s = v.std()
    return (v - v.mean()) / (s + 1e-12)


FLAT_RANGE_EPS = 0.08     # SED: a region whose raw amplitude (max-min) is below this is treated as "baseline-only flat" and turned into
                          #   a zero vector by robust_znorm (assuming the min-max-normalized [0,1] scale). Prevents baseline micro-noise from being
                          #   MAD-amplified into a false shape and a flat gap being split into multiple artifact clusters.
                          #   A zero vector, by the degenerate rule of sbd, gives SBD=0 (identical) between flats, and SBD=1 for flat-pulse.


def robust_znorm(v):
    """Robust z-normalization that discards amplitude: (v - median)/(1.4826*MAD). For Stage1.
    Flat guard: a region whose raw amplitude is below FLAT_RANGE_EPS returns a zero vector (degenerates SED's baseline flat)."""
    v = np.asarray(v, dtype=float)
    if float(v.max() - v.min()) < FLAT_RANGE_EPS:
        return np.zeros_like(v)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)


def resample(v, n=None):
    n = n or RESAMPLE_LEN
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


_ZERO_NORM = 1e-9         # a norm below this is treated as degenerate (flat = all zeros after robust_znorm)


def sbd_full_pair(a, b):
    """Full-shift SBD = 1 - max(normalized cross-correlation). Phase-tolerant (compares by waveform type).
    Degenerate (flat = zero vector) handling: 0 (identical) if both are flat, 1 (different) if only one is flat.
      Without this, two flats get SBD=1 (max distance), each becomes a singleton, and K explodes."""
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _ZERO_NORM and nb < _ZERO_NORM:
        return 0.0
    if na < _ZERO_NORM or nb < _ZERO_NORM:
        return 1.0
    cc = np.correlate(a, b, mode="full")
    return float(1.0 - cc.max() / (na * nb))


def sbd_full_matrix(mat):
    """Full-shift SBD distance matrix (Stage1: broad grouping by waveform type). Degenerate (flat) follows the same rule as sbd_full_pair."""
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
    [3]n_extrema=#find_peaks(w,prom=0.10*range)+#find_peaks(-w,same),
    [4]skew=mean(z**3) (z=(w-mean)/std, ddof=0). Length<3 gives a zero vector."""
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
    """K selection by relative silhouette comparison + merge/absorb on a precomputed distance matrix (Stage1 SBD)."""
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
    """Stage2 acceptance guard: merge subcluster pairs that are "essentially the same in both shape and amplitude submode".
    Keep them only if (a) shape difference: medoid full-SBD > STAGE2_SBD, or
    (b) a real amplitude submode: range relative difference > STAGE2_AMP_RANGE and
       (activity relative difference > STAGE2_AMP_ACT or curv relative difference > STAGE2_AMP_CURV).
    -> Merges false splits from crossing the n_extrema threshold or from cutting a continuous gradient where only range moves."""
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
    # When cycles are too few (<4), Stage2 covariance/whiten cannot be formed (A2 long-period, few-cycle series).
    # Treat as a single sub-mode and return K=1 (a pure sine is a valid reference even with few samples).
    n = len(beats_raw)
    if n < 4:
        info = {"K_s1": 1, "K_sub": {0: 1}, "K": 1, "stage1_sils": {},
                "K_pulse": 1, "K_artifact": 0}
        return np.zeros(n, dtype=int), 1, info, None
    # ---- Stage 1: shape (ignore amplitude) ----
    D1 = sbd_full_matrix(shapes_robust)
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
        k_sub[int(c1)] = len(set(lab2))
        for sub in sorted(set(lab2)):
            final[idx[lab2 == sub]] = nxt
            nxt += 1
    final = relabel(final)
    info = {"K_s1": int(info1["K"]), "K_sub": k_sub, "K": int(len(set(final))),
            "stage1_sils": info1.get("sils", {})}
    return final, len(set(final)), info, D1


def cluster_artifacts(shapes):
    """Group non-clean spans by robust z-norm shape SBD complete linkage
    (SBD < ARTIFACT_TAU means same shape). Unusual ones form their own cluster even at n=1."""
    n = len(shapes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    D = sbd_full_matrix(shapes)
    Z = linkage(D[np.triu_indices(n, 1)], method="complete")
    return relabel(fcluster(Z, t=ARTIFACT_TAU, criterion="distance") - 1)



# ============================================================================= singleton re-examination rule
def _medoid_spread(shapes_rz, idx):
    """Medoid shape of cluster idx and its internal spread (90th percentile of SBD to the medoid).
    For large clusters, evenly subsample to <=80 items to limit O(n^2) SBD (speedup)."""
    idx = np.asarray(idx)
    if len(idx) == 1:
        return shapes_rz[idx[0]], 0.0
    sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
    D = sbd_full_matrix(shapes_rz[sub])
    mi = int(D.sum(1).argmin())
    return shapes_rz[sub[mi]], float(np.percentile(D[mi], 90))


def _nearest_cluster(shape, labels, shapes_rz, exclude):
    """Return the cluster closest to shape (other than exclude): (SBD, cluster, accept threshold).
    accept threshold = max(SINGLE_FLOOR, SINGLE_FACTOR * that cluster's internal spread) <- relative comparison."""
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
    """Re-split span [s,e] (a long under-segmented unit) at its internal troughs and return the sub-spans."""
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
    # Keep full coverage: do not drop too-short fragments; merge them into the next fragment so [s,e] is tiled with no gaps.
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
    """*singleton/small-cluster re-examination rule (added 2026-06-25, generalizing a user request):
    (a) **same-shape recovery**: if an artifact cluster's (label >= k_orig) medoid is within the nearest pulse
        cluster's internal spread (medoid SBD p90) * SINGLE_FACTOR, merge that whole cluster into pulse
        (recover normal cycles that noise/distortion marked non-clean back into c0 etc. Relative comparison).
    (b) **re-split**: a size-1 artifact with length ~2 periods or more (under-segmented multi-cycle) is re-split
        at its internal troughs, and each piece merged into the nearest cluster (standalone artifact if none fits).
    (c) **keep standalone**: genuinely different shapes (partial/double/unusual) are left as is.
    Returns: the updated spans, labels. (speedup: cluster medoids are computed once per iteration)"""
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
        # (a) same-shape recovery of a whole artifact cluster into pulse
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
    pulse clusters = original pulse labels (< k_orig) that survived. New labels from split are artifacts."""
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
    # Sort spans by start position (readability: PDF/CSV in chronological order)
    sidx = sorted(range(len(spans)), key=lambda i: spans[i][0])
    spans = [spans[i] for i in sidx]
    labels = [labels[i] for i in sidx]
    return spans, labels, len(pulse_ids), len(art_ids), None
# ============================================================================= driver
def process(series_id, save=True):
    path = find_file(series_id)
    x, label, train_end = load_series(path)         # min-max(train stats) normalized, Label, train_end
    xt = x[:train_end] if train_end > 200 else x    # learn the base waveform on the normal training span
    anom_spans = label_spans(label)                 # contiguous spans where Label==1 (red band display / eval reference)

    period, short_period = choose_period(xt)
    spans, clean, peaks = segment_cycles(xt, period)
    clean_idx = [i for i, c in enumerate(clean) if c]
    # *Yahoo (semi-supervised): exclude cycles that overlap labeled anomalies from the base-waveform training pool.
    #   Excluded cycles are treated as non-clean and go to an artifact cluster (keep the few train anomalies out of the normal reference).
    n_pool0 = len(clean_idx)
    clean_idx = [i for i in clean_idx
                 if label[spans[i][0]:min(spans[i][1] + 1, len(label))].sum() == 0]
    n_excl = n_pool0 - len(clean_idx)
    # *non-periodic fallback: if periodic segmentation yields 0 clean cycles (= trend+noise dominated, no sine),
    #   or a spurious tiny period was detected (short_period), re-split with fixed-length windows (sliding fixed window).
    use_fixed = len(clean_idx) < 1 or short_period
    if use_fixed:
        spans, clean, peaks = fixed_windows(len(xt), FIXED_WIN)
        clean_idx = [i for i, (s, e) in enumerate(spans)
                     if label[s:min(e + 1, len(label))].sum() == 0]   # non-anomalous windows become the normal reference
    if len(clean_idx) < 1:
        print(f"[{series_id}] segmentation failed (period={period}, spans={len(spans)}) "
              f"-- cannot segment")
        return None

    shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    beats_raw = [xt[s:e + 1] for s, e in spans]

    # Two-stage clustering on clean units only (not misled by artifacts)
    ci = clean_idx
    labels_c, K, info, _ = cluster_two_stage([beats_raw[i] for i in ci], shapes_rz[ci])

    labels_plot = np.full(len(spans), -1, dtype=int)
    for k, i in enumerate(ci):
        labels_plot[i] = int(labels_c[k])

    # non-clean spans = artifact clusters (numbered after the pulse clusters)
    nc = [i for i in range(len(spans)) if labels_plot[i] < 0]
    n_art = 0
    if nc:
        art = cluster_artifacts(shapes_rz[nc])
        for j, i in enumerate(nc):
            labels_plot[i] = K + int(art[j])
        n_art = int(art.max()) + 1

    # *singleton re-examination rule: merge/re-split size-1/small clusters (relative comparison). Keep pulse-derived cycles.
    clean_set = set(clean_idx)
    spans, labels_plot, K, n_art, clean_set = _apply_reconcile(
        xt, spans, list(labels_plot), period, K, clean_set)
    labels_plot = np.array(labels_plot)
    K_total = K + n_art
    info["K_pulse"], info["K_artifact"] = K, n_art

    # *semi-supervised guarantee: train spans overlapping labeled anomalies must never enter the normal-reference
    #   medoid, even if reconcile moved them back to pulse (normal reference). Move such spans to a dedicated artifact cluster (prevent known-anomaly contamination).
    anom_train = [i for i, (s, e) in enumerate(spans)
                  if labels_plot[i] < K and label[s:min(e + 1, len(label))].sum() > 0]
    if anom_train:
        for i in anom_train:
            labels_plot[i] = K_total          # move all to a new artifact id
        n_art += 1
        K_total = K + n_art
        info["K_artifact"] = n_art

    # For the overlay plot use std-based z-norm (MAD diverges on saturated units and is hard to read)
    shapes_plot = np.array([znorm(resample(xt[s:e + 1])) for s, e in spans])

    # ---- Apply the learned segmentation to the test span (after train_end) too, assigning each unit to the nearest train cluster medoid ----
    #      (the base waveform is learned on train only. test is not re-clustered but classified by nearest medoid = same structure as VLM-AD).
    #      *assign every test segment to the nearest cluster (never -1, clean or non-clean).
    #        candidates are all learned clusters (pulse 0..K-1 + artifact K..K_total-1); take the nearest medoid.
    final_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    medoids = {}
    for c in range(K_total):
        idx = np.where(labels_plot == c)[0]
        if len(idx):
            sub = final_rz[idx]
            medoids[c] = sub[sbd_full_matrix(sub).sum(1).argmin()] if len(idx) > 1 else sub[0]
    # *per-beat clean decision: a beat is clean if its SBD to its cluster medoid <= CLEAN_SBD and the cluster size >= CLEAN_MIN_N.
    clu_size = {c: int(np.sum(np.asarray(labels_plot) == c)) for c in range(K_total)}
    def _clean_by_medoid(shape, lab, dev=None, seg_len=None):
        # *SED flat guard: a flat segment whose raw amplitude (max-min) is below FLAT_DEV_MIN (baseline only, no excursion)
        #   is not a pulse, so do not mark it clean. Prevents robust_znorm from amplifying noise on a flat segment and creating a false medoid match.
        if dev is not None and dev < FLAT_DEV_MIN:
            return False
        # *SED length guard (ONSET_CYCLE): one beat = one cycle (onset -> next onset), so a beat whose length falls outside the
        #   valid period range [LEN_LO_FRAC, LEN_HI_FRAC]*period is a missing/extra pulse (= irregular gap anomaly).
        #   If the pulse sits at the beat head, resample erases the length and SBD would pass, so reject by length directly.
        if ONSET_CYCLE and seg_len is not None and period and period > 0:
            if not (period * LEN_LO_FRAC <= seg_len <= period * LEN_HI_FRAC):
                return False
        if not CLEAN_BY_MEDOID:
            return lab < K
        return clu_size.get(lab, 0) >= CLEAN_MIN_N and \
            float(sbd_full_pair(medoids[lab], shape)) <= CLEAN_SBD
    def _seg_dev(arr, s, e):
        seg = arr[s:min(e + 1, len(arr))]
        return float(seg.max() - seg.min()) if len(seg) else 0.0
    clean_train = [bool(_clean_by_medoid(final_rz[i], int(labels_plot[i]), _seg_dev(xt, s, e), e - s))
                   for i, (s, e) in enumerate(spans)]
    test_spans, test_labels, clean_test = [], [], []
    x_disp = xt
    if 200 < train_end < len(x) and medoids:
        xtt = x[train_end:].astype(float)
        ts, _tc, _ = (fixed_windows(len(xtt), FIXED_WIN) if use_fixed
                      else segment_cycles(xtt, period))
        # *test coverage guarantee: strong-trend series fail trough detection on the test side, making ts empty/sparse (test
        #   uncovered, so anomalies are excluded). In that case use the train period as a fixed pitch for fixed-window splitting to always cover the whole test span.
        covered = sum(e - s for s, e in ts)
        if not ts or covered < 0.5 * len(xtt):
            ts, _tc, _ = fixed_windows(len(xtt), period if period and period > 10 else FIXED_WIN)
        for (s, e) in ts:
            shp = resample(robust_znorm(xtt[s:e + 1]))
            lab = min(medoids, key=lambda k: float(sbd_full_pair(medoids[k], shp)))
            test_spans.append((int(s + train_end), int(e + train_end)))
            test_labels.append(int(lab))
            clean_test.append(bool(_clean_by_medoid(shp, int(lab), _seg_dev(xtt, s, e), e - s)))
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
    """PDF output: wrap ROW_STEPS steps per row, draw a vertical dashed line at each split, and
    background-highlight each span with its cluster color (full coverage). Last page: cluster overlay + mean.
    If test_spans/test_labels are passed, also draw the span after train_end (test span), colored by the
    train cluster medoid it was assigned to. train/test boundary = solid blue; Label anomaly spans (anom_spans, may be multiple) = red band + red line."""
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
        # For long series, cap the max page count to avoid PDF bloat. But always show up to the last Label anomaly span.
        max_pages = 8
        if anom_max:
            max_pages = max(max_pages, int(np.ceil((anom_max + ROW_STEPS) / ROW_STEPS / ROWS_PER_PAGE)) + 1)
        max_pages = min(max_pages, 20)
        rows_cap = max_pages * ROWS_PER_PAGE
        # *Unify the y-axis scale across all rows (no per-row auto-scaling, feedback 2026-08-09).
        #   Fix ylim from the robust range (1~99 percentile) of all displayed data plus padding.
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
                # faint red band for Label anomaly spans (bottom background layer, multiple spans)
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
                ax.set_ylim(ylo, yhi)                  # unified y-axis scale across all rows
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
# Keys that <id>.py PARAMS may override (does not change the clustering structure or metrics)
TUNABLE = ("PERIOD", "RESAMPLE_LEN", "PEAK_DIST_FRAC", "PEAK_PROM", "LOCALNORM_FRAC",
           "SMOOTH_FRAC", "LEN_LO_FRAC", "LEN_HI_FRAC", "FLATTOP_MAX", "NOISE_PEAKS_MAX",
           "CLIP_CEIL_FRAC", "CLIP_RUN_MIN",
           "KMAX_S1", "KMAX_S2", "MAXLAG_FRAC", "MERGE_TAU", "SIL_MIN", "MIN_PATTERN",
           "MAHAL_RIDGE", "STAGE2_SBD", "STAGE2_AMP_RANGE", "STAGE2_AMP_ACT",
           "STAGE2_AMP_CURV", "ARTIFACT_TAU",
           "GRID_ANCHOR", "ON_THRESH_FRAC", "MIN_PERIOD", "MIN_SEGS_ND",
           "MAX_SEGS", "FIXED_WIN",
           "ONSET_CYCLE", "EVENT_SEGMENT", "PULSE_MARGIN_FRAC", "MIN_PULSE_W", "FLAT_DEV_MIN",
           "FLAT_RANGE_EPS")


def run(series_id, params=None, save=True):
    """Public API to process a single series. Called from `<id>.py`.
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
