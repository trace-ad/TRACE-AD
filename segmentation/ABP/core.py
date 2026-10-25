#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABP basic-waveform segmentation + clustering engine (core)
=========================================================

A shared engine that splits a time series (Atrial Blood Pressure) into "basic waveform patterns"
and clusters them by shape. Each series is run by a thin `<id>.py` (which just calls core.run).

Design
------
- ABP is a pulse wave (quasi-periodic signal). A basic waveform = one beat (one pulse). Make each span
  "one whole beat" (foot trough to the next foot trough) as long as possible to avoid over-splitting. Cover the whole signal with no gaps.
- z-normalize each beat + resample to a common length, then hierarchically cluster with a **shift-limited SBD**
  (which preserves phase = systolic peak position differences).
- K is chosen by relative comparison (silhouette); very similar clusters are merged and non-recurring one-off outlier beats are absorbed.
  -> clustering helps "evaluate the split" (clean clusters = a good split; a junk
    cluster = a sign of mis-splitting). Also used for per-cluster downstream learning.

Usage
-----
  python 024.py            # process series 024 -> output/024.pdf etc.
  python core.py --all     # process all 42 ABP series together (batch)
  python core.py --series 024 245

Series-specific tuning overrides core's default values via the PARAMS in `<id>.py`.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- paths / domain
# the domain this folder handles (a key of domains_dict.json). For another domain, copy the folder
# and just change this. domains_dict.json is shared in the parent (directly under Agent_Split_Waveform).
DOMAIN = "Atrial Blood Pressure (ABP)"
DATA_DIR = "./data/UCR_Anomaly_FullData"
_HERE = os.path.dirname(os.path.abspath(__file__))
DOMAINS_JSON = os.path.join(os.path.dirname(_HERE), "domains_dict.json")   # shared json in the parent
OUT_DIR = os.path.join(_HERE, "output")                                    # output within the domain

ROW_STEPS = 2000          # time steps per PDF row
ROWS_PER_PAGE = 12        # rows per PDF page
RESAMPLE_LEN = 256        # resample each beat to this length for shape comparison
KMAX_S1 = 3               # upper bound for K_s1 candidates in Stage1 (shape type)
KMAX_S2 = 3               # upper bound for K_sub candidates in Stage2 (amplitude/sharpness subdivision)
MAXLAG_FRAC = 0.10        # lag bound of the shift-limited SBD used for nearest-neighbor assignment of non-clean beats
MERGE_TAU = 0.13          # if SBD between medoids is below this, treat as "same shape" and merge (0.10->0.13)
                          # merges locally slightly-different pulses such as 243 (same-shape dicrotic, SBD~0.109) (user note)
SIL_MIN = 0.12            # if the best silhouette is below this, do not split (K=1)
MIN_PATTERN = 8           # clusters smaller than this are one-off outliers, not "recurring patterns" -> merge
                          # raised 4->8 to absorb sparse spurious splits (091: n=6) (user note 2026-06-25).
MAHAL_RIDGE = 1e-3        # ridge on the Stage2 covariance (stabilizes Mahalanobis whitening)
# Stage2 acceptance guard: adopt a subdivision only when "shapes clearly differ OR a real amplitude submode exists".
# user note (2026-06-25): the ABP K=2 series (089/091/197/244/245/246/247) were all spurious splits where c0/c1
# have the same shape (medoid SBD 0.011-0.107): (i) pulses differing only in amplitude (244-247, SBD~0.02),
# (ii) end-of-record pulse-pressure drift beats clustered at the tail (197/089, c1 concentrated in train tail 85k-100k, SBD~0.09),
# (iii) sparse cases (091, n=6). The pulse waveform is a single type (systolic peak + dicrotic notch); amplitude/tail drift are
# not a separate "basic waveform". So raise the STAGE2 thresholds to merge same-shape spurious splits (same criterion as other domains).
STAGE2_SBD = 0.12         # if full-SBD between sub-cluster medoids exceeds this, "shapes differ" (0.05->0.12)
STAGE2_AMP_RANGE = 0.50   # if the relative difference in amplitude range exceeds this, and the below holds, it is an "amplitude submode"
                          # 0.08->0.50: a pulse-pressure (amplitude) difference is not a separate basic waveform (tail drift / pulse-pressure regime)
STAGE2_AMP_ACT = 0.15     #   relative difference in activity exceeds this
STAGE2_AMP_CURV = 0.25    #   or relative difference in curv exceeds this
AMP_ASSIGN_W = 0.5        # weight on the amplitude difference (|delta range|/scale) when assigning clean test beats (only active when K_pulse>1)
ARTIFACT_TAU = 0.10       # non-clean region clustering: if SBD is below this, same-shape artifact
# clean/artifact is a per-beat decision (2026-08-13, user-specified): regardless of absolute criteria (flattop/deadflat/is_beat_shape)
#   or K_pulse, it is decided solely by "whether each beat resembles its cluster medoid (the representative beat)". Even flat/square,
#   if that shape is the representative it is normal (clean). An incomplete beat mixed into a consistent cluster (truncated / second-cycle bleed) deviates from the medoid -> artifact.
#   In a scattered cluster all beats are far from the medoid -> all beats artifact (correct behavior). SBD is length-normalized + max over all shifts.
CLEAN_BY_MEDOID = True     # enable the per-beat clean decision
CLEAN_SBD = 0.4            # a beat is clean if its SBD to the cluster medoid is at most this (~correlation >= 0.6)
CLEAN_MIN_N = 2            # if the cluster size is below this (a non-repeating singleton), all its beats are artifact
MERGE_ARTIFACTS = False   # True: merge non-clean regions into a single artifact cluster regardless of shape (K_artifact=1).
                          #   prevents dropouts/saturation/flats from fragmenting into many singletons in long recordings.
ARTIFACT_RAW_FEATURES = False  # True: cluster artifacts by raw features (level/range/trend) with ward+silhouette.
ARTIFACT_FEAT_TAU = 1.0   #   distance threshold for the raw-feature ward (standardized space). Below it, treated as one cluster (same type).
ARTIFACT_LEVEL_FEAT = False  # True: add "whether it drops to 0 (floor)" and range to the artifact features.
#   separates a high-floating flat plateau (min~5700, small range) from the full [0->plateau->0] body (min=0). For 244-247
#   (242/243 have no high flat plateau; adding it would over-split the body on trivial differences, so keep False).
_ART_GREF = None  # (gmin, grng): train/test-shared amplitude reference. Set in process() from the whole series (used for the floor decision).
# --- singleton re-examination rule (decide whether to merge/re-split size-1 clusters) ---
SINGLE_FACTOR = 1.5       # tolerance multiplier against the target cluster's internal spread (medoid SBD p90) (relative comparison)
SINGLE_FLOOR = 0.12       # lower bound on the merge-allowed SBD (even if the target is small, within this range treat as same shape)
FLATTOP_MAX = 0.20        # if the near-top flat fraction of the shape is at least this = saturated/trapezoid/plateau beat (excluded)
DEADFLAT_MAX = 0.25       # if a dead-flat (calibration plateau) run in the raw waveform exceeds this fraction = non-clean
                          # 0.08->0.25: fixes misclassifying a normal pulse's diastolic baseline (dead ratio ~0.11) as non-clean
                          #   and sending it to artifact (e.g. 242 at 144500). A square-wave plateau (0.48+) is still caught.
EDGE_MISMATCH_MAX = 0.40  # if the difference between the first-8% mean and last-8% mean (/range) exceeds this = not foot-to-foot
                          # (a calibration step beat is high at the start and low at the end; a normal pulse is diastolic and similar at both ends)
PEAKPOS_MAX = 0.45        # if the systolic peak position (0..1) exceeds this = phase-shifted fragment (excluded)
NOISE_PEAKS_MAX = 5       # if the number of peaks within the shape exceeds this = jagged noise beat (excluded)
# no size-based outlier absorption: a beat whose shape clearly differs is kept as its own cluster even if rare
# (a size-1 cluster is allowed). But segments that are "not a beat" (saturation/fragment) are excluded.


# ============================================================================= IO
def find_file(series_id):
    hits = sorted(glob.glob(os.path.join(DATA_DIR, f"{series_id}_*.txt")))
    if not hits:
        raise FileNotFoundError(f"series {series_id} not found in {DATA_DIR}")
    return hits[0]


def parse_meta(path):
    """Get the train length / anomaly span from the UCR file name."""
    base = os.path.basename(path).replace(".txt", "")
    nums = base.split("_")
    try:
        train_end, anom_s, anom_e = int(nums[-3]), int(nums[-2]), int(nums[-1])
    except ValueError:
        train_end, anom_s, anom_e = 0, 0, 0
    return train_end, anom_s, anom_e


def domain_series_ids():
    """List of series IDs belonging to this folder's DOMAIN."""
    d = json.load(open(DOMAINS_JSON))
    names = d[DOMAIN]
    return sorted(n.split("_")[0] for n in names)


abp_series_ids = domain_series_ids   # backward-compatibility alias


# ============================================================================= segmentation
def dominant_period(x):
    """Estimate the dominant period (samples per beat) via autocorrelation.

    To avoid picking up a half-period peak from the dicrotic notch etc., take the lag
    with the "highest correlation" among the autocorrelation peaks after the zero crossing as the fundamental period.
    """
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-12
    zero = np.where(ac < 0)[0]
    lo = zero[0] if len(zero) else 10
    hi = min(len(ac), lo + 3000)
    seg = ac[lo:hi]
    if len(seg) < 3:
        return 100
    peaks, props = find_peaks(seg, height=0.0)
    if len(peaks) == 0:
        return int(lo + np.argmax(seg))
    # the highest-correlation peak = the fundamental period (inherently higher than the half-period peak)
    best = peaks[int(np.argmax(props["peak_heights"]))]
    return int(lo + best)


def _local_normalize(xs, win):
    """Normalize to 0..1 with a local (min, max) to remove amplitude drift,
    so low-amplitude beats are detected as prominently as high-amplitude ones."""
    from scipy.ndimage import minimum_filter1d, maximum_filter1d
    win = max(5, int(win) | 1)
    lo = minimum_filter1d(xs, win, mode="nearest")
    hi = maximum_filter1d(xs, win, mode="nearest")
    return (xs - lo) / (hi - lo + 1e-9)


def segment_cycles(x, period):
    """Split the whole signal into one beat per foot(trough)-to-foot span, **with no gaps**.

    To avoid over-splitting, use distance=0.55*period as a lower bound and detect peaks on an
    amplitude-drift-robust locally-normalized signal to avoid misses (= under-splitting).

    Returns:
      spans  : [(s,e), ...] beats covering the whole range (including partial beats at the start/end)
      clean  : whether each span is "beat-like (usable for cluster definition)" (bool list)
      peaks  : the detected systolic peaks
    """
    n = len(x)
    w = max(5, int(period * 0.08) | 1)
    xs = savgol_filter(x, window_length=w, polyorder=2)
    norm = _local_normalize(xs, win=int(period * 1.5))     # amplitude-normalize with a ~1.5-period window
    peaks, _ = find_peaks(norm, distance=max(3, int(period * 0.55)), prominence=0.30)
    if len(peaks) < 3:
        return [], [], peaks
    # between consecutive systolic peaks, take the **end-diastolic trough (just before the next upstroke)** as the foot (cut point).
    # searching for the minimum in the latter 55% of the span (lo=a+0.45*(b-a) onward) avoids wrongly cutting mid-cycle at the
    # dicrotic notch right after systole (a deep trough in the first half; in fast-heart-rate spans it can be deeper than end-diastole)
    # (user note: three beats in the 197 tail 97250-97550 were cut at the dicrotic notch, 2026-06-25).
    feet = []
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        foot_full = a + int(np.argmin(xs[a:b]))
        # the span minimum fell in the first half (<40%) = suspected dicrotic notch. Only then re-search the latter 55% for
        # the end-diastolic trough. Normal beats (minimum already in the latter half) are left as is (limits impact on calibration boundaries).
        if (foot_full - a) < 0.40 * (b - a):
            lo = a + int(0.45 * (b - a))
            feet.append(lo + int(np.argmin(xs[lo:b])))
        else:
            feet.append(foot_full)
    # add the signal ends as boundaries too, forming foot-to-foot spans that cover [0, n] with no gaps
    bounds = sorted(set([0] + feet + [n]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
             if bounds[i + 1] > bounds[i]]
    # deterministically re-split square-wave/dropout regions: cut at the center of a sustained "0-level flat run",
    #   aligning to the full single cycle [0->plateau->0]. The 0-run center is always at the same position in train/test,
    #   so it removes foot-detection jitter (train=full period 250 / test=fragmented into half periods) and makes the two splits agree.
    #   does not fire on a normal pulse (no sustained flat at 0) (guaranteed by minrun).
    spans = _resegment_square(x, spans, period)
    spans = _split_long_pulse(x, spans, period)       # re-split an over-long pulse span of two fused beats at its trough (square waves excluded)
    # whether each span is "beat-like" (the criterion used for cluster definition). Spans that fail it (fragment/saturated trapezoid/
    # phase-shift/noise/partial beat) do not create a new cluster and are later assigned to the nearest cluster.
    lo_len, hi_len = int(period * 0.6), int(period * 1.6)
    clean = [bool(lo_len <= e - s <= hi_len) and is_beat_shape(x[s:e])
             for s, e in spans]
    return spans, clean, peaks


def _resegment_square(x, spans, period):
    """Deterministically re-split square-wave/dropout regions at the center of a sustained "0-level flat run".
    The 0-run center is determined by the signal alone and is identical in train/test, so it removes foot-detection variability and makes the two splits agree.
    Does not fire on a normal pulse (no sustained flat run at 0) (0-runs shorter than minrun are not used).
    Applied only to series known to have square-wave dropouts (ARTIFACT_RAW_FEATURES=True, 242-247).
    Not fired on other series (e.g. 089/090/197 with mid-level flats or pulse feet), since a threshold alone cannot separate them reliably."""
    if not ARTIFACT_RAW_FEATURES:
        return spans
    x = np.asarray(x, float); n = len(x)
    gmin = float(x.min()); gmax = float(x.max()); grng = float(np.ptp(x)) + 1e-9
    minrun = max(5, int(0.12 * period))

    def flat_runs(mask):
        """Return (center, length) of each run where mask is True (length >= minrun)."""
        out, i = [], 0
        while i < n:
            if mask[i]:
                j = i
                while j < n and mask[j]:
                    j += 1
                if j - i >= minrun:
                    out.append(((i + j) // 2, j - i))
                i = j
            else:
                i += 1
        return out

    low_runs = flat_runs(x < gmin + 0.10 * grng)      # sustained runs near the global minimum (= 0-drop)
    # firing guard: a square wave has a "sustained high plateau" (a normal pulse peak is instantaneous, with no plateau).
    #   the plateau is a "flat (small |diff|)" run that stays at mid-level or above (> gmin+40%) for at least minrun. Near gmax it would miss
    #   series (246/247) where the pulse spike (gmax) is higher than the plateau (6400), so judge by a mid-level threshold + flatness rather than gmax.
    hi_flat = np.abs(np.diff(x, prepend=x[0])) < 0.01 * grng
    high_runs = flat_runs((x > gmin + 0.40 * grng) & hi_flat)
    centers = [c for c, _ in low_runs]
    if len(centers) < 2 or len(high_runs) < 1:
        return spans                                  # sustained 0-level < 2 or no sustained plateau -> normal pulse etc., do not re-split
    # true-0-dropout test: fire only when the low-run level is below 25% of the high plateau (= drops nearly to 0).
    #   series like 089/197 where min = pulse foot (~4000, ~2/3 of the high plateau) with no "true 0-drop" are left unfired.
    lo_lv = float(np.median([x[c] for c, _ in low_runs]))
    hi_lv = float(np.median([x[c] for c, _ in high_runs]))
    if lo_lv > gmin + 0.25 * (hi_lv - gmin):
        return spans
    # group nearby 0-run centers into a "dropout event" region (linked when the gap < 2.5*period)
    events, cur = [], [centers[0]]
    for c in centers[1:]:
        if c - cur[-1] < 2.5 * period:
            cur.append(c)
        else:
            events.append(cur); cur = [c]
    events.append(cur)
    ev_ranges = [(ev[0], ev[-1]) for ev in events if len(ev) >= 2]
    cut_in_event = set(c for ev in events if len(ev) >= 2 for c in ev)

    def in_event(p):
        return any(lo <= p <= hi for lo, hi in ev_ranges)
    # outside the event region keep the original split boundaries; inside, cut only at 0-run centers
    bnds = set()
    for s, e in spans:
        if not in_event(s):
            bnds.add(s)
        if not in_event(e):
            bnds.add(e)
    allcuts = sorted(bnds | cut_in_event | {0, n})
    # merge short fragments (e.g. a 0-run remnant at a square-wave tail < 0.35*period) into the neighbor. Only cuts inside an event region apply
    #   (legitimate cuts in pulse spans are kept). If the gap to the previous cut is below min_len, drop that cut and absorb into the previous span.
    min_len = max(3, int(0.35 * period))              # threshold for merging short fragments within an event
    flat_len = max(3, int(0.70 * period))             # pure-flat fragments (0-run / plateau remnants) are merged up to one period
    kept = [allcuts[0]]
    for c in allcuts[1:-1]:
        seg = x[kept[-1]:c]
        gap = c - kept[-1]
        flat = len(seg) > 0 and float(np.ptp(seg)) < 0.30 * grng   # pure-flat (small range) = a remnant that is not a full single cycle
        # if it is a "short fragment within an event" or a "pure-flat sub-period fragment", drop the cut and merge (a pulse has a large range, flat=False).
        if (gap < min_len and in_event(c)) or (flat and gap < flat_len):
            continue
        kept.append(c)
    kept.append(allcuts[-1])
    return [(kept[i], kept[i + 1]) for i in range(len(kept) - 1) if kept[i + 1] > kept[i]]


def _split_long_pulse(x, spans, period):
    """Split a **pulse span** where several beats fused, at the deepest trough (foot) between systolic peaks.
    Corrects cases where foot detection misses a trough in the test region and two beats fuse into one segment
    (e.g. 242 test [282756-283180] len=424 = 2x~210 pulses -> split in two at the inter-peak trough).
    Judged by the **number of systolic peaks in the span**, not a length threshold (period=306 overestimates the local beat length ~210
    and a length threshold of 1.4*period would miss a two-beat fusion = 420). A single pulse has one peak -> not split.
    Square-wave/dropout regions (reaching the 0-baseline) are excluded since _resegment_square handles them."""
    if not ARTIFACT_RAW_FEATURES:
        return spans
    x = np.asarray(x, float)
    gmin = float(x.min()); grng = float(np.ptp(x)) + 1e-9
    dist = max(3, int(0.5 * period))
    out = []
    for s, e in spans:
        seg = x[s:e]
        # leave too-short (<0.7*period = at most a single pulse) or 0-dropping (square-wave body) untouched. Delegate the fusion test to the peak count
        #   (when local beats are short, even two pulses can total less than period, so a length threshold misses them; keep the lower bound low).
        if (e - s) < 0.7 * period or len(seg) < 3 or float(seg.min()) <= gmin + 0.20 * grng:
            out.append((s, e)); continue
        rng = float(np.ptp(seg)) + 1e-9
        pk, _ = find_peaks(seg, distance=dist, prominence=0.30 * rng)
        if len(pk) < 2:                               # one systolic peak = single pulse -> do not split
            out.append((s, e)); continue
        cuts = [s] + [s + pk[i] + int(np.argmin(seg[pk[i]:pk[i + 1]]))
                      for i in range(len(pk) - 1)] + [e]   # deepest trough between consecutive peaks = pulse foot
        cuts = sorted(set(cuts))
        out.extend((cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1))
    out.sort()
    return out


def is_beat_shape(seg):
    """Whether the shape is beat-like (rejects saturated trapezoids, phase-shift fragments, noise beats).
    Features are computed on the smoothed shape so normal beats in noisy series are not wrongly removed."""
    # non-clean if the raw waveform has a long run of "dead flat (a calibration plateau holding the DAC)".
    # a mid-level plateau (not near the top) is not caught by flattop, so detect it directly on the raw signal
    # (excludes a spurious cluster c1 that mixes beat+plateau at a calibration square-wave boundary. user note 244/245).
    rng_raw = float(seg.max() - seg.min()) + 1e-9
    dead = np.abs(np.diff(seg)) < 0.003 * rng_raw
    run = mx = 0
    for v in dead:
        run = run + 1 if v else 0
        mx = max(mx, run)
    if mx / max(1, len(seg)) > DEADFLAT_MAX:
        return False
    # foot-to-foot check: a normal pulse is diastolic and similar at both start and end. A span whose start/end means differ greatly is
    # a fragment cut mid-beat by a calibration step etc. (non-clean. the c1 step beats of 244/245, user note).
    k = max(2, len(seg) // 12)
    if abs(seg[:k].mean() - seg[-k:].mean()) / (seg.max() - seg.min() + 1e-9) > EDGE_MISMATCH_MAX:
        return False
    m = znorm(resample(seg))
    ms = savgol_filter(m, 11, 2)                         # smooth the shape (remove measurement noise)
    rng = ms.max() - ms.min() + 1e-9
    flattop = float((ms > ms.max() - 0.1 * rng).mean())  # near-top flat fraction
    peakpos = float(np.argmax(ms)) / len(ms)            # systolic peak position 0..1
    npk = len(find_peaks(ms, prominence=0.2)[0])        # number of true shape peaks (noise detection)
    return flattop < FLATTOP_MAX and peakpos < PEAKPOS_MAX and npk <= NOISE_PEAKS_MAX


# ============================================================================= shape distance (SBD)
def znorm(v):
    s = v.std()
    return (v - v.mean()) / (s + 1e-12)


def resample(v, n=RESAMPLE_LEN):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), v)


def ncc_sbd(a, b):
    """**Shift-limited** SBD of equal-length z-normalized vectors a, b = 1 - max(normalized cross-correlation).

    Ordinary SBD maximizes over the full shift range and so erases phase (systolic peak position) differences, but
    each beat is already phase-aligned at its foot origin, so the shift is limited to +/-MAXLAG_FRAC.
    This absorbs small foot-detection offsets while keeping a distance for "a different pattern with a different peak position"
    as genuinely different."""
    L = len(a)
    ml = max(1, int(L * MAXLAG_FRAC))
    cc = np.correlate(a, b, mode="full")           # length 2L-1, center index = L-1
    mid = L - 1
    cc = cc[mid - ml: mid + ml + 1]                 # only lags +/-ml
    cc /= (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(1.0 - cc.max())


def sbd_matrix(mat):
    n = len(mat)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = ncc_sbd(mat[i], mat[j])
            D[i, j] = D[j, i] = d
    return D


def robust_znorm(v):
    """Robust z-normalization that discards amplitude: (v - median) / (1.4826*MAD). For Stage1."""
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-12
    return (v - med) / (1.4826 * mad)


def sbd_full_matrix(mat):
    """Full-shift-range SBD distance matrix (Stage1: coarse grouping by waveform type; phase-tolerant)."""
    n = len(mat)
    norms = np.linalg.norm(mat, axis=1) + 1e-12
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cc = np.correlate(mat[i], mat[j], mode="full")
            d = 1.0 - cc.max() / (norms[i] * norms[j])
            D[i, j] = D[j, i] = float(d)
    return D


def d3_features(w):
    """Stage2 D3 (5 features, RAW values). The span w is raw values (no smoothing, end inclusive).
    Restores amplitude/scale + fine-shape statistics that Stage1 discarded via robust z-norm."""
    if len(w) < 3:
        return np.zeros(5)
    rng = float(w.max() - w.min())                       # 0: raw amplitude (peak-to-peak)
    activity = float(np.mean(np.abs(np.diff(w))))        # 1: slope activity
    curvature = float(np.max(np.abs(np.diff(w, n=2))))   # 2: sharpness (curvature)
    if rng <= 1e-9:
        n_ext, skew = 0, 0.0
    else:                                                # 3: count of prominent peaks + troughs (prominence relative within the span)
        n_ext = (len(find_peaks(w, prominence=0.10 * rng)[0])
                 + len(find_peaks(-w, prominence=0.10 * rng)[0]))
        z = (w - w.mean()) / (w.std() + 1e-12)           # 4: asymmetry (skew, ddof=0)
        skew = float(np.mean(z ** 3))
    return np.array([rng, activity, curvature, float(n_ext), skew])


# ============================================================================= clustering / K-selection
def medoid(idx, D):
    sub = D[np.ix_(idx, idx)]
    return idx[int(sub.sum(1).argmin())]


def select_k_precomputed(D, kmax, linkage_method):
    """K selection by relative comparison on a precomputed distance matrix (Stage1 SBD) + merge/outlier absorption."""
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
    lab = merge_close(lab, D)
    lab = absorb_singletons(lab, D)
    lab = relabel(lab)
    return lab, {"sils": sils, "best_sil": round(sil, 3), "K": len(set(lab))}


def select_k_features(F, kmax):
    """K selection by Ward + relative comparison on feature vectors (Stage2 whitened D3)."""
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
    lab = absorb_singletons_feat(lab, F)
    lab = relabel(lab)
    return lab, {"sils": sils, "best_sil": round(sil, 3), "K": len(set(lab))}


def cluster_two_stage(beats_raw, shapes_robust):
    """Two-stage clustering.
    Stage1: coarse grouping into "types" by full-shift SBD of robust z-norm shapes + complete-linkage.
    Stage2: global-standardize D3 (5 features, RAW) over all beats -> Mahalanobis whiten -> Ward to
            subdivide each Stage1 cluster by amplitude/sharpness/curvature."""
    # ---- Stage 1: shape (amplitude ignored) ----
    D1 = sbd_full_matrix(shapes_robust)
    lab1, info1 = select_k_precomputed(D1, KMAX_S1, "complete")

    # ---- Stage 2: amplitude/scale + fine-shape statistics ----
    F = np.array([d3_features(w) for w in beats_raw])            # raw D3
    Fs = (F - F.mean(0)) / (F.std(0) + 1e-9)                     # global standardization
    cov = np.cov(Fs, rowvar=False) + MAHAL_RIDGE * np.eye(Fs.shape[1])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 1e-9, None)
    Wmat = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T          # whitening matrix
    Fw = Fs @ Wmat                                              # Mahalanobis space

    final = np.full(len(beats_raw), -1, dtype=int)
    nxt, k_sub = 0, {}
    for c1 in sorted(set(lab1)):
        idx = np.where(lab1 == c1)[0]
        lab2, info2 = select_k_features(Fw[idx], KMAX_S2)
        # acceptance guard: merge sub-clusters that are essentially the same in both shape and amplitude (removes spurious splits straddling the n_ext threshold)
        lab2 = guard_stage2(lab2, shapes_robust[idx], F[idx])
        k_sub[int(c1)] = len(set(lab2))
        for sub in sorted(set(lab2)):
            final[idx[lab2 == sub]] = nxt
            nxt += 1
    final = relabel(final)
    info = {"K_s1": int(info1["K"]), "K_sub": k_sub, "K": int(len(set(final))),
            "stage1_sils": info1.get("sils", {})}
    return final, len(set(final)), info, D1


def sbd_full_pair(a, b):
    cc = np.correlate(a, b, mode="full")
    return float(1.0 - cc.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _reldiff(a, b):
    return abs(a - b) / (0.5 * (abs(a) + abs(b)) + 1e-9)


def guard_stage2(lab, shapes, F):
    """Merge Stage2 sub-cluster pairs that are "essentially the same in both shape and amplitude submode".
    A pair is kept only if (a) shapes differ: medoid full-SBD > STAGE2_SBD, or
    (b) a real amplitude submode exists = relative range difference > STAGE2_AMP_RANGE and
       (relative activity difference > STAGE2_AMP_ACT or relative curv difference > STAGE2_AMP_CURV).
    -> removes spurious splits from straddling the n_extrema threshold or from cutting a continuous gradient where only range changes.
    F is each beat's raw D3 features [range, activity, curv, n_ext, skew]."""
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


def _artifact_raw_features(beats):
    """Raw features for artifact spans. On a flat plateau (saturation/calibration), robust z-norm degenerates and shape SBD
    becomes unstable, so types are separated by raw-amplitude statistics.
    Default = [mid*rng, trend]: mid = fraction of time in the mid-band (an instantaneous pulse vs a sustained plateau), trend = last 1/3 - first 1/3 (transition in/out).
    ARTIFACT_LEVEL_FEAT=True = [floor*grng, range, mid*rng, trend]: adds floor = whether it drops near 0 (1/0),
      separating a high-floating flat plateau (min~5700, small range) from the full [0->plateau->0] body (min=0, large range) (244-247)."""
    # amplitude reference for the floor decision (consistent across train/test). If _ART_GREF is unset, estimate from beats.
    if _ART_GREF is not None:
        gmin, grng = _ART_GREF
    else:
        gmin = min(float(np.asarray(w).min()) for w in beats)
        grng = max(float(np.asarray(w).max()) for w in beats) - gmin + 1e-9
    F = []
    for w in beats:
        w = np.asarray(w, float); t3 = max(1, len(w) // 3); rng = float(np.ptp(w)) + 1e-9
        mid = float(((w > w.min() + 0.25 * rng) & (w < w.max() - 0.25 * rng)).mean())
        trend = w[-t3:].mean() - w[:t3].mean()
        if ARTIFACT_LEVEL_FEAT:
            # add floor = "whether the span drops near 0 (1) / floats high (0)" + range.
            # -> separates the full body [0->plateau->0] (floor=1, large range) from a high flat plateau (floor=0, small range).
            floor = 1.0 if float(w.min()) < gmin + 0.20 * grng else 0.0
            F.append([floor * grng, rng, mid * rng, trend])
        else:
            # shape features only: mid separates "an instantaneous pulse vs a sustained plateau", trend separates "transition in (-) vs transition out (+)".
            # mean/range/level are excluded because they confuse the transition-in with the body (the two have similar mean/amplitude).
            F.append([mid * rng, trend])
    return np.array(F)


def cluster_artifacts(shapes, beats=None):
    """Cluster non-clean regions (flat/square-wave/fragment/noise).
    Default is shape SBD (SBD < ARTIFACT_TAU = same shape). ARTIFACT_RAW_FEATURES switches to raw features (level/width/trend):
    in long recordings containing degenerate flat plateaus, this consistently clusters recurring transition-in/flat/transition-out regardless of location."""
    n = len(shapes)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    if ARTIFACT_RAW_FEATURES and beats is not None:
        F = _artifact_raw_features(beats)
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-9)
        Z = linkage(Fz, method="ward")
        # separate types by a distance threshold (silhouette-best-K picks the coarsest and collapses transition-in/flat/transition-out).
        #   in the standardized feature space, below ARTIFACT_FEAT_TAU is the same type -> recurring same-kind events are clustered consistently regardless of location.
        lab = fcluster(Z, t=ARTIFACT_FEAT_TAU, criterion="distance") - 1
        return relabel(lab)
    D = sbd_full_matrix(shapes)
    Z = linkage(D[np.triu_indices(n, 1)], method="complete")
    lab = fcluster(Z, t=ARTIFACT_TAU, criterion="distance") - 1
    return relabel(lab)


def absorb_singletons_feat(lab, F):
    """Absorb clusters with size < MIN_PATTERN into the nearest centroid in feature space."""
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
        others = [d for d in ids if d != c]
        target = min(others, key=lambda d: np.linalg.norm(cc - F[lab == d].mean(0)))
        lab[lab == c] = target
    return lab


def absorb_singletons(lab, D):
    """Absorb clusters with size < MIN_PATTERN into the cluster of the nearest medoid.
    A cluster with only one to a few beats is treated as a one-off outlier beat, not a "recurring waveform pattern"."""
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
        others = [d for d in ids if d != c]
        target = min(others, key=lambda d: D[cm, medoid(np.where(lab == d)[0], D)])
        lab[lab == c] = target
    return lab


def merge_close(lab, D):
    lab = lab.copy()
    while True:
        ids = sorted(set(lab))
        if len(ids) < 2:
            break
        meds = {c: medoid(np.where(lab == c)[0], D) for c in ids}
        closest, best_d = None, 1e9
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d = D[meds[ids[i]], meds[ids[j]]]
                if d < best_d:
                    best_d, closest = d, (ids[i], ids[j])
        if best_d < MERGE_TAU:
            a, b = closest
            lab[lab == b] = a
        else:
            break
    return lab


def relabel(lab):
    order = {c: i for i, c in enumerate(sorted(set(lab), key=lambda c: -(lab == c).sum()))}
    return np.array([order[c] for c in lab])



# ============================================================================= singleton re-examination rule
def _medoid_spread(shapes_rz, idx):
    """Medoid shape of cluster idx and its internal spread (90th percentile of SBD to the medoid).
    Large clusters are evenly subsampled to <=80 members to limit the O(n^2) SBD (speedup)."""
    idx = np.asarray(idx)
    if len(idx) == 1:
        return shapes_rz[idx[0]], 0.0
    sub = idx if len(idx) <= 80 else idx[np.linspace(0, len(idx) - 1, 80).astype(int)]
    D = sbd_full_matrix(shapes_rz[sub])
    mi = int(D.sum(1).argmin())
    return shapes_rz[sub[mi]], float(np.percentile(D[mi], 90))


def _nearest_cluster(shape, labels, shapes_rz, exclude):
    """Return the cluster closest to shape (other than exclude): (SBD, cluster, acceptance threshold).
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
    """Re-split the span [s,e] (a long under-segmented unit) at internal troughs and return the sub-spans."""
    seg = xt[s:e]
    if len(seg) < int(2 * period * 0.5):
        return [(s, e)]
    w = max(5, int(period * 0.08) | 1)
    xs = savgol_filter(seg, window_length=min(w, len(seg) - (1 - len(seg) % 2)), polyorder=2)
    norm = _local_normalize(xs, win=int(period * 1.5))
    feet, _ = find_peaks(-norm, distance=max(3, int(period * 0.55)), prominence=0.30)
    if len(feet) < 1:
        return [(s, e)]
    bounds = sorted(set([0] + [int(f) for f in feet] + [len(seg)]))
    # keep full coverage: do not discard too-short fragments; merge them into the next fragment to tile [s,e] with no gaps.
    subs, i = [], 0
    while i < len(bounds) - 1:
        a, b = bounds[i], bounds[i + 1]
        while (b - a) < int(period * 0.5) and (i + 2) < len(bounds):
            i += 1
            b = bounds[i + 1]
        subs.append((s + a, s + b))
        i += 1
    return subs


def reconcile_singletons(xt, spans, labels, period, k_orig):
    """singleton/small-cluster re-examination rule (added 2026-06-25, generalizing a user request):
    (a) **same-shape recovery**: if an artifact cluster's (label >= k_orig) medoid is within the nearest pulse cluster's
        internal spread (medoid SBD p90) x SINGLE_FACTOR, merge the whole cluster into that pulse
        (recovers normal cycles judged non-clean due to noise/distortion into c0 etc.; relative comparison).
    (b) **re-split**: a size-1 artifact of length ~2 periods or more (an under-segmented multi-cycle) is re-split at internal
        troughs, and each piece is merged into the nearest cluster (if there is no target, kept as a standalone artifact).
    (c) **keep standalone**: a genuinely different shape (partial/double/singular) is left as is.
    Returns: the updated spans, labels. (Speedup: cluster medoids are computed once per iteration.)"""
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
        # (a) same-shape recovery of a whole artifact cluster into a pulse
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
    """Apply reconcile_singletons, re-decide pulse/artifact, and renumber with pulses first.
    A pulse cluster = an original pulse label (< k_orig) that survived. New labels created by a split are artifacts."""
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


def _promote_consistent_artifacts(xt, spans, labels, k_pulse, n_art, period):
    """Later stage that decides clean/artifact by cluster consistency. Among artifact clusters (label >= k_pulse),
    promote to pulse (clean) those satisfying: (1) beat count >= PROMOTE_MIN_N (recurring in train), (2) median length in [PROMOTE_LEN_LO*P, PROMOTE_LEN_HI*P]
    (completes one cycle), (3) median internal medoid SBD <= PROMOTE_SBD (consistent shape).
    -> fixes is_beat_shape's per-beat flattop wrongly dropping a normal alternate morphology that consistently recurs (even if flat/square/trapezoid)
    to artifact. Incomplete fragments (truncated) / inconsistent / extreme-length stay artifact."""
    if not PROMOTE_ARTIFACTS or n_art == 0:
        return labels, k_pulse, n_art
    labels = list(int(l) for l in labels)
    lo, hi = PROMOTE_LEN_LO * period, PROMOTE_LEN_HI * period
    promote = []
    for c in range(k_pulse, k_pulse + n_art):
        idx = [i for i in range(len(labels)) if labels[i] == c]
        if len(idx) < PROMOTE_MIN_N:
            continue                                    # does not recur -> stays artifact
        lens = np.array([spans[i][1] - spans[i][0] for i in idx])
        if not (lo <= np.median(lens) <= hi):
            continue                                    # does not complete one cycle (incomplete/concatenated) -> artifact
        sh = np.array([resample(robust_znorm(xt[spans[i][0]:spans[i][1] + 1])) for i in idx])
        med = sh[sbd_full_matrix(sh).sum(1).argmin()]
        if float(np.median([sbd_full_pair(med, s) for s in sh])) <= PROMOTE_SBD:
            promote.append(c)                           # internally consistent (a normal alternate morphology even if rare) -> promote to clean
    if not promote:
        return labels, k_pulse, n_art
    remaining = [c for c in range(k_pulse, k_pulse + n_art) if c not in promote]
    new_k = k_pulse + len(promote)
    mp = {c: c for c in range(k_pulse)}
    for j, c in enumerate(promote):
        mp[c] = k_pulse + j                             # promoted artifact -> end of pulses
    for j, c in enumerate(remaining):
        mp[c] = new_k + j                               # remaining artifact -> after the pulses
    return [mp[l] for l in labels], new_k, len(remaining)
# ============================================================================= driver
def process(series_id, save=True):
    path = find_file(series_id)
    x = np.loadtxt(path).astype(float)
    train_end, anom_s, anom_e = parse_meta(path)

    # the basic waveform is learned from the "normal training region" (UCR: the leading train_end is normal)
    xt = x[:train_end] if train_end > 200 else x
    global _ART_GREF
    _ART_GREF = (float(x.min()), float(np.ptp(x)) + 1e-9)   # train/test-shared amplitude reference for the floor decision

    period = PERIOD if PERIOD else dominant_period(xt)
    spans, clean, peaks = segment_cycles(xt, period)
    clean_idx = [i for i, c in enumerate(clean) if c]
    if len(clean_idx) < 3:
        print(f"[{series_id}] segmentation failed (period={period}, clean={len(clean_idx)})")
        return None

    # per beat: a robust z-norm shape for Stage1, a raw span (end inclusive) for Stage2
    shapes_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans])
    beats_raw = [xt[s:e + 1] for s, e in spans]

    # two-stage clustering on "beat-like" spans only (not misled by artifacts)
    ci = clean_idx
    labels_c, K, info, D1c = cluster_two_stage(
        [beats_raw[i] for i in ci], shapes_rz[ci])

    # clean beats get the validated two-stage clusters (pulse clusters 0..K-1).
    spans_plot = spans
    labels_plot = np.full(len(spans), -1, dtype=int)
    for k, i in enumerate(ci):
        labels_plot[i] = int(labels_c[k])

    # non-clean regions (flat/square-wave/fragment/noise) are also grouped by shape as "clustering candidates",
    # numbered after the pulse clusters (same shapes group together; singular ones stay standalone even at n=1).
    nc = [i for i in range(len(spans)) if labels_plot[i] < 0]
    n_art = 0
    if nc:
        art = cluster_artifacts(shapes_rz[nc], [beats_raw[i] for i in nc])
        for j, i in enumerate(nc):
            labels_plot[i] = K + int(art[j])
        n_art = int(art.max()) + 1

    # singleton re-examination rule: merge/re-split size-1/small clusters (relative comparison). Keep pulse-origin beats.
    spans, labels_plot, K, n_art, _ = _apply_reconcile(
        xt, spans, list(labels_plot), period, K, set(clean_idx))
    labels_plot = np.array(labels_plot)
    # MERGE_ARTIFACTS: after reconcile, merge the remaining artifact clusters (label >= K_pulse) into one.
    #   doing this after reconcile keeps legitimate same-shape recovery (into pulses) while merging only the many
    #   singletons of non-physiological artifacts (dropout/saturation/flat) into one cluster (not absorbed into pulses).
    if MERGE_ARTIFACTS and n_art > 1:
        labels_plot = np.where(labels_plot >= K, K, labels_plot)
        n_art = 1
    spans_plot = spans
    K_total = K + n_art
    info["K_pulse"], info["K_artifact"] = K, n_art

    # for the overlay display use std-based z-norm (robust/MAD diverges on saturated beats and is hard to read)
    shapes_plot = np.array([znorm(resample(xt[s:e + 1])) for s, e in spans_plot])

    # ---- apply the learned split to the test region (train_end onward) too, assigning each unit to the nearest train pulse cluster medoid ----
    #      (the basic waveform is learned on train; test is classified by nearest medoid without re-clustering = same structure as VLM-AD).
    final_rz = np.array([resample(robust_znorm(xt[s:e + 1])) for s, e in spans_plot])
    pulse_medoids = {}
    clu_size = {}
    for c in range(K_total):       # medoid of every pulse + artifact cluster
        idx = np.where(np.asarray(labels_plot) == c)[0]
        clu_size[c] = len(idx)
        if len(idx):
            sub = final_rz[idx]
            pulse_medoids[c] = sub[sbd_full_matrix(sub).sum(1).argmin()] if len(idx) > 1 else sub[0]

    def _clean_by_medoid(shape, lab):
        """per-beat clean decision: clean if SBD to the cluster medoid is <= CLEAN_SBD (similar) and
        the cluster size is >= CLEAN_MIN_N (recurring). Singleton/dissimilar is artifact."""
        if not CLEAN_BY_MEDOID:
            return lab < K                                   # legacy: pulse cluster (< K_pulse) = clean
        return clu_size.get(lab, 0) >= CLEAN_MIN_N and \
            float(sbd_full_pair(pulse_medoids[lab], shape)) <= CLEAN_SBD

    clean_train = [bool(_clean_by_medoid(final_rz[i], int(labels_plot[i])))
                   for i in range(len(spans_plot))]
    # when ARTIFACT_RAW_FEATURES: prepare to consistently assign test non-clean events (flat/drop/transition) to the train artifact clusters
    #   by raw features (standardized). Clean beats go to a pulse by nearest SBD as before.
    art_cent, art_fmean, art_fstd = {}, None, None
    if ARTIFACT_RAW_FEATURES and n_art > 0:
        a_idx = [i for i in range(len(spans_plot)) if labels_plot[i] >= K]
        if a_idx:
            Fa = _artifact_raw_features([xt[spans_plot[i][0]:spans_plot[i][1] + 1] for i in a_idx])
            art_fmean, art_fstd = Fa.mean(0), Fa.std(0) + 1e-9
            Faz = (Fa - art_fmean) / art_fstd
            acc = {}
            for pos, i in enumerate(a_idx):
                acc.setdefault(int(labels_plot[i]), []).append(Faz[pos])
            art_cent = {c: np.mean(v, 0) for c, v in acc.items()}
    # per-pulse-cluster amplitude (median range): when K_pulse>1 (amplitude-regime split), so that clean test-beat assignment
    #   is decided by amplitude as well as shape SBD (SBD is amplitude-invariant, so without this test would not separate by amplitude).
    pulse_amp = {}
    for c in range(K):
        idx = [i for i in range(len(spans_plot)) if labels_plot[i] == c]
        if idx:
            pulse_amp[c] = float(np.median([np.ptp(xt[spans_plot[i][0]:spans_plot[i][1] + 1]) for i in idx]))
    amp_scale = (max(pulse_amp.values()) - min(pulse_amp.values())) + 1e-9 if len(pulse_amp) > 1 else 1.0
    test_spans, test_labels, clean_test = [], [], []
    x_disp = xt
    if 200 < train_end < len(x) and pulse_medoids:
        xtt = x[train_end:].astype(float)
        # apply the exact same split path as train (segment_cycles + _resegment_square) to test too.
        #   consistent square-wave splitting is handled by _resegment_square inside segment_cycles, so no extra split is done here
        #   (the former period-based long-span split re-fragmented _resegment_square's 250-period with >1.5*period,
        #   causing train/test mismatch, so it was removed).
        ts, tc, _ = segment_cycles(xtt, period)
        for (s, e), cln in zip(ts, tc):                     # assign every test segment to the nearest cluster (never -1)
            if art_cent and not cln:                         # non-clean event -> nearest artifact cluster by raw features (consistent)
                f = (_artifact_raw_features([xtt[s:e + 1]])[0] - art_fmean) / art_fstd
                lab = min(art_cent, key=lambda c: float(np.linalg.norm(art_cent[c] - f)))
            elif art_cent:                                   # clean beat -> to a pulse cluster (shape SBD + amplitude)
                shp = resample(robust_znorm(xtt[s:e + 1])); pb = float(np.ptp(xtt[s:e + 1]))
                lab = min(range(K), key=lambda k: float(sbd_full_pair(pulse_medoids[k], shp))
                          + (AMP_ASSIGN_W * abs(pb - pulse_amp.get(k, pb)) / amp_scale if K > 1 else 0.0))
            else:                                            # default: nearest SBD over all clusters (K_pulse>1 also factors in amplitude)
                shp = resample(robust_znorm(xtt[s:e + 1])); pb = float(np.ptp(xtt[s:e + 1]))
                lab = min(pulse_medoids, key=lambda k: float(sbd_full_pair(pulse_medoids[k], shp))
                          + (AMP_ASSIGN_W * abs(pb - pulse_amp.get(k, pb)) / amp_scale if (K > 1 and k < K) else 0.0))
            test_spans.append((int(s + train_end), int(e + train_end)))
            test_labels.append(int(lab))
            clean_test.append(bool(_clean_by_medoid(resample(robust_znorm(xtt[s:e + 1])), int(lab))))
        x_disp = np.concatenate([xt, xtt])

    print(f"[{series_id}] train_len={train_end:6d} period={period:4d}  "
          f"clean_beats={len(ci)} K_pulse={K} K_artifact={n_art}  K={K_total}  "
          f"test_seg={len(test_spans)}  {info}")
    K = K_total
    if save:
        plot(series_id, x_disp, spans_plot, labels_plot, K, anom_s, anom_e, shapes_plot,
             train_end, k_pulse=info["K_pulse"], test_spans=test_spans, test_labels=test_labels)
    return {"series": series_id, "train_len": int(train_end), "period": int(period),
            "n_segments": len(spans_plot), "n_clean": len(ci), "K": int(K), "info": info,
            "segments": [{"i": i, "start": int(s), "end": int(e), "cluster": int(l),
                          "clean": bool(clean_train[i]), "region": "train"}
                         for i, ((s, e), l) in enumerate(zip(spans_plot, labels_plot))]
                        + [{"i": len(spans_plot) + j, "start": int(s), "end": int(e),
                            "cluster": int(l), "clean": bool(clean_test[j]), "region": "test"}
                           for j, ((s, e), l) in enumerate(zip(test_spans, test_labels))]}


def plot(series_id, x, spans, labels, K, anom_s, anom_e, shapes, train_end, k_pulse=None,
         test_spans=None, test_labels=None):
    """PDF output: wrap every ROW_STEPS (=2000) steps per row, separate each beat with vertical lines at the split points,
    and background-highlight by its cluster color. Append cluster overlays on the final page.
    If test_spans/test_labels are passed, the region from train_end onward (test) is also drawn, colored by the assigned
    train pulse cluster medoid. train/test boundary = solid blue line, injected anomaly span (anom_s..anom_e) = solid red line + light red band."""
    from matplotlib.backends.backend_pdf import PdfPages
    os.makedirs(OUT_DIR, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    n = len(x)
    all_spans = list(spans) + list(test_spans or [])
    all_labels = list(labels) + list(test_labels or [])
    out = os.path.join(OUT_DIR, f"{series_id}.pdf")
    # unify the PDF vertical-axis scale across all rows (no per-row auto-scaling). To avoid being crushed by extreme spikes
    #   (saturation/dropout), compute a robust percentile range once over the whole signal and apply it to all axes.
    ylo, yhi = np.percentile(x, 0.3), np.percentile(x, 99.7)
    if yhi <= ylo:
        ylo, yhi = float(np.min(x)), float(np.max(x)) + 1e-9
    ymrg = 0.05 * (yhi - ylo)
    YLIM = (ylo - ymrg, yhi + ymrg)

    with PdfPages(out) as pdf:
        # ---- waveform pages: 1 row = ROW_STEPS steps ----
        nrows = int(np.ceil(n / ROW_STEPS))
        # cap the max page count on long series to avoid a bloated PDF, but always show up to the injected anomaly span anom_e.
        max_pages = int(np.ceil(nrows / ROWS_PER_PAGE))
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
                # light red band for the injected anomaly span (bottom background layer)
                if anom_e and anom_s and not (anom_e <= w0 or anom_s >= w1):
                    ax.axvspan(max(anom_s, w0), min(anom_e, w1), color="red", alpha=0.10, lw=0, zorder=0)
                ax.plot(range(w0, w1), x[w0:w1], color="black", lw=0.5, zorder=3)
                # color each span overlapping this row by cluster color (train+test, number color for both pulse and artifact)
                for (s, e), lab in zip(all_spans, all_labels):
                    if e <= w0 or s >= w1:
                        continue
                    col = "lightgray" if lab < 0 else cmap(lab % 10)
                    ax.axvspan(max(s, w0), min(e, w1), color=col,
                               alpha=0.30, lw=0, zorder=1)
                    if w0 <= s < w1:                          # vertical line at the split point
                        ax.axvline(s, color="dimgray", lw=0.5, ls="--", zorder=2)
                ax.set_ylim(*YLIM)                            # vertical-axis scale shared across all rows
                if w0 <= train_end < w1:                      # train/test boundary
                    ax.axvline(train_end, color="blue", lw=1.6, ls="-", zorder=5)
                for ab in (anom_s, anom_e):                   # anomaly span boundaries
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
                fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # ---- final page: per-cluster overlays (many clusters wrap in a grid) ----
        ncol = min(7, max(K, 1))
        nrow = int(np.ceil(max(K, 1) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.4 * nrow),
                                squeeze=False)
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
        for c in range(K, nrow * ncol):                       # turn off the leftover panels
            axs[c // ncol, c % ncol].axis("off")
        fig.suptitle(f"{series_id}  cluster shapes (z-norm, resampled)", fontsize=11)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)


# keys among core's defaults that may be overridden by a <id>.py PARAMS
TUNABLE = ("RESAMPLE_LEN", "MERGE_ARTIFACTS", "ARTIFACT_RAW_FEATURES", "ARTIFACT_FEAT_TAU", "ARTIFACT_LEVEL_FEAT", "KMAX_S1", "KMAX_S2", "MAXLAG_FRAC", "MERGE_TAU", "SIL_MIN", "AMP_ASSIGN_W",
           "MIN_PATTERN", "MAHAL_RIDGE", "STAGE2_SBD", "STAGE2_AMP_RANGE",
           "STAGE2_AMP_ACT", "STAGE2_AMP_CURV", "ARTIFACT_TAU",
           "FLATTOP_MAX", "DEADFLAT_MAX", "EDGE_MISMATCH_MAX", "PEAKPOS_MAX", "NOISE_PEAKS_MAX",
           "PERIOD")  # PERIOD: specify the period manually (None = auto-estimate)
PERIOD = None


def run(series_id, params=None, save=True):
    """Public API that processes a single series. Called from `<id>.py`.
    params can override core's defaults (TUNABLE) (for series-specific tuning)."""
    g = globals()
    saved = {k: g[k] for k in TUNABLE}
    try:
        for k, v in (params or {}).items():
            if k not in TUNABLE:
                raise KeyError(f"unknown param '{k}'. valid: {TUNABLE}")
            g[k] = v
        r = process(series_id, save=save)
        if r:
            # per-series beat-span + cluster-assignment CSV
            os.makedirs(OUT_DIR, exist_ok=True)
            with open(os.path.join(OUT_DIR, f"{series_id}_segments.csv"), "w") as f:
                f.write("seg_index,start,end,cluster,clean,region\n")
                for s in r["segments"]:
                    f.write(f"{s['i']},{s['start']},{s['end']},"
                            f"{s['cluster']},{int(s['clean'])},{s.get('region', 'train')}\n")
        return r
    finally:
        for k, v in saved.items():       # restore the globals (so other series are unaffected)
            g[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", nargs="+", help="series id(s) e.g. 024 245")
    ap.add_argument("--all", action="store_true", help="all ABP series")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    ids = abp_series_ids() if args.all else (args.series or ["024", "026", "027", "028", "089", "142", "245"])
    results = []
    for sid in ids:
        r = process(sid, save=not args.no_save)
        if r:
            results.append(r)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # per-series cluster-count summary CSV
    with open(os.path.join(OUT_DIR, "summary.csv"), "w") as f:
        f.write("series,train_len,period,n_segments,n_clean,K\n")
        for r in results:
            f.write(f"{r['series']},{r['train_len']},{r['period']},"
                    f"{r['n_segments']},{r['n_clean']},{r['K']}\n")
    # per-beat (basic waveform) span and cluster-assignment CSV
    with open(os.path.join(OUT_DIR, "segments.csv"), "w") as f:
        f.write("series,seg_index,start,end,cluster,clean,region\n")
        for r in results:
            for s in r["segments"]:
                f.write(f"{r['series']},{s['i']},{s['start']},{s['end']},"
                        f"{s['cluster']},{int(s['clean'])},{s.get('region', 'train')}\n")

    print(f"\n-> {len(results)} series. output: summary.json / summary.csv / segments.csv / <id>.pdf in {OUT_DIR}")


if __name__ == "__main__":
    main()
