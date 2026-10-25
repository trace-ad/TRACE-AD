"""Channel fusion: min-max per channel -> sum -> 1-beat moving average.

The ``sh`` channel additionally gets a flexible re-alignment (``shflex``) on the
first and last test beats, where onset/length has no in-distribution precedent
and the plain SBD residual is unreliable.
"""
from collections import defaultdict

import numpy as np

from .render import znorm


def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / ((x.max() - x.min()) + 1e-9)


def moving_average(s, w):
    return s if w <= 1 else np.convolve(s, np.ones(w) / w, mode="same")


def _shflex(qseg, stream, step=2):
    """Per-point |q - best-aligned normal window| by sliding over a normal stream."""
    Lq = len(qseg)
    if len(stream) < Lq + 1:
        stream = np.tile(stream, int(np.ceil((Lq + 1) / max(1, len(stream)))))
    q = znorm(qseg)
    best = 1e18
    bd = np.abs(q - q.mean())
    for st in range(0, len(stream) - Lq + 1, step):
        w = znorm(stream[st:st + Lq])
        d = np.abs(q - w)
        if d.mean() < best:
            best = d.mean()
            bd = d
    return bd


def _shflex_best(qseg, templates):
    """Best ``_shflex`` residual across several normal 2-cycle templates.

    Each template is one clean reference beat repeated twice (a 2-period
    template).  A boundary beat may be a partial cycle or span up to ~2 cycles,
    so we slide it over each 2-cycle template and keep the closest-fitting one.
    """
    best = None
    best_mean = 1e18
    for st in templates:
        bd = _shflex(qseg, st)
        m = float(bd.mean())
        if m < best_mean:
            best_mean = m
            best = bd
    if best is None:
        q = znorm(qseg)
        return np.abs(q - q.mean())
    return best


def fuse(scores, ul, sig, te, rows, T, ref_spans=None):
    """Fuse per-channel score arrays into one anomaly score of length ``T``.

    ``ref_spans`` (cluster -> list of (start, end)) is the same phase-aligned
    normal-beat reference set the channels use (post ``select_refs``); when given
    it builds the ``shflex`` normal stream, keeping the edge sh re-alignment
    consistent with the memory bank.  Falls back to the first 12 clean beats.
    """
    w = max(1, int(ul))
    chans = []

    if "sh" in scores:
        sh = scores["sh"].copy()
        if ref_spans is not None:
            beats = {c: sorted(segs) for c, segs in ref_spans.items() if segs}
        else:
            by = defaultdict(list)
            for r in rows:
                if r["region"] == "train" and r["clean"] == "1":
                    by[int(r["cluster"])].append((int(r["start"]), int(r["end"])))
            beats = {c: sorted(segs)[:12] for c, segs in by.items() if segs}
        # one 2-cycle template per reference beat (the beat repeated twice)
        templates = {c: [np.tile(sig[s:e + 1].astype(float), 2) for s, e in bl]
                     for c, bl in beats.items()}
        anyC = next(iter(templates)) if templates else None
        ts = sorted([r for r in rows if r["region"] == "test"], key=lambda r: int(r["start"]))
        if anyC is not None and len(ts) >= 2:
            for r in (ts[0], ts[-1]):
                c = int(r["cluster"]); s, e = int(r["start"]), int(r["end"])
                qs, qe = s - te, min(e - te, T - 1)
                if qe <= qs:
                    continue
                tpl = templates.get(c, templates[anyC])
                rr = _shflex_best(sig[s:e + 1].astype(float), tpl)
                sh[qs:qe + 1] = np.interp(np.linspace(0, len(rr) - 1, qe - qs + 1),
                                          np.arange(len(rr)), rr)
        chans.append(minmax(sh))

    for k in scores:
        if k == "sh":
            continue
        chans.append(minmax(scores[k]))

    fused = np.sum(chans, axis=0)
    return moving_average(fused, w)
