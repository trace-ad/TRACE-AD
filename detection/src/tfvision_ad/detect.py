"""Detector: per-series signal + beat segmentation -> per-channel score arrays.

Dataset-agnostic.  ``data.py`` adapters turn UCR/TSB files into the
``(signal, train_end, seg_rows)`` triple this consumes.

``detect`` also returns a per-stage timing dict (seconds) so inference cost can
be recorded alongside the scores.
"""
import time
from collections import defaultdict

import numpy as np

from . import channels as ch_
from .encode import encode, grid_columns
from .render import render_beat, znorm


def _reference_beats(rows, ref):
    """Clean train beats grouped by cluster (the phase-aligned memory bank)."""
    trc = [r for r in rows if r["region"] == "train" and (ref != "clean" or r["clean"] == "1")]
    by = defaultdict(list)
    for r in trc:
        by[int(r["cluster"])].append(r)
    return by


def select_refs(rows, sig, n, strategy, seed, sbd_len):
    """Pick ``n`` clean beats from a cluster (N_ref ablation).

    Keeps all beats when ``strategy`` is None, ``n`` is None, or the cluster has
    <= n beats.  Distance is Euclidean on z-normalized, length-normalized beat
    shapes; the medoid is the beat with the smallest total distance.
      random  : n random beats (seeded)
      medoid  : n nearest to the medoid (representative)
      diverse : farthest-point sampling from the medoid (coverage)
    """
    if strategy is None or n is None or len(rows) <= n:
        return rows
    F = np.array([znorm(ch_.resample(sig[int(r["start"]):int(r["end"]) + 1], sbd_len)) for r in rows])
    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
    med = int(np.argmin(D.sum(1)))
    if strategy == "random":
        idx = np.random.RandomState(seed).choice(len(rows), n, replace=False)
    elif strategy == "medoid":
        idx = np.argsort(D[med])[:n]
    elif strategy == "diverse":
        idx = [med]
        while len(idx) < n:
            dmin = D[idx].min(0)
            dmin[idx] = -1.0
            idx.append(int(np.argmax(dmin)))
        idx = np.array(idx)
    else:
        return rows
    return [rows[int(i)] for i in idx]


def detect(sig, te, rows, cfg, model, device):
    """Return ``(scores, ul, T, timing)``.

    ``scores`` maps channel name -> float array of length T (test region).
    ``ul`` is the mean beat length (moving-average window used at fusion).
    ``timing`` maps stage name -> seconds; ``ref_total`` is memory-bank build,
    ``detect_total`` is test-time detection, ``total`` is their sum.
    """
    R = cfg.grid_cols
    ks = list(cfg.ng_sweep) if cfg.ng_sweep else [cfg.ng_window]
    T = len(sig) - te
    lo, hi = float(sig[:te].min()), float(sig[:te].max())          # train-only, leak-free
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 or 1.0
    P = 224 // R
    shifts = ch_.sbd_shifts(cfg.sbd_len, cfg.sbd_margin)
    tm = {"render_ref": 0.0, "encode_ref": 0.0, "prep": 0.0, "sh": 0.0,
          "render_test": 0.0, "encode_test": 0.0, "amp_fine": 0.0, "amp_ng": 0.0}

    by = _reference_beats(rows, cfg.ref)
    if cfg.ref_strategy is not None and cfg.ref_count is not None:
        by = {c: select_refs(lst, sig, cfg.ref_count, cfg.ref_strategy, cfg.ref_seed, cfg.sbd_len)
              for c, lst in by.items()}
    tsg = sorted([r for r in rows if r["region"] == "test"], key=lambda r: int(r["start"]))
    if not by or not tsg:
        return None

    units = []
    for r in tsg:
        c = int(r["cluster"]); s, e = int(r["start"]), int(r["end"])
        qs, qe = s - te, min(e - te, T - 1)
        if qe > qs:
            units.append((c, qs, qe, s, e))
    if not units:
        return None

    keys = []
    if "amp_fine" in cfg.channels:
        keys.append("amp_fine")
    if "sh" in cfg.channels:
        keys.append("sh")
    if "amp_ng" in cfg.channels:
        if cfg.ng_causal and cfg.ng_sweep:
            keys += [f"amp_ng{tag}{k}" for tag in ("i", "e") for k in ks]
        elif cfg.ng_sweep:
            keys += [f"amp_ng{k}" for k in ks]
        else:
            keys += ["amp_ng"]
    scores = {k: np.full(T, np.nan) for k in keys}

    def fill(arr, cs, qs, qe, Lseg):
        for j in range(qs, qe + 1):
            col = min(R - 1, int((j - qs) * 224 / max(Lseg - 1, 1)) // P)
            if not np.isnan(cs[col]):
                arr[j] = cs[col]

    # ---- sh channel (SBD, vertical-independent) --------------------------- #
    if "sh" in cfg.channels:
        t = time.perf_counter()
        refsh = {c: np.array([znorm(ch_.resample(sig[int(r["start"]):int(r["end"]) + 1], cfg.sbd_len))
                              for r in lst]) for c, lst in by.items()}
        fnsh = {c: np.fft.rfft(v, axis=1) for c, v in refsh.items()}
        anyS = next(iter(refsh))
        for (c, qs, qe, s, e) in units:
            Lseg = qe - qs + 1
            NORM = refsh.get(c, refsh[anyS]); FN = fnsh.get(c, fnsh[anyS])
            res = ch_.sbd_residual(znorm(ch_.resample(sig[s:e + 1], cfg.sbd_len)), FN, NORM, cfg.sbd_len, shifts)
            rq = np.interp(np.linspace(0, cfg.sbd_len - 1, Lseg), np.arange(cfg.sbd_len), res)
            for kk, j in enumerate(range(qs, qe + 1)):
                scores["sh"][j] = rq[kk]
        tm["sh"] = time.perf_counter() - t

    # ---- amp_fine / amp_ng channels (CLIP patch matching) ----------------- #
    need_clip = ("amp_fine" in cfg.channels) or ("amp_ng" in cfg.channels)
    if need_clip:
        t = time.perf_counter()
        Ra = []; fkey = []
        for c, lst in by.items():
            for r in lst:
                Ra.append(render_beat(sig[int(r["start"]):int(r["end"]) + 1].astype(float), lo, hi, cfg.vscale))
                fkey.append(c)
        tm["render_ref"] = time.perf_counter() - t

        t = time.perf_counter()
        ea = encode(Ra, model, device, cfg.batch_size)
        tm["encode_ref"] = time.perf_counter() - t

        t = time.perf_counter()
        aref = defaultdict(list)
        for k, c in enumerate(fkey):
            aref[c].append(ea[k])
        gA = {c: [grid_columns(x[0], x[1], R) for x in v] for c, v in aref.items()}
        anyA = next(iter(gA))
        NC = {}; BK = {}; CMS = {}
        for c, gs in gA.items():
            rcp = [ch_.col_patches(g, f, R) for (g, f) in gs]
            NC[c] = ch_.col_profile([ch_.col_mean(rc) for rc in rcp], R)
            BK[c] = ch_.build_bank(rcp, R, device)
            arrs = []; msks = []
            for g, f in gs:
                a, m = ch_.col_mean_arr(ch_.col_mean(ch_.col_patches(g, f, R)), R)
                arrs.append(a); msks.append(m)
            CMS[c] = (np.stack(arrs), np.stack(msks))
        tm["prep"] = time.perf_counter() - t

        t = time.perf_counter()
        Ta = [render_beat(sig[s:e + 1].astype(float), lo, hi, cfg.vscale) for (c, qs, qe, s, e) in units]
        tm["render_test"] = time.perf_counter() - t

        t = time.perf_counter()
        ta = encode(Ta, model, device, cfg.batch_size)
        tm["encode_test"] = time.perf_counter() - t

        for u, (c, qs, qe, s, e) in enumerate(units):
            Lseg = qe - qs + 1
            isedge = cfg.edge_relax and (u == 0 or u == len(units) - 1)
            de_fine = R if isedge else cfg.phase_tol
            de_ng = (R // 2) if isedge else cfg.phase_tol
            qa = grid_columns(ta[u][0], ta[u][1], R)
            exc = ch_.excursion_cols(sig[s:e + 1].astype(float), center, half, R)
            ncols = NC.get(c, NC[anyA])
            qcp = ch_.col_patches(qa[0], qa[1], R)
            qcm = ch_.col_mean(qcp)
            # Delta_q (global per-query offset). Disabled when cfg.global_align is
            # False: alignment then relies on the per-column +/- phase_tol search only.
            sh = ch_.best_phase_shift(qcm, ncols, R) if cfg.global_align else 0
            if "amp_fine" in cfg.channels:
                t = time.perf_counter()
                aB, abc = BK.get(c, BK[anyA])
                csr = ch_.amp_fine_score(qcp, sh, aB, abc, R, de_fine, device)
                csr = ch_.apply_excursion(csr, exc, cfg.vscale, cfg.tau)
                fill(scores["amp_fine"], csr, qs, qe, Lseg)
                tm["amp_fine"] += time.perf_counter() - t
            if "amp_ng" in cfg.channels:
                t = time.perf_counter()
                qarr, qmsk = ch_.col_mean_arr(qcm, R)
                Rc, Rmsk = CMS.get(c, CMS[anyA])
                if cfg.ng_causal and cfg.ng_sweep:
                    variants = [("i", 0), ("e", -1)]      # experiment: both anchorings
                elif cfg.ng_causal:
                    variants = [("", 0)]                  # confirmed: causal include-current
                else:
                    variants = [("", None)]               # centred
                for tag, eoff in variants:
                    ng = ch_.amp_ng_score(qarr, qmsk, Rc, Rmsk, sh, ks, R, de_ng, device,
                                          causal=cfg.ng_causal, end_off=(eoff or 0))
                    for k in ks:
                        off = ((k - 1) - (eoff or 0)) if cfg.ng_causal else k // 2
                        wi = (np.arange(R)[:, None] + (np.arange(k) - off)[None, :]) % R
                        excw = exc[wi].max(1)
                        ngk = ch_.apply_excursion(ng[k], excw, cfg.vscale, cfg.tau)
                        key = f"amp_ng{tag}{k}" if cfg.ng_sweep else "amp_ng"
                        fill(scores[key], ngk, qs, qe, Lseg)
                tm["amp_ng"] += time.perf_counter() - t

    for k in scores:
        scores[k][np.isnan(scores[k])] = np.nanmin(scores[k])
    ul = max(1, int(np.mean([int(r["end"]) - int(r["start"]) + 1 for r in rows])))

    tm["ref_total"] = tm["render_ref"] + tm["encode_ref"] + tm["prep"]
    tm["detect_total"] = tm["sh"] + tm["render_test"] + tm["encode_test"] + tm["amp_fine"] + tm["amp_ng"]
    tm["total"] = tm["ref_total"] + tm["detect_total"]
    tm["nref_used"] = int(sum(len(v) for v in by.values()))
    ref_spans = {c: [(int(r["start"]), int(r["end"])) for r in lst] for c, lst in by.items()}
    return scores, ul, T, tm, ref_spans
