"""The three anomaly-score channels and their building blocks.

All functions are phase-aware: a beat is split into ``R`` phase columns and
matching is restricted to a ``+/- phase_tol`` circular neighbourhood after the
query is phase-aligned to the reference.

  amp_fine : per-patch 1-NN, weakest-link (min) over each column   -> local shape
  amp_ng   : n-gram column-window match (window = k)               -> mid-scale shape
  sh       : SBD residual on the whole beat (amplitude-invariant)  -> global shape
"""
import numpy as np
import torch


# --------------------------------------------------------------------------- #
# column features
# --------------------------------------------------------------------------- #
def col_patches(g, fg, R):
    """Per-column list of foreground patch features (or None if empty)."""
    return [g[np.where(fg[:, w])[0], w, :] if fg[:, w].any() else None
            for w in range(R)]


def col_mean(cols):
    """L2-normalized mean feature per column."""
    return [None if c is None else (lambda m: m / (np.linalg.norm(m) + 1e-9))(c.mean(0))
            for c in cols]


def col_profile(ncol, R):
    """Reference profile: mean column feature across reference beats."""
    out = []
    for w in range(R):
        vs = [nc[w] for nc in ncol if nc[w] is not None]
        out.append((lambda m: m / (np.linalg.norm(m) + 1e-9))(np.mean(vs, 0)) if vs else None)
    return out


def best_phase_shift(qcm, ncols, R):
    """Circular shift (0..R-1) that best aligns query column means to reference."""
    D = next((v.shape[0] for v in list(qcm) + list(ncols) if v is not None), None)
    if D is None:
        return 0
    Q = np.zeros((R, D)); qv = np.zeros(R, bool)
    N = np.zeros((R, D)); nv = np.zeros(R, bool)
    for w in range(R):
        if qcm[w] is not None:
            Q[w] = qcm[w]; qv[w] = True
        if ncols[w] is not None:
            N[w] = ncols[w]; nv[w] = True
    C = Q @ N.T
    best = (-1e9, 0); ar = np.arange(R)
    for sh in range(R):
        w2 = (ar + sh) % R
        valid = qv & nv[w2]; nn = int(valid.sum())
        if nn > 0:
            avg = C[ar[valid], w2[valid]].sum() / nn
            if avg > best[0]:
                best = (avg, sh)
    return best[1]


# --------------------------------------------------------------------------- #
# amp_fine : per-patch weakest-link
# --------------------------------------------------------------------------- #
def build_bank(rcp, R, device):
    """Flatten all reference beats' column patches into a GPU bank."""
    pats = []; cols = []
    for rc in rcp:
        for w in range(R):
            if rc[w] is not None:
                pats.append(rc[w]); cols += [w] * len(rc[w])
    if not pats:
        return None, None
    return (torch.tensor(np.concatenate(pats, 0), device=device),
            torch.tensor(cols, device=device, dtype=torch.long))


def amp_fine_score(qcp, sh, bank, bank_cols, R, phase_tol, device):
    """1 - min best-cosine over patches in each query column (weakest-link)."""
    if bank is None:
        return np.full(R, np.nan)
    qs = []; qcol = []
    for w in range(R):
        if qcp[w] is not None:
            qs.append(qcp[w]); qcol += [w] * len(qcp[w])
    if not qs:
        return np.full(R, np.nan)
    Q = torch.tensor(np.concatenate(qs, 0), device=device)
    qc = torch.tensor(qcol, device=device, dtype=torch.long)
    S = Q @ bank.T
    c = (qc + sh) % R
    d = (bank_cols[None, :] - c[:, None]).abs()
    circ = torch.minimum(d, R - d)
    S = S.masked_fill(circ > phase_tol, -1e9)
    best = S.max(1).values.cpu().numpy()
    qcol = np.array(qcol); cs = np.full(R, np.nan)
    for w in range(R):
        m = qcol == w
        if m.any() and (best[m] > -1e8).all():
            cs[w] = 1 - best[m].min()
    return cs


# --------------------------------------------------------------------------- #
# amp_ng : n-gram column-window match
# --------------------------------------------------------------------------- #
def col_mean_arr(cm, R):
    """Column-mean list -> dense [R, D] array + [R] mask."""
    D = next((c.shape[0] for c in cm if c is not None), 1)
    arr = np.zeros((R, D), np.float32); msk = np.zeros(R, bool)
    for w in range(R):
        if cm[w] is not None:
            arr[w] = cm[w]; msk[w] = True
    return arr, msk


def amp_ng_score(qarr, qmsk, ref_arr, ref_msk, sh, ks, R, phase_tol, device,
                 causal=False, end_off=0):
    """For each window size k in ``ks`` return cs[R] = 1 - best window cosine.

    Window cosine = mean per-column cosine over the k-column window, matched to
    the same window in each reference with +/- ``phase_tol`` phase slide, max
    over references.  ``causal=True`` anchors the window to the past (AR-style):
    its right edge sits ``end_off`` columns from the current one, so the window
    spans [w-(k-1)+end_off, w+end_off].  ``end_off=0`` includes the current
    column at the right edge; ``end_off=-1`` excludes it (pure past predictors).
    Non-causal keeps the centred window.
    """
    Q = torch.tensor(qarr, device=device)
    Rt = torch.tensor(ref_arr, device=device)          # [N, R, D]
    cos3 = torch.einsum('ad,nbd->abn', Q, Rt)          # [a, b, N]
    qm = torch.tensor(qmsk, device=device)
    rm = torch.tensor(ref_msk, device=device)          # [N, R]
    valid = qm[:, None, None] & rm.T[None, :, :]
    cos3 = torch.where(valid, cos3, torch.tensor(-1e9, device=device))
    ww = torch.arange(R, device=device)
    out = {}
    for k in ks:
        joff = (torch.arange(k, device=device) - (k - 1) + end_off) if causal \
            else (torch.arange(k, device=device) - k // 2)
        acc = torch.full((R,), -1e18, device=device)
        for dd in range(-phase_tol, phase_tol + 1):
            qcols = (ww[:, None] + joff[None, :]) % R
            rcols = (ww[:, None] + sh + dd + joff[None, :]) % R
            g = cos3[qcols[:, :, None].expand(R, k, 1).reshape(R, k), rcols.reshape(R, k)]
            sc = g.mean(1)
            acc = torch.maximum(acc, sc.max(1).values)
        cs = torch.where(acc > -1e8, 1 - acc,
                         torch.tensor(float('nan'), device=device)).cpu().numpy()
        out[k] = cs
    return out


# --------------------------------------------------------------------------- #
# batched (multi-beat) variants -- numerically equal to the per-beat ones above
# --------------------------------------------------------------------------- #
def best_phase_shift_batch(Qarr, Qmsk, Narr, nv, R):
    """Vectorized ``best_phase_shift`` over a batch of beats sharing one profile.

    ``Qarr`` [B, R, D] / ``Qmsk`` [B, R] are the dense column means + masks of
    each query beat; ``Narr`` [R, D] / ``nv`` [R] the reference profile.  Returns
    an int array [B] of best circular shifts.  Computed in float64 (like the
    per-beat path, whose Q/N buffers are ``np.zeros`` float64) so the argmax is
    identical.
    """
    B = Qarr.shape[0]
    if B == 0:
        return np.zeros(0, int)
    Qf = Qarr.astype(np.float64)
    Nf = Narr.astype(np.float64)
    C = np.einsum('bwd,vd->bwv', Qf, Nf)          # [B, w, v]
    ar = np.arange(R)
    w2 = (ar[:, None] + ar[None, :]) % R          # [sh, w] = (w+sh)%R
    w_idx = np.broadcast_to(ar[None, :], (R, R))  # [sh, w] = w
    Csh = C[:, w_idx, w2]                          # [B, sh, w]
    validsh = Qmsk[:, w_idx] & nv[w2][None]        # [B, sh, w]
    cnt = validsh.sum(-1)                          # [B, sh]
    sums = np.where(validsh, Csh, 0.0).sum(-1)     # [B, sh]
    avg = np.where(cnt > 0, sums / np.maximum(cnt, 1), -1e9)
    return avg.argmax(1).astype(int)               # [B], first-occurrence tie-break


def amp_ng_score_batch(Qarr, Qmsk, ref_arr, ref_msk, sh, k, R, phase_tol, device,
                       causal=False, end_off=0, chunk=256):
    """Batched ``amp_ng_score`` for a single window size ``k``.

    ``Qarr`` [B, R, D] / ``Qmsk`` [B, R] the query column means + masks;
    ``ref_arr`` [N, R, D] / ``ref_msk`` [N, R] the reference bank (shared across
    the batch); ``sh`` [B] the per-beat circular shift.  Returns cs [B, R]
    (numpy, NaN where undefined) equal to stacking per-beat ``amp_ng_score``.
    """
    B = Qarr.shape[0]
    out = np.empty((B, R), np.float32)
    if B == 0:
        return out
    Rt = torch.tensor(ref_arr, device=device)              # [N, R, D]
    rm = torch.tensor(ref_msk, device=device)              # [N, R]
    rmT = rm.t().contiguous()                              # [c, N]
    Nn = rm.shape[0]
    ww = torch.arange(R, device=device)
    joff = (torch.arange(k, device=device) - (k - 1) + end_off) if causal \
        else (torch.arange(k, device=device) - k // 2)
    a_idx = (ww[:, None] + joff[None, :]) % R              # [R, k]
    neg = torch.tensor(-1e9, device=device)
    nan = torch.tensor(float('nan'), device=device)
    for i0 in range(0, B, chunk):
        i1 = min(B, i0 + chunk)
        b = i1 - i0
        Q = torch.tensor(Qarr[i0:i1], device=device)       # [b, R, D]
        qm = torch.tensor(Qmsk[i0:i1], device=device)      # [b, R]
        shb = torch.tensor(sh[i0:i1], device=device, dtype=torch.long)  # [b]
        cos4 = torch.einsum('bad,ncd->bacn', Q, Rt)        # [b, a, c, N]
        valid = qm[:, :, None, None] & rmT[None, None, :, :]
        cos4 = torch.where(valid, cos4, neg)
        cos4f = cos4.reshape(b, R * R, Nn)                 # [b, R*R, N]
        acc = torch.full((b, R), -1e18, device=device)
        for dd in range(-phase_tol, phase_tol + 1):
            c_idx = (ww[None, :, None] + shb[:, None, None] + dd + joff[None, None, :]) % R  # [b, R, k]
            flat = a_idx[None] * R + c_idx                 # [b, R, k]
            gidx = flat.reshape(b, R * k, 1).expand(b, R * k, Nn)
            g = torch.gather(cos4f, 1, gidx).reshape(b, R, k, Nn)
            sc = g.mean(2)                                 # [b, R, N]
            acc = torch.maximum(acc, sc.max(2).values)     # [b, R]
        cs = torch.where(acc > -1e8, 1 - acc, nan)
        out[i0:i1] = cs.cpu().numpy()
    return out


def amp_fine_score_batch(qcps, sh_list, bank, bank_cols, R, phase_tol, device,
                         chunk_patches=20000):
    """Batched ``amp_fine_score`` over beats sharing one bank.

    ``qcps`` is a list of per-beat column-patch lists (as ``col_patches``
    returns); ``sh_list`` the per-beat shifts.  Returns cs [B, R] (numpy)
    equal to stacking per-beat ``amp_fine_score``.
    """
    B = len(qcps)
    out = np.full((B, R), np.nan, np.float32)
    if bank is None or B == 0:
        return out
    pats = []; beat = []; worig = []; cshift = []
    for bi, qcp in enumerate(qcps):
        sh = int(sh_list[bi])
        for w in range(R):
            if qcp[w] is not None:
                p = qcp[w]; n = len(p)
                pats.append(p)
                beat.append(np.full(n, bi, np.int64))
                worig.append(np.full(n, w, np.int64))
                cshift.append(np.full(n, (w + sh) % R, np.int64))
    if not pats:
        return out
    Pq = np.concatenate(pats, 0)
    beat = np.concatenate(beat); worig = np.concatenate(worig); cshift = np.concatenate(cshift)
    best = np.empty(len(Pq), np.float32)
    for i0 in range(0, len(Pq), chunk_patches):
        i1 = min(len(Pq), i0 + chunk_patches)
        Q = torch.tensor(Pq[i0:i1], device=device)
        S = Q @ bank.T
        c = torch.tensor(cshift[i0:i1], device=device, dtype=torch.long)
        d = (bank_cols[None, :] - c[:, None]).abs()
        circ = torch.minimum(d, R - d)
        S = S.masked_fill(circ > phase_tol, -1e9)
        best[i0:i1] = S.max(1).values.cpu().numpy()
    key = beat * R + worig
    mn = np.full(B * R, np.inf, np.float64)
    np.minimum.at(mn, key, best.astype(np.float64))
    mn = mn.reshape(B, R)
    valid = (mn != np.inf) & (mn > -1e8)
    out[valid] = (1 - mn[valid]).astype(np.float32)
    return out


# --------------------------------------------------------------------------- #
# sh : SBD residual on the whole beat
# --------------------------------------------------------------------------- #
def sbd_shifts(sbd_len, sbd_margin):
    return np.array(list(range(0, sbd_margin + 1)) +
                    list(range(sbd_len - sbd_margin, sbd_len)))


def resample(a, n):
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)


def sbd_residual(q, ref_fft, ref_norm, sbd_len, shifts):
    """Per-point |q - best-shift(nearest reference)| under bounded circular shift."""
    fq = np.fft.rfft(q)
    cc = np.fft.irfft(fq[None, :] * np.conj(ref_fft), n=sbd_len, axis=1)
    ccs = cc[:, shifts]
    bc = ccs.max(1)
    bs = shifts[ccs.argmax(1)]
    i = int(np.argmax(bc))
    return np.abs(q - np.roll(ref_norm[i], bs[i]))


# --------------------------------------------------------------------------- #
# amplitude excursion (train-fixed vertical range)
# --------------------------------------------------------------------------- #
def excursion_cols(seg, center, half, R):
    """Per-column max distance from center over the train half-range."""
    L = len(seg); exc = np.zeros(R)
    for w in range(R):
        a = int(w * L / R); b = int((w + 1) * L / R); b = b if b > a else a + 1
        chunk = seg[a:min(b, L)]
        if len(chunk):
            exc[w] = max(abs(chunk.max() - center), abs(chunk.min() - center)) / (half + 1e-9)
    return exc


def apply_excursion(cs, exc, f, tau):
    """score = 1 - (1 - s_bnd) * exp(-max(0, exc - f) / tau).

    ``s_bnd`` is the column boundary score (nearest valid column if the column
    itself is empty).  Excursions beyond the frame (exc > f) push the score up.
    """
    R = len(cs)
    valid = ~np.isnan(cs)
    if not valid.any():
        return np.clip(1 - np.exp(-np.maximum(0, exc - f) / tau), 0, 1)
    idx = np.where(valid)[0]
    sbnd = np.empty(R)
    for w in range(R):
        sbnd[w] = cs[w] if valid[w] else cs[idx[np.argmin(np.abs(idx - w))]]
    over = np.maximum(0.0, exc - f)
    return 1 - (1 - sbnd) * np.exp(-over / tau)
