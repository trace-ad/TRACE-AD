"""Configuration for TFVision-AD.

The default values are the *confirmed* single setting used in the main
experiments.  Every field is exposed as a CLI option (see ``run.py``) so that
ablation studies can vary one knob at a time while everything else stays fixed.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Config:
    # --- CLIP backbone -------------------------------------------------------
    clip_model: str = "openai/clip-vit-base-patch16"
    batch_size: int = 128          # image encode batch (unified with baselines)

    # --- phase grid ----------------------------------------------------------
    grid_cols: int = 28            # R : number of phase bins a beat is split into
    phase_tol: int = 4             # DE: +/- per-column tolerance (delta). KEPT: each
                                   # column searches +/- phase_tol to absorb phase jitter.
    # Delta_q: single per-query global offset (best_phase_shift to the cluster profile).
    # RESTORED (2026-10-05): one circular shift per beat aligning the query's column
    # means to the cluster mean profile, applied before the per-column +/- delta search.
    # Recovers ABP/Accel/MITDB; UCR-250 flat 0.5197 -> 0.5205. False => sh forced to 0.
    global_align: bool = True

    # --- channels ------------------------------------------------------------
    # amp_fine : per-patch 1-NN (weakest-link) local shape
    # sh       : SBD global shape (amplitude-invariant)
    # amp_ng   : n-gram column-window shape (window size = ng_window)
    channels: Tuple[str, ...] = ("amp_fine", "sh", "amp_ng")
    ng_window: int = 7             # k : amp_ng column-window width (confirmed k=7)
    # amp_ng window anchoring: False = centred (+/-k//2), True = causal / AR-style
    # (current column at the RIGHT edge, window = current + (k-1) preceding cols).
    # Confirmed = centred (ng_causal=False); unifies UCR and TSB on one recipe.
    ng_causal: bool = False
    # optional width sweep: when non-empty, amp_ng is computed for every k here
    # and saved as amp_ng{k} (for picking a width offline); overrides ng_window.
    ng_sweep: Tuple[int, ...] = ()

    # --- image vertical range ------------------------------------------------
    # Beats are drawn on a train-only min-max range multiplied by vscale.
    # (the confirmed 't150' scheme uses 1.50 so normal amplitude variation stays
    #  inside the frame and only true amplitude excursions hit the top/bottom.)
    vscale: float = 1.50
    tau: float = 0.5               # soft-clip temperature for excursion penalty

    # --- SBD (sh channel) ----------------------------------------------------
    sbd_len: int = 128             # LN: length beats are resampled to for SBD
    sbd_margin: int = 48           # SM: max circular shift searched in SBD

    # --- reference selection -------------------------------------------------
    ref: str = "clean"             # which train beats form the memory bank
    edge_relax: bool = True        # widen the per-column tolerance to +/-R//2 on the
                                   # first/last test cycle (delta is kept, so this applies).

    # --- reference subsampling (N_ref ablation) ------------------------------
    # Cap the number of clean beats kept per cluster.  None keeps all.  If a
    # cluster has <= ref_count clean beats, all are kept ("use as many as exist").
    ref_count: Optional[int] = 20
    # How to pick ref_count beats within a cluster:
    #   None      : keep all (no subsampling)
    #   "random"  : random ref_count beats (seeded)
    #   "medoid"  : ref_count nearest to the cluster medoid (representative)
    #   "diverse" : farthest-point sampling from the medoid (coverage)
    ref_strategy: Optional[str] = "diverse"
    ref_seed: int = 0              # seed for ref_strategy="random"

    def __post_init__(self):
        if isinstance(self.channels, list):
            self.channels = tuple(self.channels)
