# Prompt for the segmentation agent

Fill in this template and hand it to the AI agent when requesting cycle segmentation
+ two-stage clustering for a new domain. ABP and RESP were finalized this way. You
only need to fill in the `<...>` placeholders.

---

## Copy-paste prompt (fill in and hand over)

```
Add a new domain "<domain name = key in domains_dict.json, e.g. "Gait">" with the
same structure as the existing ABP/ and RESP/ (core.py + <id>.py + cluster_diag.py +
output/ + diagnostics/ + README.md). Follow the "Adding a new domain" procedure in
README.md.

[Data]
- Location: ./data/UCR_Anomaly_FullData/ *.txt (one value per line)
- Target series: all series belonging to "<domain name>" in domains_dict.json
- Training region: the leading train_end of the UCR filename suffix
  _<train_end>_<anom_s>_<anom_e> is normal
- Cycle unit: <what is one unit? e.g. one breath for respiration, one stride for
  gait; if unknown, estimate the dominant period by autocorrelation>
- Reference cluster count (visual guidance, not ground truth): <e.g. 076:1>. Pass
  it to the evaluation agent, and if it disagrees have it explain why (segmentation
  failure or a genuine sub-mode).

[Fixed — do not change]
- core.py / cluster_diag.py may be implemented from scratch for the domain (need not
  be a copy of ABP). But the two-stage clustering structure, operations, and metrics
  are fixed:
  Stage 1 = robust z-norm shape, full-shift SBD + complete linkage + silhouette-based
            relative K selection.
  Stage 2 = D3 (5 raw features) global standardize -> Mahalanobis whiten -> Ward +
            acceptance guards.
  Non-clean cycles -> artifact clusters by SBD + complete linkage. Comparisons are
  relative, not absolute-threshold.
- Singleton / small-cluster re-examination rule (required): after clusters are
  finalized, re-examine small (especially size-1) / isolated artifact clusters:
  (a) Same-shape recovery: if its medoid lies within the nearest pulse cluster's
      internal spread (p90 of medoid SBD) x a factor, treat it as a normal cycle
      flagged non-clean due to noise/distortion and merge it into the pulse cluster
      (relative comparison).
  (b) Re-split: if size 1 and length >= ~2 periods (an under-segmented multi-cycle
      span), re-split at the internal valley and merge each piece into the nearest
      cluster (independent artifact if none).
  (c) Keep independent: genuinely different shapes (partial cycle / doubled /
      singular) remain as independent clusters.
  -> reduces spurious singleton / small artifact clusters and recovers normal cycles
     into c0 etc.
- Phase alignment (segmentation quality, required): for each cycle, use the sharpest,
  most consistent landmark (trough, peak, or rising edge are all acceptable) as the
  reference, place the cut at the same-phase cycle boundary (a low point) every time,
  and contain one whole waveform in-phase within the segment. From several candidate
  anchors (trough / peakmin = valley between two peaks / upstroke = pre-rise trough /
  a series-specific sharp landmark if needed), automatically select the one that
  maximizes the phase-alignment SNR = range(mean shape) / mean per-point std. Avoid
  cutting at a peak apex (which splits the waveform in two) — a peak may serve as the
  "reference" but the cut goes at a low point. Clustering itself is phase-invariant via
  full-shift SBD, but aligning the phase sharpens the mean shape and makes
  segmentation errors easier to spot. For flat plateau waveforms lacking a sharp
  landmark (e.g. taichi square waves), full alignment is hard; specify a
  series-specific anchor (e.g. the single spike). Full coverage is always maintained
  (short fragments are merged into neighbors, not discarded).
- Sub-period fragment merging (segmentation quality): an anchor may insert an extra
  cut in a wide stationary / baseline region, creating a fragment segment shorter than
  half a period (< SHORT_SPAN_FRAC * period, default 0.45). Since this is part of a
  neighboring cycle, not an independent waveform, merge it into the adjacent segment
  (default: forward = into the next cycle; the trailing fragment only goes backward).
  Length guard required: do not merge in a direction that would exceed the clean length
  upper bound (LEN_HI_FRAC * period), to avoid padding a full cycle into an over-long,
  different-shape cluster. Full coverage is unchanged (only one cut is removed).

[Adjustable — per series or core default]
- PERIOD, RESAMPLE_LEN, peak detection / smoothing / length band / clean-filter
  thresholds, KMAX_S1/S2, MERGE_TAU, SIL_MIN, MIN_PATTERN, STAGE2_*, ARTIFACT_TAU, etc.
- Domain-wide phenomena (saturation clipping, whether a phase assumption holds, etc.)
  -> core defaults; series-specific -> that series' PARAMS in <id>.py.

[Required procedure]
1. First plot a few series to inspect morphology (unimodal / multimodal, symmetry,
   noise, presence of saturation clipping).
2. On a few representative series, produce segmentation -> two-stage clustering ->
   PDF + diagnostic figures.
3. Launch several read-only evaluation sub-agents in parallel to visually assess the
   figures:
   - Segmentation: is each segment one whole unit; any over/under-segmentation,
     partial-unit mixing, or leftover artifacts?
   - Clustering: are separate clusters genuinely distinct (not over-split)? Is any
     single cluster mixing different shapes (a missed under-split)? Judge whether a
     Stage-2 split is a genuine sub-mode or a threshold-driven spurious split, with
     mean-shape inspection taking priority. Also explain agreement / disagreement with
     the reference cluster count.
4. Adjust PARAMS or core defaults based on the feedback -> rerun -> re-evaluate.
   Repeat until "accept" is returned.
5. Apply to all series. Record the method, tuning rationale, and results in README.md.
```
