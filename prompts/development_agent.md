# Development-Agent Prompt (full)

The complete version of the paper Appendix "Development-Agent Instructions (Summary)".
Fill in the `<...>` placeholders and hand this to the development agent when requesting
cycle segmentation + two-stage clustering for a new domain.

```
Build a cycle segmentation + two-stage clustering program for domain <name>.
Learn the base waveform from the normal training region only; reuse the learned
clusters on test (assign each test cycle to the nearest training cluster medoid,
no reclustering of test).

[Data]
- Files: <path> (UCR_Anomaly_FullData/*.txt, one value per line).
- Series: all IDs listed under <name> in domains_dict.json.
- Training (normal) region: the leading `train_end` of each file
  (filename suffix _<train_end>_<anom_start>_<anom_end>).
- Cycle unit: <e.g. one breath / one heartbeat / one day; if unknown, estimate
  the dominant period by autocorrelation>.
- Reference cluster count (visual guidance, not ground truth): <e.g. 076:1>.

[Fixed — do not change]
- Two-stage clustering, structure/operations/metrics fixed:
  Stage 1 = robust z-norm shape, full-shift SBD, complete linkage, silhouette-based K.
  Stage 2 = D3 dynamics features (range, activity, curvature, #extrema, skew),
            global standardize -> Mahalanobis whiten -> Ward; accept a split only
            if clusters differ in shape (medoid full-SBD) or form a genuine
            amplitude sub-mode. Non-clean cycles -> artifact clusters by SBD.
  Comparisons are relative, not absolute-threshold.
- Phase-consistent boundaries at a cycle foot; never cut through peaks; full
  coverage; merge sub-period fragments into neighbors with a length guard.
- Re-examine singleton / small clusters: absorb into nearest normal cluster if
  within its internal SBD spread; re-split under-segmented multi-cycle spans;
  else keep independent.
- Per-cycle clean: a cycle is clean iff (cluster size >= CLEAN_MIN_N=2) and
  (SBD to cluster medoid <= CLEAN_SBD=0.40). Clean/artifact may coexist in a cluster.

[Adjustable — start from the supplied configuration]
- period estimation, peak detection, smoothing, cycle-length constraints,
  resampling length, K_max, merge / silhouette thresholds, Stage-2 guards.
- Domain-wide phenomena (saturation clipping, etc.) -> core defaults;
  series-specific -> that series' PARAMS. Record the final configuration.

[Procedure]
1. Plot a few series to inspect morphology.
2. Run segmentation + clustering on representative series and emit, per series,
   a segmentation figure and a cluster-diagnostic figure (top: raw-amplitude
   overlays; bottom: z-normalized shapes with cluster means).
3. Submit the figure paths to the evaluation agents; collect their feedback.
4. Revise the relevant settings or implementation; regenerate diagnostics; repeat
   until the evaluators report no issues.
5. Apply to all series. Record per-series PARAMS and the rationale.

[Test-time application]
Reuse the resulting segmentation program on test data; assign each query cycle to
the nearest established training-cluster medoid under SBD (no reclustering).
```
