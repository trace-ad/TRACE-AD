# Cluster-Count Resolution (LLM-assisted)

The substance of the footnote in the paper's key-parameters table:
"Ambiguous cluster counts are assessed with LLM assistance."
When the Stage-1 cluster count K_s1 is ambiguous, the evaluation agent decides it
by comparing the cluster mean shapes.

```
When the number of Stage-1 shape clusters K_s1 is ambiguous, decide it by
comparing the cluster mean shapes under the following rules:

- Prioritize medoid shape similarity over size balance, silhouette, or
  separability when choosing K_s1.
- Downgrade to K_s1 = 1 if the medoids satisfy SBD < 0.20 or Pearson r > 0.85.
- Do not choose K_s1 >= 2 on size balance or separability alone; require an
  explicit rationale for the shape difference (e.g. peak- vs valley-centered,
  amplitude, or a breakdown of correlation).
```
