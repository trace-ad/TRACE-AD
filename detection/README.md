# TFVision-AD

**Training-free vision (CLIP) anomaly detection for quasi-periodic time series.**

Each beat of a quasi-periodic signal (ECG, ABP, respiration, gait, …) is rendered
to an image and matched, patch by patch, against a **phase-aligned memory bank of
normal beats**. Anomalies are scored along three disentangled channels
(amplitude / period-length / shape). There is **no training and no LLM** — the
normal reference *is* the model.

The normal baseline is supplied structurally: beats are split, normal morphologies
are clustered, and the resulting phase-aligned reference set tells the detector
what "normal" looks like for *this* signal, which is what keeps false positives
down without a learned model.

## The three channels

| channel    | granularity        | what it sees                     | amplitude-sensitive |
|------------|--------------------|----------------------------------|---------------------|
| `amp_fine` | patch (point)      | local micro-deformations         | yes (via vertical axis) |
| `amp_ng`   | column window (k)  | shape of a short contiguous run  | yes (via vertical axis) |
| `sh`       | whole beat         | global shape (z-normalized, SBD) | no (shape only)     |

- **`amp_fine`** — per-patch 1-NN to the reference bank, **weakest-link** (min) over
  each phase column. Sensitive to local breaks (a distorted QRS, a notch, a spike).
- **`amp_ng`** — n-gram over `k` consecutive columns (column-mean granularity);
  robust to single-column phase noise. Fills the scale between `amp_fine` and `sh`.
- **`sh`** — SBD (shape-based distance) residual on the whole beat after length/phase
  normalization; amplitude-invariant global shape.

Channels are min-max normalized, summed, and smoothed by a one-beat moving average.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires a CLIP checkpoint (`openai/clip-vit-base-patch16`); it is downloaded on
first use (set `HF_HUB_OFFLINE=1` with a warm cache to run offline).

## Data layout

The detector consumes, per series:

- **raw signal** — UCR `*.txt` (`{idx}_UCR_Anomaly_{name}_{train_end}_{a0}_{a1}.txt`)
  or TSB-AD-U `*.csv` (columns `Data`,`Label`; `train_end` in the filename).
- **beat segmentation** — `{sid}_segments.csv` with columns
  `region` (train/test), `clean` (0/1), `cluster`, `start`, `end`.

Segmentations are produced by the pipeline under [`segmentation/`](../segmentation/)
(beat split + two-stage clustering). Run those programs to generate the
`*_segments.csv` files for each dataset.

## Usage

```bash
# UCR, confirmed config, save channel scores + per-stage timing + metrics
python run.py --dataset UCR \
    --data-dir /path/UCR_Anomaly_FullData \
    --seg-dir  /path/segments/ECG \
    --out-dir  out/ucr_ecg --eval

# TSB-AD subset
python run.py --dataset TSB \
    --data-dir /path/TSB-AD-U \
    --seg-dir  /path/segments/TSB_MITDB \
    --out-dir  out/tsb_mitdb --eval
```

Each series writes `{sid}.npz` (channel score arrays + labels + mean beat length),
`{sid}.timing.json` (per-stage seconds), and a `summary.json` over all series.

### Ablation options

Everything is a flag; defaults are the single confirmed setting
([`configs/default.yaml`](configs/default.yaml)).

| flag | default | meaning |
|------|---------|---------|
| `--grid-cols` | 28 | phase bins per beat (R) |
| `--phase-tol` | 4 | ± column phase tolerance (DE) |
| `--ng-window` | 7 | `amp_ng` window width (k) |
| `--vscale` | 1.50 | vertical range multiplier ("t150") |
| `--tau` | 0.5 | excursion soft-clip temperature |
| `--sbd-len` / `--sbd-margin` | 128 / 48 | SBD resample length / max shift |
| `--channels` | amp_fine,sh,amp_ng | subset of channels to use |
| `--ref` | clean | reference beats for the memory bank |
| `--no-edge-relax` | (off) | disable phase relaxation on first/last test beat |

Example — drop the shape channel:
`python run.py ... --channels amp_fine,amp_ng`

## Evaluation

`--eval` fuses the channels and reports point-wise (best-F1, AUC-PR, AUC-ROC,
precision, recall) and Range/VUS (R_AUC-ROC/PR, VUS-ROC/PR) metrics.

- Point-wise metrics are standard scikit-learn.
- Range/VUS come from the vendored upstream **VUS** code
  ([`evaluation/vus/`](evaluation/vus/); see `NOTICE`), called with
  `subsequence_length=None` so the tolerance window equals the mean labelled
  anomaly length — matching the settings used by the baselines, so numbers are
  directly comparable.

## Reproducibility

The refactor is byte-for-byte identical to the research code it was cleaned from
(verified: channel arrays and all nine metrics match with `max|diff| = 0`).
Results are deterministic given a fixed `*_segments.csv`, so ship/pin the exact
segmentation used for any reported table.

## Attribution

Range/VUS metrics are vendored from the VUS project (Paparrizos et al., PVLDB
2022); see `NOTICE`. All other code is MIT-licensed (`LICENSE`).
