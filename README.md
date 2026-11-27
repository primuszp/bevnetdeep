# BevNetDeep

Code and reproducibility materials for *Vehicle Detection in Airborne LiDAR
Point Clouds Using a Multi-Channel Bird’s-Eye View Representation* by Péter
Primusz, Balázs Kisfaludi and Kornél Czimber, accepted in *Remote Sensing*.
The implementation retains the `bordereye.ml.bev` Python namespace.

## Install and run a synthetic example

From the repository root in a Python 3.10 or newer environment:

```text
python -m pip install .
python examples/synthetic_inference.py
python scripts/summarize_counts.py
python scripts/summarize_counts.py --baselines
```

The example builds the ordered 12 BEV channels from a synthetic parking scene
and runs CPU inference, score calculation and rotated NMS. It uses randomly
initialised weights by default and checks the pipeline rather than detection
accuracy. With the retained checkpoint available, pass `--weights
/path/to/bev_v19_final.pt` to run the same scene with the trained model.
The checkpoint identifier and SHA-256 are in
[`models/retained-checkpoint.json`](models/retained-checkpoint.json).

## Data and evaluation

Obtain [DALES Objects](https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales)
and [WiSAR3D](https://wisar3d.github.io/) from their providers. The paper's
evaluator expects preprocessed BEV `.npz` tiles under `train/` and `test/`,
including the paper-specific valid and ignored vehicle boxes. The original
datasets alone do not recover the edited reference boxes, generated ignore
boxes or the exact WiSAR3D mosaics. The raw Jeli survey is private.

[`splits/seed-123-train-val.json`](splits/seed-123-train-val.json) records the
checkpoint's training and validation tile basenames. The test manifests contain
11 DALES tiles and 42 WiSAR3D mosaics. WiSAR3D mosaic numbers are local to each
preprocessed split, so the same basename can appear under both `train/` and
`test/`. The archived test mosaics lack the source-frame identifiers needed to
independently verify separation at the source-frame level.

With the prepared dataset and trained weights available, run the evaluator as
follows. For WiSAR3D, use `--source wisar --pattern "wisar_m*.npz"` and a
distinct output directory.

```text
bordereye-bev-evaluate --weights path/to/bev_v19_final.pt --dataset-dir path/to/bev --out-dir output/dales --source dales --split test --pattern "dales_*.npz" --box-target valid --conf 0.25 --nms-iou 0.20 --match-iou 0.30 --ap-iou-list 0.30,0.50 --decode-score-mode hm_seg_center_obb --no-amp
```

The included CSVs support offline verification of the retained seed-123 TP,
FP and FN counts and the published DALES segmentation-baseline counts. The
three-run mean and sample standard deviation table is transcribed from the
paper. Ranked AP and individual seed results cannot be recomputed from the
aggregate table. For the rule-based baseline and uncertainty analysis, see
[`scripts/bev`](scripts/bev).

## Training recipe

[`configs/seed-123.json`](configs/seed-123.json) records the retained run's
training options. [`configs/benchmark-protocol.json`](configs/benchmark-protocol.json)
records the shared settings for seeds 42, 123 and 2026. Complete per-run recipes
and checkpoints for the other two historical runs are unavailable in the
archived materials used for this release.

```text
bordereye-bev-train --config configs/seed-123.json --dataset-dir path/to/bev --split-manifest splits/seed-123-train-val.json --resume path/to/warm-start.pt --output-dir output/seed-123
```

This command requires the original prepared training inputs and warm-start
weights to replay the recorded run. Use `--device cpu --no-amp` for local
diagnostics and the synthetic example for a quick installation check.

The source code is licensed under [Apache-2.0](LICENSE). Data and weights
have their own redistribution terms. The citation metadata is in
[`CITATION.cff`](CITATION.cff).
