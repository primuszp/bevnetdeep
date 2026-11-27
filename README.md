# BevNetDeep

Code and reproducibility materials for *Vehicle Detection in Airborne LiDAR
Point Clouds Using a Multi-Channel Bird’s-Eye View Representation* by Péter
Primusz, Balázs Kisfaludi and Kornél Czimber, accepted in *Remote Sensing*.
The implementation retains the `bordereye.ml.bev` Python namespace.

For a worked CPU training example, inference on a LAS/LAZ survey, and a guide
to the output files, see [Training and inference](examples/README.md).
The guide is also available in [Hungarian (magyar)](examples/README.hu.md).

## Install and run a synthetic example

From the repository root in a Python 3.10 or newer environment:

```text
python -m pip install .
python examples/synthetic_inference.py
python scripts/summarize_counts.py
python scripts/summarize_counts.py --baselines
```

The example builds the ordered 12 BEV channels from a synthetic parking scene
and runs CPU inference, score calculation and rotated NMS. The default run
uses randomly initialised weights as a pipeline check.

To train a small model from scratch on generated data and use the saved weights:

```text
python -m bordereye.ml.bev.data.build_dataset --sources synth --out output/tutorial/data --n-train 4 --n-test 1 --tile-size 32 --veh-per-scene 2 4 --workers 1 --no-viz
bordereye-bev-train --config examples/training-cpu.json --dataset-dir output/tutorial/data --output-dir output/tutorial/training
python examples/synthetic_inference.py --weights output/tutorial/training/tutorial.pt
```

This one-epoch example exercises the training pipeline. For survey inference,
use the trained vehicle detector below.

## Trained weights

Clone the repository with [Git LFS](https://git-lfs.com/) installed and run
the following from the checkout:

```text
git lfs pull
python examples/synthetic_inference.py --weights models/bevnetdeep-vehicle-detection.pt
```

| File | Purpose |
|---|---|
| `models/bevnetdeep-vehicle-detection.pt` | Seed-123 vehicle detector used for the paper's detailed results. |
| `models/bevnetdeep-training-warm-start.pt` | Initial trained weights for the recorded training recipe. |

Each file contains the original model tensors and the architecture and
normalisation metadata needed to load them. See
[`models/checkpoints.json`](models/checkpoints.json) for SHA-256 checksums.
Load checkpoints only from trusted sources. The loader supports historical
PyTorch checkpoints that use Python pickle.

Replace `survey.laz` with the path to your metric LAS/LAZ point cloud:

```text
bordereye-bev-detect --input "survey.laz" --weights models/bevnetdeep-vehicle-detection.pt --output-dir output/survey --confidence 0.25 --decode-score-mode hm_seg_center_obb --nms-iou 0.20 --device cpu --no-amp
```

For this input, open `output/survey/survey_detections.png` for a visual check,
`survey_boxes.csv` for box coordinates and scores, or `survey_vehicles.geojson`
in a GIS. All three files are written to the same output directory.

## Data and evaluation

* **DALES Objects** includes the instance labels used for vehicle references.
  Request access through the University of Dayton's [download form](https://forms.gle/NFuLM6MrgEz8MmoV7).
  The [dataset page](https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/dales) describes both versions.
* **WiSAR3D** is distributed as a labels archive and four point-cloud parts.
  Download all five from the [Technion download page](https://cgmdata.ece.technion.ac.il/public/data/WiSAR3D/).

The paper's
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
bordereye-bev-evaluate --weights models/bevnetdeep-vehicle-detection.pt --dataset-dir path/to/bev --out-dir output/dales --source dales --split test --pattern "dales_*.npz" --box-target valid --conf 0.25 --nms-iou 0.20 --match-iou 0.30 --ap-iou-list 0.30,0.50 --decode-score-mode hm_seg_center_obb --no-amp
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
bordereye-bev-train --config configs/seed-123.json --dataset-dir path/to/bev --split-manifest splits/seed-123-train-val.json --resume models/bevnetdeep-training-warm-start.pt --output-dir output/seed-123
```

This command requires the original prepared training inputs. It initialises
model weights from the warm-start file and starts a fresh optimiser, as in the
recorded run. Use `--device cpu --no-amp` for local diagnostics and the
synthetic example for a quick installation check.

The source code is licensed under [Apache-2.0](LICENSE). Data and weights
have their own redistribution terms. The citation metadata is in
[`CITATION.cff`](CITATION.cff).
