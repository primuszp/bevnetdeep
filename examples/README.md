# BevNetDeep training and inference

**English** · [Magyar](README.hu.md)

To detect vehicles in an airborne LiDAR survey, start with the trained detector and the LAS/LAZ example below. To practise training a model, follow the CPU tutorial. It generates its own labelled scenes, trains a small network and reloads the resulting weights.

## Installation

Open a terminal in the repository root, alongside `pyproject.toml` and `models/`. Activate a Python environment (the examples were tested with Python 3.12), then run

```text
python -m pip install .
git lfs pull
```

The first command installs the package and its `bordereye-bev-*` commands. The second downloads the two checkpoints tracked by Git LFS. Run the remaining commands from the same directory and environment.

## Detect vehicles in a LAS/LAZ survey

Replace the example input path with your point cloud. Its coordinates should be in metres, with the correct coordinate reference system in the file header for use in a GIS.

```text
bordereye-bev-detect --input "D:/LiDAR/parking.laz" --weights models/bevnetdeep-vehicle-detection.pt --output-dir output/parking --device cpu --no-amp --confidence 0.25 --decode-score-mode hm_seg_center_obb --nms-iou 0.20
```

By default, the detector uses LAS class 2 when the cloud contains enough ground points and at least one point outside that class. Otherwise it applies a morphological ground filter. Heights above ground and the spatial distribution of returns provide the model's 12 bird's-eye view (BEV) channels. The `--confidence` option sets the minimum detection score. The `--nms-iou` option controls when overlapping rotated boxes are suppressed.

| File in `output/parking/` | What to use it for |
|---|---|
| `parking_detections.png` | Check where the detector placed the boxes. |
| `parking_boxes.csv` | Analyse individual detections. Each row gives the centre (`cx`, `cy`), length and width in metres (`L`, `W`), orientation in degrees (`theta_deg`) and detection score (`score`). |
| `parking_vehicles.geojson` | Inspect vehicle footprints on a map, for example in QGIS. |
| `parking_detections.json` | Read the detections, point counts, ground classification method and output paths from another program. |
| `parking_mask.ers` and its binary raster | Inspect the segmentation mask. Keep the header and raster together. |

To try the trained model before supplying a survey, run it on a synthetic parking scene. The script prints the detection count in the terminal.

```text
python examples/synthetic_inference.py --weights models/bevnetdeep-vehicle-detection.pt
```

## Train a small model on CPU

**1. Generate labelled scenes.** This creates four `.npz` tiles in `output/tutorial/data/train/` and one held-out tile in `output/tutorial/data/test/`.

```text
python -m bordereye.ml.bev.data.build_dataset --sources synth --out output/tutorial/data --n-train 4 --n-test 1 --tile-size 32 --veh-per-scene 2 4 --workers 1 --no-viz
```

**2. Train for one epoch.** The example uses three tiles for training and one for validation. It samples 64 × 64 pixel crops in batches of two. The separate test tile is available for later evaluation. [CPU training configuration](training-cpu.json)

```text
bordereye-bev-train --config examples/training-cpu.json --dataset-dir output/tutorial/data --output-dir output/tutorial/training
```

**3. Run inference with the weights you just trained.**

```text
python examples/synthetic_inference.py --weights output/tutorial/training/tutorial.pt
```

In `output/tutorial/training/`, `tutorial.pt` holds the model selected on validation and `tutorial_last.pt` holds the final epoch. Read `train_log.csv` for per-epoch metrics or open `training_curves.png` to inspect the curves. The one-epoch run demonstrates the workflow. For survey inference, use the research checkpoint from the LAS/LAZ example.

## Files used in the paper

| Repository path | Purpose |
|---|---|
| `models/bevnetdeep-vehicle-detection.pt` | Trained detector used for the paper's detailed evaluation. Pass it to `--weights` for inference. |
| `models/bevnetdeep-training-warm-start.pt` | Initial weights for the reported training run. Pass them to `--resume` with the seed-123 recipe. |
| `configs/seed-123.json` | Full training configuration, including the CUDA device, learning rate and source sampling weights. |
| `splits/seed-123-train-val.json` | Tile filenames for the fixed training and validation split. |
| `splits/test-tiles.json`, `results/` | Test tile lists and archived tables for the reported evaluation. |
| `examples/synthetic_inference.py` | Python example showing how to build the BEV channels, load the model and call `BevPredictor.predict()`. |

For the full training recipe, point `--dataset-dir` to the parent of `train/` and `test/` in the prepared dataset. It must contain the synthetic, sparse synthetic, DALES and WiSAR3D tiles used in the experiment. Reproducing the split and targets requires the derived vehicle boxes, ignored boxes and WiSAR3D mosaics used in the study. [Dataset downloads and reproducibility notes](../README.md#data-and-evaluation)

```text
bordereye-bev-train --config configs/seed-123.json --dataset-dir "D:/LiDAR/prepared-bev" --split-manifest splits/seed-123-train-val.json --resume models/bevnetdeep-training-warm-start.pt --output-dir output/seed-123
```

The recipe starts from the warm-start weights with a fresh optimiser. After training, pass `output/seed-123/best_model_consistency.pt` to the detector's `--weights` option to use the validation-selected model on a survey.
