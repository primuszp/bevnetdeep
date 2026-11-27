"""Audit spatial separation of the DALES BEV train/test partition.

The BEV NPZ files use local raster coordinates, so their ``origin`` cannot
establish separation between DALES source tiles. This script instead reads
the metric bounds from every raw DALES LAS header and reports the closest
train/test pair as axis-aligned source-tile footprints. A zero separation
means the source footprints touch or overlap; it is not evidence of direct
file reuse.

Run on the exact frozen dataset used for the paper, for example:

    python scripts/bev/audit_dales_split.py --dataset-dir data/bev --out-dir output/paper_v19_detection/dales_split_audit

The JSON and CSV outputs are intended to be archived with the manuscript
evidence.  Do not infer spatial independence from directory names alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import laspy
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True,
                        help="BEV dataset root; used to locate the default LAS root")
    parser.add_argument("--las-root", default="",
                        help="raw DALES root containing train/ and test/ LAS files")
    parser.add_argument("--bounds", choices=("las_header", "filename"), default="las_header",
                        help="use raw LAS header bounds, or the DALES filename grid coordinates")
    parser.add_argument("--name-unit-m", type=float, default=100.0,
                        help="metres represented by one coordinate unit in a DALES filename")
    parser.add_argument("--tile-width-m", type=float, default=500.0,
                        help="nominal side length of a DALES source tile in filename mode")
    parser.add_argument("--out-dir", required=True,
                        help="directory for dales_split_audit.{json,csv}")
    return parser.parse_args()


def footprint_from_las_header(path: Path, split: str) -> dict[str, object]:
    """Read a DALES source tile's two-dimensional metric extent from its LAS header."""
    with laspy.open(path) as reader:
        mins = np.asarray(reader.header.mins, dtype=np.float64)
        maxs = np.asarray(reader.header.maxs, dtype=np.float64)
    x0, y0 = map(float, mins[:2])
    x1, y1 = map(float, maxs[:2])
    if not x1 > x0 or not y1 > y0:
        raise ValueError(f"{path} has invalid LAS bounds")
    return {
        "split": split,
        "tile": path.stem,
        "path": str(path),
        "x_min_m": x0,
        "y_min_m": y0,
        "x_max_m": x1,
        "y_max_m": y1,
        "area_m2": (x1 - x0) * (y1 - y0),
    }


def footprint_from_filename(path: Path, split: str, name_unit_m: float,
                            tile_width_m: float) -> dict[str, object]:
    """Decode DALES' ``<easting>_<northing>.las`` source-tile grid name."""
    match = re.fullmatch(r"(\d+)_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"cannot decode DALES grid coordinates from {path.name}")
    if name_unit_m <= 0 or tile_width_m <= 0:
        raise ValueError("filename-grid unit and tile width must be positive")
    x0 = float(match.group(1)) * name_unit_m
    y0 = float(match.group(2)) * name_unit_m
    return {
        "split": split,
        "tile": path.stem,
        "path": str(path),
        "x_min_m": x0,
        "y_min_m": y0,
        "x_max_m": x0 + tile_width_m,
        "y_max_m": y0 + tile_width_m,
        "area_m2": tile_width_m * tile_width_m,
    }


def gap_m(a: dict[str, object], b: dict[str, object]) -> float:
    """Euclidean gap between two closed, axis-aligned raster footprints."""
    dx = max(float(a["x_min_m"]) - float(b["x_max_m"]),
             float(b["x_min_m"]) - float(a["x_max_m"]), 0.0)
    dy = max(float(a["y_min_m"]) - float(b["y_max_m"]),
             float(b["y_min_m"]) - float(a["y_max_m"]), 0.0)
    return float(np.hypot(dx, dy))


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset_dir)
    las_root = Path(args.las_root) if args.las_root else dataset.parent.parent / "sources" / "DALES"
    out = Path(args.out_dir)
    groups: dict[str, list[dict[str, object]]] = {"train": [], "test": []}
    for split in groups:
        directory = las_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"missing split directory: {directory}")
        for path in sorted(directory.glob("*.las")):
            if args.bounds == "las_header":
                groups[split].append(footprint_from_las_header(path, split))
            else:
                groups[split].append(footprint_from_filename(
                    path, split, args.name_unit_m, args.tile_width_m))
    if not groups["train"] or not groups["test"]:
        raise ValueError("no DALES tiles found in both train and test splits")

    pairs = []
    for train in groups["train"]:
        for test in groups["test"]:
            pairs.append({
                "train_tile": train["tile"],
                "test_tile": test["tile"],
                "gap_m": gap_m(train, test),
            })
    pairs.sort(key=lambda item: (float(item["gap_m"]), str(item["train_tile"]), str(item["test_tile"])))
    closest = pairs[0]
    report = {
        "method": (
            "axis-aligned raw-DALES-LAS footprint separation from header metric bounds"
            if args.bounds == "las_header" else
            "axis-aligned DALES source-tile separation from filename grid coordinates"
        ),
        "las_root": str(las_root),
        "bounds_mode": args.bounds,
        "filename_grid_unit_m": args.name_unit_m if args.bounds == "filename" else None,
        "nominal_tile_width_m": args.tile_width_m if args.bounds == "filename" else None,
        "train_tiles": len(groups["train"]),
        "test_tiles": len(groups["test"]),
        "min_train_test_gap_m": closest["gap_m"],
        "closest_train_tile": closest["train_tile"],
        "closest_test_tile": closest["test_tile"],
        "interpretation": (
            "A positive gap quantifies raster-footprint separation. A zero gap means "
            "the stored footprints touch or overlap; neither result alone establishes "
            "independence of acquisition conditions or scenes."
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "dales_split_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (out / "dales_split_audit_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["train_tile", "test_tile", "gap_m"])
        writer.writeheader()
        writer.writerows(pairs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
