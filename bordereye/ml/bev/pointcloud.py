"""Classify one LAS/LAZ cloud into ground/off-ground and run BEV inference.

The command preserves the proven Jeli inference path: it only owns input
validation, ground classification, optional classified-cloud exports, and a
machine-readable JSON result assembled from the BEV detector CSV.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from bordereye.io import read_las_data
from bordereye.ml.bev.data.paths import output_root
from bordereye.terrain import csf_ground_mask, ground_mask

from ..._masks import in_classes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    io = parser.add_argument_group("input/output")
    io.add_argument("--input", required=True, help="input .las or .laz point cloud")
    io.add_argument("--output-dir", default=str(output_root() / "pointcloud"))
    io.add_argument("--prefix", default="", help="output basename; default: input stem")
    io.add_argument(
        "--output-json",
        default="auto",
        help="JSON result path; 'auto' writes <output-dir>/<prefix>_detections.json, "
        "'-' prints JSON, 'none' disables it",
    )
    io.add_argument(
        "--output-geojson",
        default="auto",
        help="vehicle GeoJSON path; 'auto' writes beside the other outputs, 'none' disables it",
    )
    io.add_argument(
        "--output-shp",
        default="none",
        help="OBB shapefile path; 'auto' writes linked OBB and centre SHPs, 'none' disables",
    )
    io.add_argument(
        "--output-mask-ers",
        default="auto",
        help="binary segmentation mask ERS path; 'auto' writes beside the other outputs, "
        "'none' disables it",
    )
    io.add_argument(
        "--save-splits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="retain the ground and off-ground LAS/LAZ files",
    )
    io.add_argument("--ground-output", default="", help="explicit ground cloud output path")
    io.add_argument("--off-ground-output", default="", help="explicit off-ground output path")
    io.add_argument(
        "--classified-output",
        default="",
        help="optional complete cloud with ground=2 and off-ground=1 classifications",
    )
    io.add_argument("--split-extension", choices=("las", "laz"), default="las")
    io.add_argument("--overwrite", action="store_true", help="replace explicit cloud outputs")

    ground = parser.add_argument_group("ground classification")
    ground.add_argument(
        "--ground-method",
        choices=("auto", "las-class", "pmf", "csf"),
        default="auto",
        help="auto uses LAS ground classes when usable, otherwise PMF",
    )
    ground.add_argument("--ground-classes", default="2", help="comma-separated LAS ground classes")
    ground.add_argument("--ground-cell", type=float, default=1.0, help="PMF raster cell [m]")
    ground.add_argument("--ground-max-window", type=float, default=20.0, help="PMF window [m]")
    ground.add_argument("--ground-slope", type=float, default=0.15)
    ground.add_argument("--ground-initial-distance", type=float, default=0.15)
    ground.add_argument("--ground-max-distance", type=float, default=2.5)
    ground.add_argument("--ground-threshold", type=float, default=0.25)
    ground.add_argument("--below-tolerance", type=float, default=0.10)
    ground.add_argument("--terrain-tile-size", type=float, default=250.0)
    ground.add_argument("--csf-exe", default="", help="required with --ground-method csf")
    ground.add_argument("--csf-cloth-resolution", type=float, default=0.5)
    ground.add_argument("--csf-rigidness", type=int, default=None)
    ground.add_argument("--csf-slope-smooth", action="store_true")
    ground.add_argument("--minimum-ground-points", type=int, default=10)

    bev = parser.add_argument_group("BEV inference")
    bev.add_argument(
        "--weights", required=True,
        help="trained BEV weights file or retained alias (v19/v20)",
    )
    bev.add_argument("--cell", default="auto", help="'auto' uses the weights file's cell")
    bev.add_argument(
        "--features", choices=("auto", "generic", "generic_v2", "legacy"), default="auto"
    )
    bev.add_argument("--confidence", type=float, default=0.30)
    bev.add_argument("--mask-threshold", type=float, default=0.5)
    bev.add_argument(
        "--crs-name",
        default="auto",
        help="Input CRS for EOV export; 'auto' reads LAS/LAZ metadata",
    )
    bev.add_argument("--ers-datum", default="auto")
    bev.add_argument("--ers-projection", default="auto")
    bev.add_argument("--nms-iou", type=float, default=0.20)
    bev.add_argument(
        "--decode-score-mode",
        choices=("hm", "hm_seg_center", "hm_seg_obb", "hm_seg_center_obb"),
        default="hm",
    )
    bev.add_argument("--window", type=int, default=512)
    bev.add_argument("--overlap", type=int, default=64)
    bev.add_argument(
        "--infer-batch-size",
        type=int,
        default=0,
        help="sliding windows per model call; 0 selects a conservative device-aware value",
    )
    bev.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    bev.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    bev.add_argument("--chunk-size", type=int, default=2_000_000)
    bev.add_argument("--bn-adapt", action="store_true")
    bev.add_argument("--bn-passes", type=int, default=10)
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _output_argument(value: str) -> str:
    return value if value.lower() in {"auto", "none"} else str(_path(value))


def _reported_ers_path(value: str, output_dir: Path, prefix: str) -> str | None:
    if value.lower() == "none":
        return None
    path = output_dir / f"{prefix}_mask.ers" if value.lower() == "auto" else _path(value)
    return str(path if path.suffix.lower() == ".ers" else Path(str(path) + ".ers"))


def _reported_shapefiles(
    value: str, output_dir: Path, prefix: str
) -> dict[str, str] | None:
    if value.lower() == "none":
        return None
    obb = output_dir / f"{prefix}_vehicle_obbs.shp" if value.lower() == "auto" else _path(value)
    if obb.suffix.lower() != ".shp":
        obb = obb.with_suffix(".shp")
    centers = obb.with_name(f"{obb.stem}_centers.shp")
    return {"obb": str(obb), "centers": str(centers)}


def _ground_classes(value: str) -> set[int]:
    try:
        classes = {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as exc:
        raise ValueError("--ground-classes must contain comma-separated integers") from exc
    if not classes or min(classes) < 0 or max(classes) > 255:
        raise ValueError("--ground-classes must contain values in [0, 255]")
    return classes


def classify_ground(input_path: Path, xyz: np.ndarray, classification: np.ndarray, args):
    """Return ``(mask, effective_method)`` for one validated point cloud."""
    classes = _ground_classes(args.ground_classes)
    class_mask = in_classes(classification, classes)
    method = args.ground_method
    if method == "auto":
        usable = (
            int(class_mask.sum()) >= args.minimum_ground_points and int((~class_mask).sum()) > 0
        )
        method = "las-class" if usable else "pmf"
    if method == "las-class":
        mask = class_mask
    elif method == "pmf":
        mask = ground_mask(
            input_path,
            args.ground_cell,
            max_window_size=args.ground_max_window,
            slope=args.ground_slope,
            initial_distance=args.ground_initial_distance,
            max_distance=args.ground_max_distance,
            classify_threshold=args.ground_threshold,
            below_tolerance=args.below_tolerance,
            chunk_size=args.chunk_size,
            tile_size=args.terrain_tile_size,
        )
    else:
        if not args.csf_exe:
            raise ValueError("--ground-method csf requires --csf-exe")
        mask = csf_ground_mask(
            xyz,
            _path(args.csf_exe),
            cloth_resolution=args.csf_cloth_resolution,
            classify_threshold=args.ground_threshold,
            b_slope_smooth=args.csf_slope_smooth,
            rigidness=args.csf_rigidness,
        )
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(xyz),):
        raise RuntimeError(f"ground classifier returned shape {mask.shape}, expected {(len(xyz),)}")
    if int(mask.sum()) < args.minimum_ground_points:
        raise RuntimeError(
            f"only {int(mask.sum())} ground points found; lower the thresholds or choose another method"
        )
    if not (~mask).any():
        raise RuntimeError("ground classification produced no off-ground points")
    return mask, method


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError(f"point-cloud output must end in .las or .laz: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_cloud(source, path: Path, mask: np.ndarray | None, overwrite: bool) -> None:
    _prepare_output(path, overwrite)
    output = source.__class__(copy.deepcopy(source.header))
    selected = source.points if mask is None else source.points[mask]
    output.points = selected.copy()
    output.write(str(path))


def _write_classified(source, path: Path, mask: np.ndarray, overwrite: bool) -> None:
    _prepare_output(path, overwrite)
    output = copy.deepcopy(source)
    output.classification = np.where(mask, 2, 1).astype(np.uint8)
    output.write(str(path))


def _read_detections(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {name: float(value) for name, value in row.items()} for row in csv.DictReader(stream)
        ]


def _write_json(value: str, payload: dict[str, Any], output_dir: Path, prefix: str) -> None:
    if value.lower() == "none":
        return
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if value == "-":
        print(text, end="")
        return
    path = output_dir / f"{prefix}_detections.json" if value == "auto" else _path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  -> {path}")


def _run_jeli(args, ground_path: Path, off_ground_path: Path, output_dir: Path, prefix: str):
    from bordereye.ml.bev.jeli import main as jeli_main

    forwarded = [
        "--ground",
        str(ground_path),
        "--off-ground",
        str(off_ground_path),
        "--weights",
        str(_path(args.weights)),
        "--out",
        str(output_dir),
        "--output-prefix",
        prefix,
        "--output-geojson",
        _output_argument(args.output_geojson),
        "--output-shp",
        _output_argument(args.output_shp),
        "--output-mask-ers",
        _output_argument(args.output_mask_ers),
        "--crs-name",
        args.crs_name,
        "--mask-threshold",
        str(args.mask_threshold),
        "--ers-datum",
        args.ers_datum,
        "--ers-projection",
        args.ers_projection,
        "--cell",
        str(args.cell),
        "--features",
        args.features,
        "--conf",
        str(args.confidence),
        "--nms-iou",
        str(args.nms_iou),
        "--decode-score-mode",
        args.decode_score_mode,
        "--win",
        str(args.window),
        "--overlap",
        str(args.overlap),
        "--infer-batch-size",
        str(args.infer_batch_size),
        "--device",
        args.device,
        "--chunk",
        str(args.chunk_size),
        "--bn-passes",
        str(args.bn_passes),
        "--amp" if args.amp else "--no-amp",
    ]
    if args.bn_adapt:
        forwarded.append("--bn-adapt")
    jeli_main(forwarded)


def main(argv=None) -> None:
    args = parse_args(argv)
    input_path = _path(args.input)
    if input_path.suffix.lower() not in {".las", ".laz"} or not input_path.is_file():
        raise FileNotFoundError(f"input LAS/LAZ not found: {input_path}")
    weights = _path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"BEV weights not found: {weights}")
    if args.chunk_size <= 0 or args.minimum_ground_points <= 0:
        raise ValueError("--chunk-size and --minimum-ground-points must be positive")

    output_dir = _path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix.strip() or input_path.stem
    if Path(prefix).name != prefix:
        raise ValueError("--prefix must be a filename basename")

    print(f"[1/3] reading + classifying: {input_path}")
    source = read_las_data(input_path)
    xyz = np.column_stack((source.x, source.y, source.z)).astype(np.float64)
    classification = np.asarray(source.classification, dtype=np.int32)
    mask, method = classify_ground(input_path, xyz, classification, args)
    n_ground, n_off = int(mask.sum()), int((~mask).sum())
    print(f"  method={method} points={len(xyz):,} ground={n_ground:,} off-ground={n_off:,}")

    with tempfile.TemporaryDirectory(prefix="bordereye-bev-") as temporary:
        temp = Path(temporary)
        extension = args.split_extension
        ground_path = (
            _path(args.ground_output)
            if args.ground_output
            else output_dir / f"{prefix}_ground.{extension}"
            if args.save_splits
            else temp / f"ground.{extension}"
        )
        off_ground_path = (
            _path(args.off_ground_output)
            if args.off_ground_output
            else output_dir / f"{prefix}_off_ground.{extension}"
            if args.save_splits
            else temp / f"off_ground.{extension}"
        )
        cloud_outputs = [ground_path, off_ground_path]
        if args.classified_output:
            cloud_outputs.append(_path(args.classified_output))
        if input_path in cloud_outputs:
            raise ValueError("cloud outputs must not overwrite --input")
        if len(set(cloud_outputs)) != len(cloud_outputs):
            raise ValueError("ground, off-ground, and classified outputs must be distinct")
        if (
            args.output_json.lower() not in {"auto", "none", "-"}
            and _path(args.output_json) == input_path
        ):
            raise ValueError("--output-json must not overwrite --input")
        print("[2/3] writing classified subsets ...")
        _write_cloud(source, ground_path, mask, args.overwrite)
        _write_cloud(source, off_ground_path, ~mask, args.overwrite)
        if args.classified_output:
            _write_classified(source, _path(args.classified_output), mask, args.overwrite)

        print("[3/3] running BEV inference ...")
        _run_jeli(args, ground_path, off_ground_path, output_dir, prefix)

    csv_path = output_dir / f"{prefix}_boxes.csv"
    detections = _read_detections(csv_path)
    payload = {
        "input": str(input_path),
        "weights": str(weights),
        "ground_method": method,
        "point_counts": {"total": len(xyz), "ground": n_ground, "off_ground": n_off},
        "detection_count": len(detections),
        "detections": detections,
        "csv": str(csv_path),
        "geojson": (
            str(output_dir / f"{prefix}_vehicles.geojson")
            if args.output_geojson.lower() == "auto"
            else None if args.output_geojson.lower() == "none" else str(_path(args.output_geojson))
        ),
        "shapefiles": _reported_shapefiles(args.output_shp, output_dir, prefix),
        "mask_ers": _reported_ers_path(args.output_mask_ers, output_dir, prefix),
    }
    _write_json(args.output_json, payload, output_dir, prefix)


if __name__ == "__main__":
    main()
