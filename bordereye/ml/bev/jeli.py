"""Run supplied BEV weights on the Jeli ground/off-ground LAS pair.

The command derives the weights' channel schema, runs sliding-window OBB
and segmentation inference, and writes compact PNG/CSV diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import laspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPoly

from bordereye import RasterGrid
from bordereye.ml.bev.core.inference import STRIDE
from bordereye.ml.bev.core.normalization import validate_norm_config
from bordereye.ml.bev.data import bev_dataset as DSM
from bordereye.ml.bev.data.paths import output_root, project_root
from bordereye.ml.bev.outputs import (
    write_detection_shapefiles,
    write_detections_geojson,
    write_segmentation_mask_ers,
)
from bordereye.ml.bev.predict import BevPredictor
from bordereye.ml.models.checkpoints import resolve_device
from bordereye.paths import data_file
from bordereye.raster_ops import fill_nodata_nn
from bordereye.terrain import height_above_ground


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground", default="jeli_ground_mini.las")
    p.add_argument("--off-ground", default="jeli_cars_off_ground.las")
    p.add_argument(
        "--weights", required=True,
        help="trained BEV weights file or checkpoint ID (e.g. paper-model)",
    )
    p.add_argument("--out", default=str(output_root() / "jeli"))
    p.add_argument("--output-prefix", default="jeli_bev", help="basename used for PNG/CSV outputs")
    p.add_argument(
        "--output-geojson",
        default="auto",
        help="vehicle GeoJSON path; 'auto' writes <out>/<prefix>_vehicles.geojson, "
        "'none' disables it",
    )
    p.add_argument(
        "--output-shp",
        default="none",
        help="OBB shapefile path; 'auto' writes OBB and centre SHPs below <out>, "
        "'none' disables them",
    )
    p.add_argument(
        "--output-mask-ers",
        default="auto",
        help="binary mask ERS path; 'auto' writes <out>/<prefix>_mask.ers, 'none' disables it",
    )
    p.add_argument(
        "--crs-name",
        default="auto",
        help="Input CRS for EOV export; 'auto' reads LAS/LAZ metadata",
    )
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument(
        "--ers-datum",
        default="auto",
        help="ERS datum; auto maps EPSG:23700 to HD72 and otherwise uses RAW",
    )
    p.add_argument(
        "--ers-projection",
        default="auto",
        help="ERS projection; auto maps EPSG:23700 to EOV and otherwise uses RAW",
    )
    p.add_argument("--cell", default="auto", help="'auto' uses the weights file's cell")
    p.add_argument(
        "--features", default="auto", choices=("auto", "generic", "generic_v2", "legacy")
    )
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument(
        "--ablate-channels",
        default="",
        help="comma-separated channel names to mean-ablate (replace "
        "with the channel's mean) — tests which channels suppress "
        "a detection, e.g. an under-canopy vehicle",
    )
    p.add_argument(
        "--probe-xy",
        default="",
        help="'x,y' world coords: save a per-channel + heatmap + seg "
        "panel cropped around this point (diagnose why a spot is "
        "missed)",
    )
    p.add_argument("--probe-radius-m", type=float, default=12.0)
    p.add_argument(
        "--save-channels",
        default="",
        help="comma-separated channel names to dump as "
        "jeli_channels.npz (with origin/cell) for offline analysis",
    )
    p.add_argument(
        "--decode-score-mode",
        choices=("hm", "hm_seg_center", "hm_seg_obb", "hm_seg_center_obb"),
        default="hm",
    )
    p.add_argument("--nms-iou", type=float, default=0.20)
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument(
        "--infer-batch-size",
        type=int,
        default=0,
        help="sliding windows per model call; 0 selects a conservative device-aware value",
    )
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use CUDA autocast during sliding-window inference",
    )
    p.add_argument("--chunk", type=int, default=4_000_000)
    p.add_argument("--bn-adapt", action="store_true", default=False)
    p.add_argument("--bn-passes", type=int, default=10)
    return p.parse_args(argv)


def _resolve(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    workspace = project_root() / p
    if workspace.exists():
        return workspace
    # A bare measured-input name resolves through the data/ layout.
    return data_file(path_text)


def _output_target(value: str, out_dir: Path, automatic_name: str) -> Path | None:
    if value.lower() == "none":
        return None
    return out_dir / automatic_name if value.lower() == "auto" else _resolve(value)


def _crs_name(value: str, las_path: Path) -> str | None:
    if value.lower() == "none":
        return None
    if value.lower() != "auto":
        return value
    try:
        with laspy.open(str(las_path)) as reader:
            crs = reader.header.parse_crs()
        if crs is None:
            return None
        authority = crs.to_authority()
        return f"{authority[0]}:{authority[1]}" if authority else crs.to_string()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return None


def _crs_wkt(value: str, las_path: Path) -> str | None:
    if value.lower() == "none":
        return None
    try:
        if value.lower() == "auto":
            with laspy.open(str(las_path)) as reader:
                crs = reader.header.parse_crs()
        else:
            from pyproj import CRS

            crs = CRS.from_user_input(value)
        return None if crs is None else crs.to_wkt()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return None


def _ers_coordinate_space(args, crs_name: str | None) -> tuple[str, str]:
    is_eov = (crs_name or "").upper() in {"EPSG:23700", "EOV"}
    datum = args.ers_datum
    projection = args.ers_projection
    if datum.lower() == "auto":
        datum = "HD72" if is_eov or projection.upper() == "EOV" else "RAW"
    if projection.lower() == "auto":
        projection = "EOV" if is_eov or datum.upper() == "HD72" else "RAW"
    return datum, projection


def _checkpoint_meta(ckpt: Path | dict) -> tuple[float | None, tuple[str, ...], dict]:
    ck = (
        ckpt
        if isinstance(ckpt, dict)
        else torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    )
    cell = float(ck["cell"]) if "cell" in ck else None
    names = tuple(str(x) for x in ck.get("channel_names", ()))
    norm_config = validate_norm_config(ck.get("norm_config", {"mode": "legacy_tile_percentile"}))
    return cell, names, norm_config


def _features_from_names(names: tuple[str, ...], requested: str) -> str:
    if requested != "auto":
        return requested
    gen_v2 = tuple(getattr(DSM, "GENERIC_V2_CHANNEL_NAMES", ()))
    if names == gen_v2:
        return "generic_v2"
    # Named model subset: build exactly the channels required by the weights.
    if names and set(names) <= set(gen_v2):
        return "generic_v2"
    if names == tuple(getattr(DSM, "GENERIC_CHANNEL_NAMES", ())):
        return "generic"
    if names == tuple(getattr(DSM, "LEGACY_CHANNEL_NAMES", ())):
        return "legacy"
    raise ValueError(f"Cannot infer channel feature schema from checkpoint names: {list(names)}")


def _read_xyz_files(paths: tuple[Path, ...], chunk: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read LAS files into one preallocated array without chunk concatenations."""
    counts = []
    for path in paths:
        with laspy.open(str(path)) as reader:
            counts.append(int(reader.header.point_count))
    xyz = np.empty((sum(counts), 3), np.float64)
    offset = 0
    for path, expected in zip(paths, counts, strict=True):
        written = 0
        with laspy.open(str(path)) as reader:
            for points in reader.chunk_iterator(chunk):
                size = len(points)
                target = xyz[offset + written : offset + written + size]
                target[:, 0] = points.x
                target[:, 1] = points.y
                target[:, 2] = points.z
                written += size
        if written != expected:
            raise RuntimeError(
                f"LAS point count changed while reading {path}: {expected} -> {written}"
            )
        offset += written
    return xyz, tuple(counts)


def _bn_adapt(
    model: torch.nn.Module, ch_tensor: torch.Tensor, win: int, n_passes: int, dev: torch.device
) -> int:
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d))
    ]
    if not batch_norms or n_passes <= 0:
        print("  BN adaptation skipped (model has no BatchNorm layers)")
        return 0
    C, H, W = ch_tensor.shape
    parameters = list(model.parameters())
    requires_grad = [param.requires_grad for param in parameters]
    module_training = [(module, module.training) for module in model.modules()]
    momenta = [(module, module.momentum) for module in batch_norms]
    for param in parameters:
        param.requires_grad_(False)
    for module in batch_norms:
        module.train()
        module.momentum = 0.1
    rng = np.random.default_rng(42)
    tile = torch.zeros(1, C, win, win, dtype=torch.float32, device=dev)
    try:
        with torch.inference_mode():
            for i in range(n_passes):
                r0 = int(rng.integers(0, max(H - win, 1)))
                c0 = int(rng.integers(0, max(W - win, 1)))
                h_crop = min(win, H - r0)
                w_crop = min(win, W - c0)
                tile.zero_()
                tile[0, :, :h_crop, :w_crop] = ch_tensor[:, r0 : r0 + h_crop, c0 : c0 + w_crop]
                model(tile)
                print(f"  BN adapt {i + 1}/{n_passes}", end="\r", flush=True)
    finally:
        for param, state in zip(parameters, requires_grad, strict=True):
            param.requires_grad_(state)
        for module, state in module_training:
            module.training = state
        for module, momentum in momenta:
            module.momentum = momentum
    print(f"  BN adaptation complete ({n_passes} passes)        ")
    return n_passes


def main(argv=None):
    args = parse_args(argv)
    weights = _resolve(args.weights)
    out_dir = _resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("--output-prefix must be a non-empty filename prefix")
    console_log = os.environ.get("JELI_CONSOLE_LOG", "").strip()
    if console_log:
        stream = open(console_log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = stream
        sys.stderr = stream
    dev = resolve_device(args.device)

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False, mmap=True)
    ck_cell, ck_names, norm_config = _checkpoint_meta(checkpoint)
    cell = ck_cell if args.cell == "auto" else float(args.cell)
    if cell is None:
        cell = 0.20
    features = _features_from_names(ck_names, args.features)
    print(f"weights={weights}")
    print(f"cell={cell:g} features={features} device={dev} normalization={norm_config['mode']}")

    ground_path = _resolve(args.ground)
    off_path = _resolve(args.off_ground)
    print(f"[1/5] LAS input + ground DTM: {ground_path}")
    xyz, (n_ground, n_off_ground) = _read_xyz_files((ground_path, off_path), args.chunk)
    ground = xyz[:n_ground]
    if n_ground == 0:
        raise RuntimeError(f"No ground points read from {ground_path}")
    print(f"  {n_ground:,} ground pts; DTM @ {cell:g} m")
    dtm = fill_nodata_nn(RasterGrid.from_point_cloud(ground, cell_size=cell), stat="min")

    print(f"[2/5] off-ground LAS + HAG: {off_path}")
    off_ground = xyz[n_ground:]
    if n_off_ground == 0:
        raise RuntimeError(f"No off-ground points read from {off_path}")
    # Channel construction consumes float32 HAG. Keeping a duplicate float64
    # array would waste ~246 MB on the 61.6M-point Jeli reference cloud.
    hag = np.empty(len(xyz), np.float32)
    hag[:n_ground] = 0.0
    for i in range(0, n_off_ground, args.chunk):
        part = off_ground[i : i + args.chunk]
        hag[n_ground + i : n_ground + i + len(part)] = height_above_ground(part, dtm)
    hag_off_ground = hag[n_ground:]
    print(
        f"  {n_off_ground:,} off-ground pts; "
        f"HAG [{np.nanmin(hag_off_ground):.2f},{np.nanmax(hag_off_ground):.2f}] m"
    )

    finite = np.isfinite(hag)
    if not finite.all():
        xyz, hag = xyz[finite], hag[finite]
    bounds = (
        float(xyz[:, 0].min()),
        float(xyz[:, 1].min()),
        float(xyz[:, 0].max()),
        float(xyz[:, 1].max()),
    )

    print("[3/5] building BEV channels ...")
    channels, r, names = DSM.build_channels(
        xyz, hag, cell, bounds, features, channel_names=ck_names or None
    )
    if ck_names and tuple(names) != ck_names:
        raise ValueError(
            "Built channel schema does not match weights:\n"
            f"  built:  {list(names)}\n"
            f"  weights: {list(ck_names)}"
        )
    H, W = channels.shape[1:]
    print(f"  channels={channels.shape} origin={tuple(np.round(r.origin, 2))}")

    predictor = BevPredictor(checkpoint, device=dev, channel_names=tuple(names))
    del checkpoint
    model = predictor.model
    ch = predictor.normalize(channels)

    if args.ablate_channels.strip():
        ablate = [s.strip() for s in args.ablate_channels.split(",") if s.strip()]
        for nm in ablate:
            if nm in names:
                i = names.index(nm)
                ch[i] = float(ch[i].mean())
                print(f"  ablated channel '{nm}' -> mean {float(ch[i].flat[0]):.3f}")
            else:
                print(f"  [warn] ablate channel '{nm}' not in schema {list(names)}")

    if args.bn_adapt:
        print(f"  [BN-adapt] {args.bn_passes} random passes")
        ch_t = torch.from_numpy(ch).to(dev)
        _bn_adapt(model, ch_t, args.win, args.bn_passes, dev)
        del ch_t

    print("[4/5] sliding-window inference ...")
    origin = np.asarray(r.origin, np.float64)
    prediction = predictor.predict(
        ch,
        cell=cell,
        normalized=True,
        window=args.win,
        overlap=args.overlap,
        confidence=args.conf,
        score_mode=args.decode_score_mode,
        amp=args.amp,
        batch_size=args.infer_batch_size,
        origin=origin,
        nms_iou=args.nms_iou,
    )
    dets = prediction.detections
    full_hm = prediction.heatmap
    full_seg = prediction.segmentation
    max_hm = float(full_hm.max(initial=0.0))
    max_seg = float(full_seg.max(initial=0.0))
    print(f"  max heatmap={max_hm:.4f} max seg={max_seg:.4f}")
    print(f"  detected vehicles: {len(dets)}")

    print("[5/5] rendering + CSV ...")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_channels.strip():
        want = [s.strip() for s in args.save_channels.split(",") if s.strip()]
        payload = {nm: channels[names.index(nm)].astype(np.float32) for nm in want if nm in names}
        payload["origin"] = np.asarray(origin, np.float64)
        payload["cell"] = np.float64(cell)
        np.savez_compressed(out_dir / "jeli_channels.npz", **payload)
        print(f"  saved channels -> jeli_channels.npz ({list(payload)})")
    bg_name = (
        "veh_h_p95" if "veh_h_p95" in names else ("veh_hmax" if "veh_hmax" in names else names[0])
    )
    bg = channels[names.index(bg_name)]
    extent = (origin[0], origin[0] + W * cell, origin[1], origin[1] + H * cell)
    vmax = np.percentile(bg[bg > 0], 99) if (bg > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(min(22, W / 60 + 4), min(26, H / 60 + 4)))
    ax.imshow(bg, cmap="viridis", origin="upper", extent=extent, vmax=vmax)
    for de in dets:
        ax.add_patch(MplPoly(de["verts"], closed=True, fill=False, edgecolor="#00e5ff", lw=1.0))
    ax.set_title(f"Jeli BEV detections: {len(dets)}  cell={cell:g} conf={args.conf}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_detections.png", dpi=180)
    plt.close(fig)

    seg_up = np.repeat(
        np.repeat(full_seg, STRIDE, axis=0),
        STRIDE,
        axis=1,
    )[:H, :W]
    crs_name = _crs_name(args.crs_name, ground_path)
    geojson_path = _output_target(
        args.output_geojson, out_dir, f"{prefix}_vehicles.geojson"
    )
    if geojson_path is not None:
        write_detections_geojson(
            dets,
            geojson_path,
            crs_name=crs_name,
            name=prefix,
        )
        print(f"  -> {geojson_path}")
    shp_path = _output_target(args.output_shp, out_dir, f"{prefix}_vehicle_obbs.shp")
    if shp_path is not None:
        obb_path, center_path = write_detection_shapefiles(
            dets,
            shp_path,
            crs_wkt=_crs_wkt(args.crs_name, ground_path),
        )
        print(f"  -> {obb_path}")
        print(f"  -> {center_path}")
    ers_path = _output_target(args.output_mask_ers, out_dir, f"{prefix}_mask.ers")
    if ers_path is not None:
        ers_datum, ers_projection = _ers_coordinate_space(args, crs_name)
        written_ers = write_segmentation_mask_ers(
            seg_up,
            r,
            ers_path,
            threshold=args.mask_threshold,
            datum=ers_datum,
            projection=ers_projection,
        )
        print(f"  -> {written_ers} (+ binary raster)")
    fig, ax = plt.subplots(figsize=(min(22, W / 60 + 4), min(26, H / 60 + 4)))
    ax.imshow(bg, cmap="gray", origin="upper", extent=extent, vmax=vmax)
    ax.imshow(
        np.ma.masked_less(seg_up, 0.5),
        cmap="magma",
        origin="upper",
        extent=extent,
        vmin=0,
        vmax=1,
        alpha=0.55,
    )
    for de in dets:
        ax.add_patch(MplPoly(de["verts"], closed=True, fill=False, edgecolor="#00e5ff", lw=1.0))
    ax.set_title(f"Jeli BEV segmentation + detections: {len(dets)}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_segmentation.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(min(22, W / 60 + 4), min(26, H / 60 + 4)))
    im = ax.imshow(seg_up, cmap="magma", origin="upper", extent=extent, vmin=0, vmax=1)
    for de in dets:
        ax.add_patch(MplPoly(de["verts"], closed=True, fill=False, edgecolor="#00e5ff", lw=1.0))
    ax.set_title(f"Jeli BEV segmentation probability: {len(dets)} detections")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_seg_probability.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(min(22, W / 60 + 4), min(26, H / 60 + 4)))
    ax.imshow(seg_up >= 0.5, cmap="gray", origin="upper", extent=extent, vmin=0, vmax=1)
    for de in dets:
        ax.add_patch(MplPoly(de["verts"], closed=True, fill=False, edgecolor="#00e5ff", lw=1.0))
    ax.set_title(f"Jeli BEV segmentation mask >= 0.5: {len(dets)} detections")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_seg_mask_only.png", dpi=180)
    plt.close(fig)

    if dets:
        cxs = np.array([d["cx"] for d in dets])
        cys = np.array([d["cy"] for d in dets])
        pad = 25.0
        x0, x1 = cxs.min() - pad, cxs.max() + pad
        y0, y1 = cys.min() - pad, cys.max() + pad
        fig, ax = plt.subplots(figsize=(12, 12 * (y1 - y0) / max(x1 - x0, 1e-3)))
        ax.imshow(bg, cmap="viridis", origin="upper", extent=extent, vmax=vmax)
        for de in dets:
            ax.add_patch(MplPoly(de["verts"], closed=True, fill=False, edgecolor="#00e5ff", lw=2.0))
            ax.plot(de["cx"], de["cy"], "+", color="#ff2d2d", ms=8)
            ax.text(
                de["cx"],
                de["cy"] + 2,
                f"{de['score']:.2f}",
                color="white",
                fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"),
            )
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(f"Jeli BEV zoom: {len(dets)} detections")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_zoom.png", dpi=170)
        plt.close(fig)

    if args.probe_xy.strip():
        px, py = (float(v) for v in args.probe_xy.split(","))
        rad = float(args.probe_radius_m)
        c0p = max(int((px - rad - origin[0]) / cell), 0)
        c1p = min(int((px + rad - origin[0]) / cell), W)
        r1p = min(int(H - (py - rad - origin[1]) / cell), H)
        r0p = max(int(H - (py + rad - origin[1]) / cell), 0)
        hm_up = np.repeat(
            np.repeat(full_hm, STRIDE, axis=0),
            STRIDE,
            axis=1,
        )[:H, :W]
        panels = [(nm, channels[i]) for i, nm in enumerate(names)]
        panels += [("HEATMAP", hm_up), ("SEG_PROB", seg_up)]
        ncol = 5
        nrow = (len(panels) + ncol - 1) // ncol
        figp, axp = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow))
        for ax, (nm, img) in zip(axp.ravel(), panels, strict=False):
            crop = img[r0p:r1p, c0p:c1p]
            ax.imshow(crop, cmap="magma")
            peak = float(crop.max()) if crop.size else 0.0
            ax.set_title(f"{nm}  max={peak:.2f}", fontsize=8)
            ax.axis("off")
        for ax in axp.ravel()[len(panels) :]:
            ax.axis("off")
        figp.suptitle(
            f"probe @ ({px:.0f},{py:.0f}) r={rad:g}m  "
            f"HEATMAP max here={hm_up[r0p:r1p, c0p:c1p].max():.3f} "
            f"(scene max {max_hm:.3f})",
            fontsize=11,
        )
        figp.tight_layout()
        figp.savefig(out_dir / f"{prefix}_probe.png", dpi=110)
        plt.close(figp)
        print(
            f"  probe panel -> {prefix}_probe.png  "
            f"(heatmap max at probe = {hm_up[r0p:r1p, c0p:c1p].max():.3f})"
        )

    with (out_dir / f"{prefix}_boxes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cx",
                "cy",
                "L",
                "W",
                "theta_deg",
                "z",
                "height",
                "score",
                "hm_score",
                "seg_center_score",
                "seg_obb_support",
            ]
        )
        for de in dets:
            w.writerow(
                [
                    f"{de['cx']:.3f}",
                    f"{de['cy']:.3f}",
                    f"{de['L']:.3f}",
                    f"{de['W']:.3f}",
                    f"{np.degrees(de['th']):.2f}",
                    f"{de.get('z', 0):.2f}",
                    f"{de.get('height', 0):.2f}",
                    f"{de['score']:.3f}",
                    f"{de.get('hm_score', 0):.3f}",
                    f"{de.get('seg_center_score', 0):.3f}",
                    f"{de.get('seg_obb_support', 0):.3f}",
                ]
            )
    print(f"  -> {out_dir / f'{prefix}_detections.png'}")
    print(f"  -> {out_dir / f'{prefix}_segmentation.png'}")
    print(f"  -> {out_dir / f'{prefix}_seg_probability.png'}")
    print(f"  -> {out_dir / f'{prefix}_seg_mask_only.png'}")
    print(f"  -> {out_dir / f'{prefix}_boxes.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
