"""
Parallel multi-source BEV dataset generator.

Generates train/ and test/ NPZ tiles for all supported sources using a
configurable worker pool. Output is organized into clean subdirectories
with unified {source}_{id}.npz filenames.

Directory layout:
  {out}/
    train/
      dales_5080_54400.npz
      synth_train_0000.npz
      yuto_619375_4846250.npz
      eclair_364000_6676000.npz
      isprs.npz
    test/
      dales_5100_54440.npz
      synth_test_0000.npz
      isprs.npz
    viz/
      train/  (PNG previews)
      test/

Sources:
  synth    procedurally generated scenes with perfect OBB labels
  dales    DALESObjects PLY (ground-based, instance OBBs, shadow-close fp_mask)
  yuto     YutoSemantic LAS (airborne, 500 m tiles, seg-close fp_mask)
  eclair   ECLAIR LAZ (airborne, 500 m tiles, sequential LAZ streaming)
  isprs    Vaihingen PTS (airborne, two splits)

Usage:
  python -m bordereye.ml.bev.data.build_dataset --sources all
  python -m bordereye.ml.bev.data.build_dataset --sources dales,synth --workers 6
  python -m bordereye.ml.bev.data.build_dataset --sources synth --n-train 200 --n-test 40
  python -m bordereye.ml.bev.data.build_dataset --sources dales --cell 0.20 --no-viz
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bordereye import RasterGrid
from bordereye.io import read_dales_ply, read_las_full
from bordereye.ml.bev.core.footprint_masks import fp_mask_from_boxes
from bordereye.ml.bev.core.mask_obb_teacher import DEFAULT_PRIOR, length_prior_boxes_from_mask
from bordereye.ml.bev.data import bev_dataset as DS
from bordereye.ml.bev.data.build_contract import (
    DATASET_VERSION,
    MASK_POLICY,
    PSEUDO_OBB_POLICY,
)
from bordereye.ml.bev.data.build_contract import (
    save_args as _save_args,
)
from bordereye.ml.bev.data.build_contract import (
    should_skip_existing as _should_skip_existing,
)
from bordereye.ml.bev.data.paths import dataset_root
from bordereye.ml.bev.data.synthetic_builder import _save_viz, _worker_synth
from bordereye.paths import datasets_root
from bordereye.raster_ops import fill_nodata_nn
from bordereye.terrain import height_above_ground
from bordereye.tiles import tile_point_cloud

from ...._masks import in_classes

# ── constants ──────────────────────────────────────────────────────────────────
DALES_GROUND = 1
DALES_VEHICLE = (3, 4)
YUTO_GROUND = 1
YUTO_VEHICLE = (5, 6)
YUTO_TILE_SIZE = 500.0
ECLAIR_GROUND = 2
ECLAIR_VEHICLE = 11
ECLAIR_TILE_SIZE = 500.0
ECLAIR_ROOT = datasets_root() / "ECLAIR"
ECLAIR_DATASET_DIR = ECLAIR_ROOT / "dataset"
ECLAIR_LABELS = ECLAIR_DATASET_DIR / "labels.json"
ECLAIR_POINTCLOUDS = ECLAIR_DATASET_DIR / "pointclouds"
ECLAIR_LAZ = ECLAIR_ROOT / "pointclouds" / "eclair-full.laz"


def _limit_or_none(value):
    if value is None or int(value) < 0:
        return None
    return int(value)


def _build_dtm(xyz, ground_mask, cell):
    """Build a ground DTM, or return ``None`` for unusable source tiles."""
    ground = xyz[ground_mask]
    if len(ground) < 10:
        return None
    raster = RasterGrid.from_point_cloud(ground, cell_size=cell)
    return fill_nodata_nn(raster, stat="min")


def _mask_teacher_pseudo(seg_mask, origin, cell):
    boxes, scores = length_prior_boxes_from_mask(
        np.asarray(seg_mask).astype(bool),
        np.asarray(origin, np.float64),
        float(cell),
        car_length=DEFAULT_PRIOR.length_m,
        car_width=DEFAULT_PRIOR.width_m,
    )
    return boxes.astype(np.float32).reshape(-1, 7), scores.astype(np.float32)


def _weak_source_pseudo(tile_xyz, hag, vehicle_mask, cell, bounds, r, *, min_points=6):
    boxes_semantic, boxes_ignore = DS.boxes_from_semantic_components(
        tile_xyz, hag, vehicle_mask, cell, bounds, min_points=min_points
    )
    seg_mask = DS.vehicle_point_seg_mask(tile_xyz, hag, vehicle_mask, r, cell)
    boxes_teacher, teacher_score = _mask_teacher_pseudo(seg_mask, r.origin, cell)
    if len(boxes_teacher) > 0:
        boxes_pseudo = boxes_teacher
        pseudo_score = teacher_score
    else:
        boxes_pseudo = boxes_semantic
        pseudo_score = np.ones(len(boxes_semantic), np.float32)
    # Same support rule as the synthsparse worker: a pseudo box the scan left
    # (nearly) no seg evidence in is unlearnable from this tile (survey-edge
    # slivers carry boxes over ~0 points). Move it to ignore so it costs
    # neither heatmap loss nor eval recall.
    if len(boxes_pseudo):
        from bordereye.ml.bev.core.mask_obb_teacher import candidate_support

        seg_b = seg_mask > 0
        sup_px = np.array(
            [
                candidate_support(
                    seg_b,
                    r.origin,
                    cell,
                    float(b[0]),
                    float(b[1]),
                    float(b[2]),
                    float(b[3]),
                    float(np.deg2rad(b[4])),
                )
                * (float(b[2]) * float(b[3]) / cell**2)
                for b in boxes_pseudo
            ]
        )
        unsup = sup_px < 3.0
        if unsup.any():
            boxes_ignore = np.concatenate(
                [
                    np.asarray(boxes_ignore, np.float32).reshape(-1, 7),
                    boxes_pseudo[unsup].reshape(-1, 7),
                ],
                axis=0,
            )
            boxes_pseudo = boxes_pseudo[~unsup]
            pseudo_score = pseudo_score[~unsup]
    # keep_unexplained: on semantic sources, seg components no pseudo box
    # explains are almost always vehicles the teacher missed; keep them in the
    # footprint target instead of silently deleting the evidence.
    fp_mask = fp_mask_from_boxes(
        seg_mask, boxes_pseudo, r.origin, cell, keep_unexplained=True, sparse_box_fill=True
    )
    extra = {
        # BevCrops reads boxes_pseudo_score to scale per-box pseudo regression
        # weight by teacher fit quality; the *_mask_teacher keys keep the raw
        # teacher output for audits even when boxes_pseudo comes from it.
        "boxes_pseudo_score": pseudo_score.astype(np.float32),
        "boxes_pseudo_semantic": boxes_semantic.astype(np.float32).reshape(-1, 7),
        "boxes_pseudo_mask_teacher": boxes_teacher.astype(np.float32).reshape(-1, 7),
        "boxes_pseudo_mask_teacher_score": teacher_score.astype(np.float32),
    }
    return boxes_pseudo, boxes_ignore, seg_mask, fp_mask, extra


# ── synthetic scene generation ─────────────────────────────────────────────────
def _xyz_bounds(xyz) -> tuple[float, float, float, float]:
    """Axis-aligned XY extent of a point array as ``(x0, y0, x1, y1)``."""
    return (
        float(xyz[:, 0].min()),
        float(xyz[:, 1].min()),
        float(xyz[:, 0].max()),
        float(xyz[:, 1].max()),
    )


def _write_weak_source_tile(tile_id, out_path, viz_dir, xyz, hag, vehicle_mask,
                            bounds, cfg, source, split, *, min_points=6,
                            require_evidence=True):
    """Build, save and optionally render one weakly-supervised tile.

    YUTO, ISPRS and ECLAIR all derive their supervision the same way — pseudo
    OBBs fitted to a per-point vehicle mask, no trusted boxes — and differ only
    in how the points and that mask are read. Keeping the shared half here means
    the pseudo-box policy cannot drift between sources.

    Returns the pseudo-box count, or ``None`` when *require_evidence* is set and
    the tile yielded neither a pseudo box nor a false-positive region.
    """
    save_args = _save_args(cfg, PSEUDO_OBB_POLICY)
    channels, r, channel_names = DS.build_channels(
        xyz, hag, cfg["cell"], bounds, cfg["features"]
    )
    boxes_valid = DS._empty_boxes()
    boxes_pseudo, boxes_ignore, seg_mask, fp_mask, extra = _weak_source_pseudo(
        xyz, hag, vehicle_mask, cfg["cell"], bounds, r, min_points=min_points
    )
    if require_evidence and len(boxes_pseudo) == 0 and not fp_mask.any():
        return None
    DS.save_bev_npz(
        out_path,
        channels,
        r,
        boxes_valid,
        boxes_pseudo,
        boxes_ignore,
        seg_mask,
        channel_names,
        save_args,
        source,
        split,
        fp_mask=fp_mask,
        extra=extra,
    )
    if viz_dir is not None:
        _save_viz(
            viz_dir / f"{tile_id}.png",
            channels,
            channel_names,
            r,
            boxes_valid,
            boxes_pseudo,
            fp_mask=fp_mask,
        )
    return len(boxes_pseudo)


def _worker_dales(job):
    """Process one DALESObjects PLY tile."""
    ply_path, split, cfg, out_dir, viz_dir = job
    stem = Path(ply_path).stem.replace("_new", "")
    tile_id = f"dales_{stem}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    try:
        ply = read_dales_ply(ply_path)
        sem = np.asarray(ply["sem_class"], np.int32)
        ins_raw = np.asarray(ply["ins_class"], np.int64)
        xyz = np.column_stack(
            [ply["x"].astype(float), ply["y"].astype(float), ply["z"].astype(float)]
        )
        inst = np.zeros(len(sem), np.uint32)
        veh = in_classes(sem, DALES_VEHICLE) & (ins_raw > 0)
        if veh.any():
            _, inv = np.unique(ins_raw[veh], return_inverse=True)
            inst[veh] = inv.astype(np.uint32) + 1

        ground = xyz[sem == DALES_GROUND]
        dtm = fill_nodata_nn(RasterGrid.from_point_cloud(ground, cell_size=cfg["cell"]), stat="min")
        hag = height_above_ground(xyz, dtm)
        bounds = _xyz_bounds(xyz)

        save_args = _save_args(cfg)
        channels, r, channel_names = DS.build_channels(
            xyz, hag, cfg["cell"], bounds, cfg["features"]
        )
        boxes_valid, boxes_ignore = DS.boxes_from_instances(xyz, sem, inst, hag)
        boxes_pseudo = DS._empty_boxes()
        seg_mask = DS.vehicle_point_seg_mask(xyz, hag, veh, r, cfg["cell"])
        seg_box_mask = DS.vehicle_box_seg_mask(boxes_valid, r, cfg["cell"])
        seg_box_wt = DS.vehicle_box_seg_weight(seg_box_mask, cfg["cell"])
        fp_mask = fp_mask_from_boxes(seg_mask, boxes_valid, r.origin, cfg["cell"])

        DS.save_bev_npz(
            out_path,
            channels,
            r,
            boxes_valid,
            boxes_pseudo,
            boxes_ignore,
            seg_mask,
            channel_names,
            save_args,
            "dales",
            split,
            seg_box_mask=seg_box_mask,
            seg_box_weight=seg_box_wt,
            fp_mask=fp_mask,
        )
        if viz_dir is not None:
            _save_viz(
                viz_dir / f"{tile_id}.png",
                channels,
                channel_names,
                r,
                boxes_valid,
                boxes_pseudo,
                fp_mask=fp_mask,
                seg_box_mask=seg_box_mask,
            )
        return tile_id, {"valid": len(boxes_valid), "ignore": len(boxes_ignore), "split": split}
    except Exception:
        return str(Path(ply_path).stem), {"error": traceback.format_exc()}


def _worker_yuto(job):
    """Process one 500 m YUTO sub-tile."""
    tile_xyz, hag_tile, vm_tile, bounds, split, cfg, out_dir, viz_dir = job
    tile_id = f"yuto_{int(bounds[0])}_{int(bounds[1])}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    try:
        if not bool(np.asarray(vm_tile).any()):
            # Same policy as eclair: tiles without any vehicle evidence carry
            # no supervision signal and would skew split balance.
            return tile_id, {"skipped": "no_vehicle"}
        n_pseudo = _write_weak_source_tile(
            tile_id, out_path, viz_dir, tile_xyz, hag_tile, vm_tile,
            bounds, cfg, "yuto", split,
        )
        if n_pseudo is None:
            return tile_id, {"skipped": "no_boxes"}
        return tile_id, {"pseudo": n_pseudo, "split": split}
    except Exception:
        return tile_id, {"error": traceback.format_exc()}


def _worker_isprs(job):
    """Process one ISPRS PTS file."""
    pts_path, split, cfg, out_dir, viz_dir = job
    tile_id = f"isprs_{split}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    try:
        arr = np.loadtxt(pts_path, dtype=np.float64)
        xyz = arr[:, :3].astype(np.float64)
        labels = arr[:, 6].astype(np.int32)
        ground = xyz[labels == 2]
        if len(ground) < 100:
            ground = xyz[in_classes(labels, (1, 2))]
        dtm = fill_nodata_nn(RasterGrid.from_point_cloud(ground, cell_size=cfg["cell"]), stat="min")
        hag = height_above_ground(xyz, dtm)
        n_pseudo = _write_weak_source_tile(
            tile_id, out_path, viz_dir, xyz, hag, labels == 3,
            _xyz_bounds(xyz), cfg, "isprs", split,
            min_points=10, require_evidence=False,
        )
        return tile_id, {
            "pseudo": n_pseudo,
            "car_pts": int((labels == 3).sum()),
            "split": split,
        }
    except Exception:
        return f"isprs_{split}", {"error": traceback.format_exc()}


# ── ECLAIR: sequential column-strip LAZ reader + tile worker ──────────────────


def _process_eclair_tile(xyz_t, cls_t, x0, y0, split, cfg, out_dir, viz_dir):
    """Process one ECLAIR 500x500 m tile (called serially within the LAZ stream)."""
    tile_id = f"eclair_{int(x0)}_{int(y0)}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    bounds = (float(x0), float(y0), float(x0 + ECLAIR_TILE_SIZE), float(y0 + ECLAIR_TILE_SIZE))
    dtm = _build_dtm(xyz_t, cls_t == ECLAIR_GROUND, cfg["cell"])
    if dtm is None:
        return None
    hag = height_above_ground(xyz_t, dtm)
    n_pseudo = _write_weak_source_tile(
        tile_id, out_path, viz_dir, xyz_t, hag, cls_t == ECLAIR_VEHICLE,
        bounds, cfg, "eclair", split,
    )
    if n_pseudo is None:
        return None
    return tile_id, {"pseudo": n_pseudo}


def _process_eclair_file(job):
    """Process one official ECLAIR pointcloud_*.laz tile."""
    laz_path, split, cfg, out_dir, viz_dir = job
    laz_path = Path(laz_path)
    tile_id = f"eclair_{laz_path.stem}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if viz_dir is not None:
            Path(viz_dir).mkdir(parents=True, exist_ok=True)
        data = read_las_full(laz_path)
        xyz = data["xyz"]
        cls = data["classification"]
        if not (cls == ECLAIR_VEHICLE).any():
            return f"eclair_{laz_path.stem}", {"skipped": "no_vehicle"}
        dtm = _build_dtm(xyz, cls == ECLAIR_GROUND, cfg["cell"])
        if dtm is None:
            return f"eclair_{laz_path.stem}", {"skipped": "no_ground"}
        hag = height_above_ground(xyz, dtm)
        bounds = (
            float(xyz[:, 0].min()),
            float(xyz[:, 1].min()),
            float(xyz[:, 0].max()),
            float(xyz[:, 1].max()),
        )
        save_args = _save_args(cfg, PSEUDO_OBB_POLICY)
        channels, r, channel_names = DS.build_channels(
            xyz, hag, cfg["cell"], bounds, cfg["features"]
        )
        boxes_pseudo, boxes_ignore, seg_mask, fp_mask, extra = _weak_source_pseudo(
            xyz, hag, cls == ECLAIR_VEHICLE, cfg["cell"], bounds, r
        )
        if len(boxes_pseudo) == 0 and not fp_mask.any():
            # Ignore-only tiles (no pseudo box, no usable footprint target)
            # carry no supervision and act as unfair negatives in eval.
            return f"eclair_{laz_path.stem}", {"skipped": "no_boxes"}
        boxes_valid = DS._empty_boxes()
        DS.save_bev_npz(
            out_path,
            channels,
            r,
            boxes_valid,
            boxes_pseudo,
            boxes_ignore,
            seg_mask,
            channel_names,
            save_args,
            "eclair",
            split,
            fp_mask=fp_mask,
            extra=extra,
        )
        if viz_dir is not None:
            _save_viz(
                viz_dir / f"{tile_id}.png",
                channels,
                channel_names,
                r,
                boxes_valid,
                boxes_pseudo,
                fp_mask=fp_mask,
            )
        return tile_id, {
            "pseudo": len(boxes_pseudo),
            "ignore": len(boxes_ignore),
            "vehicle_pts": int((cls == ECLAIR_VEHICLE).sum()),
            "split": split,
        }
    except Exception:
        return f"eclair_{laz_path.stem}", {"error": traceback.format_exc()}


# ── source orchestrators ───────────────────────────────────────────────────────


def run_synth(cfg, out_dir, viz_dir, workers):
    n_train = max(0, int(cfg["n_train"]))
    n_test = max(0, int(cfg["n_test"]))
    print(
        f"\n[synth] generating {n_train} train + {n_test} test scenes  workers={workers}",
        flush=True,
    )
    jobs = [
        (i, "train", cfg, out_dir / "train", viz_dir / "train" if viz_dir else None)
        for i in range(n_train)
    ] + [
        (i, "test", cfg, out_dir / "test", viz_dir / "test" if viz_dir else None)
        for i in range(n_test)
    ]
    return _run_pool(jobs, _worker_synth, workers)


def run_dales(cfg, out_dir, viz_dir, workers):
    rng = np.random.default_rng(cfg.get("seed", 0))
    jobs = []
    for split, limit_raw in (
        ("train", cfg["source_limit_train"]),
        ("test", cfg["source_limit_test"]),
    ):
        limit = _limit_or_none(limit_raw)
        if limit == 0:
            continue
        base = datasets_root() / "DALESObjects" / split
        all_tiles = sorted(base.glob("*_new.ply"))
        if limit is not None and limit < len(all_tiles):
            idxs = rng.choice(len(all_tiles), size=limit, replace=False)
            all_tiles = [all_tiles[i] for i in sorted(idxs)]
        for p in all_tiles:
            jobs.append((str(p), split, cfg, out_dir / split, viz_dir / split if viz_dir else None))
    print(f"\n[dales] {len(jobs)} tiles  workers={workers}", flush=True)
    return _run_pool(jobs, _worker_dales, workers)


def run_yuto(cfg, out_dir, viz_dir, workers):
    yuto_dir = datasets_root() / "YutoSemantic" / "v1"
    las_files = sorted(yuto_dir.glob("*.las"))
    if not las_files:
        print(f"\n[yuto] no LAS files found in {yuto_dir}")
        return []

    rng = np.random.default_rng(cfg.get("seed", 0))
    las_files = list(las_files)
    rng.shuffle(las_files)
    # YUTO has no official test split. Hold out whole LAS flights so the
    # train/test boundary has no spatial leakage between adjacent tiles.
    test_frac = float(cfg.get("yuto_test_frac", 0.0))
    n_test_files = round(test_frac * len(las_files)) if len(las_files) > 1 else 0
    split_of = {p: ("test" if i < n_test_files else "train") for i, p in enumerate(las_files)}
    limits = {
        "train": _limit_or_none(cfg["source_limit_train"]),
        "test": _limit_or_none(cfg["source_limit_test"]),
    }
    counts = {"train": 0, "test": 0}
    results = []
    for las_path in las_files:
        split = split_of[las_path]
        limit = limits[split]
        if limit is not None and counts[split] >= limit:
            continue
        print(f"  [yuto] tiling {las_path.name} ({split}) ...", flush=True)
        try:
            data = read_las_full(las_path)
        except OSError as e:
            print(f"  skipped (unreadable): {e}")
            continue
        xyz = data["xyz"]
        cl = data["classification"]
        vm = in_classes(cl, YUTO_VEHICLE)
        dtm = _build_dtm(xyz, cl == YUTO_GROUND, cfg["cell"])
        if dtm is None:
            print(f"  skipped (no ground): {las_path.name}")
            continue
        hag = height_above_ground(xyz, dtm)
        jobs = []
        for tile_xyz, bounds in tile_point_cloud(xyz, YUTO_TILE_SIZE, min_points=500):
            if limit is not None and counts[split] + len(jobs) >= limit:
                break
            tile_id = f"yuto_{int(bounds[0])}_{int(bounds[1])}"
            out_path = out_dir / split / f"{tile_id}.npz"
            if _should_skip_existing(out_path, cfg):
                counts[split] += 1
                continue
            mask = (
                (xyz[:, 0] >= bounds[0])
                & (xyz[:, 0] < bounds[2])
                & (xyz[:, 1] >= bounds[1])
                & (xyz[:, 1] < bounds[3])
            )
            jobs.append(
                (
                    tile_xyz,
                    hag[mask],
                    vm[mask],
                    bounds,
                    split,
                    cfg,
                    out_dir / split,
                    viz_dir / split if viz_dir else None,
                )
            )
        del data, xyz, cl, vm, hag, dtm
        # One pool pass per LAS file: memory holds a single flight's tiles
        # instead of the entire source before dispatch.
        file_results = _run_pool(jobs, _worker_yuto, workers)
        counts[split] += len(file_results)
        results.extend(file_results)
    print(f"[yuto] total {len(results)} tiles (train={counts['train']} test={counts['test']})")
    return results


def run_eclair_official(cfg, out_dir, viz_dir, workers):
    """Process the official ECLAIR release: labels.json + pointcloud_*.laz."""
    if not ECLAIR_LABELS.exists() or not ECLAIR_POINTCLOUDS.exists():
        return None
    try:
        import laspy

        if not laspy.LazBackend.detect_available():
            print("  [eclair] no LAZ backend — pip install lazrs")
            return []
    except Exception as e:
        print(f"  [eclair] {e}")
        return []
    with ECLAIR_LABELS.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    by_split = {"train": [], "test": []}
    missing = 0
    for row in labels:
        split_raw = str(row.get("split", "train")).lower()
        out_split = "test" if split_raw == "test" else "train"
        path = ECLAIR_POINTCLOUDS / str(row["tile_name"])
        if not path.exists():
            missing += 1
            continue
        by_split[out_split].append((path, split_raw))
    rng = np.random.default_rng(cfg.get("seed", 0))
    jobs = []
    for out_split, limit_key in (("train", "source_limit_train"), ("test", "source_limit_test")):
        rows = list(by_split[out_split])
        rng.shuffle(rows)
        limit = _limit_or_none(cfg.get(limit_key))
        if limit is not None:
            rows = rows[:limit]
        for path, split_raw in rows:
            jobs.append(
                (
                    str(path),
                    split_raw,
                    cfg,
                    out_dir / out_split,
                    viz_dir / out_split if viz_dir else None,
                )
            )
    print(
        f"\n[eclair] official labels={len(labels)} "
        f"train+val={len(by_split['train'])} test={len(by_split['test'])} "
        f"jobs={len(jobs)} missing={missing} workers={workers}",
        flush=True,
    )
    return _run_pool(jobs, _process_eclair_file, max(1, workers))


def run_eclair_legacy(cfg, out_dir, viz_dir):
    """ECLAIR legacy mode: single large LAZ file, column-strip streaming."""
    if not ECLAIR_LAZ.exists():
        print(f"\n[eclair] LAZ not found: {ECLAIR_LAZ}")
        return []
    try:
        import laspy

        if not laspy.LazBackend.detect_available():
            print("  [eclair] no LAZ backend — pip install lazrs")
            return []
    except Exception as e:
        print(f"  [eclair] {e}")
        return []

    with laspy.open(str(ECLAIR_LAZ)) as f:
        hdr = f.header
        x_min, y_min = float(hdr.mins[0]), float(hdr.mins[1])
        x_max, y_max = float(hdr.maxs[0]), float(hdr.maxs[1])
        print(
            f"\n[eclair] {hdr.point_count:,} pts  "
            f"{x_max - x_min:.0f}x{y_max - y_min:.0f} m  (sequential)",
            flush=True,
        )

    T = ECLAIR_TILE_SIZE
    x_starts = np.arange(np.floor(x_min / T) * T, x_max, T)
    y_starts = np.arange(np.floor(y_min / T) * T, y_max, T)
    results = []

    for col_idx, x0 in enumerate(x_starts):
        x1 = x0 + T
        buf_xyz, buf_cls = [], []
        with laspy.open(str(ECLAIR_LAZ)) as f:
            for chunk in f.chunk_iterator(2_000_000):
                cx = np.asarray(chunk.x, np.float64)
                m = (cx >= x0) & (cx < x1)
                if not m.any():
                    continue
                buf_xyz.append(
                    np.column_stack(
                        [
                            cx[m].astype(np.float32),
                            np.asarray(chunk.y, np.float64)[m].astype(np.float32),
                            np.asarray(chunk.z, np.float64)[m].astype(np.float32),
                        ]
                    )
                )
                buf_cls.append(np.asarray(chunk.classification, np.uint8)[m])
        if not buf_xyz:
            continue
        xyz_strip = np.concatenate(buf_xyz)
        cls_strip = np.concatenate(buf_cls)
        del buf_xyz, buf_cls
        print(
            f"  col {col_idx + 1}/{len(x_starts)}: x=[{x0:.0f},{x1:.0f}) {len(xyz_strip):,} pts",
            flush=True,
        )
        for y0 in y_starts:
            limit = _limit_or_none(cfg["source_limit_train"])
            if limit is not None and len(results) >= limit:
                break
            y1 = y0 + T
            tile_m = (xyz_strip[:, 1] >= y0) & (xyz_strip[:, 1] < y1)
            if not (cls_strip[tile_m] == ECLAIR_VEHICLE).any():
                continue
            res = _process_eclair_tile(
                xyz_strip[tile_m],
                cls_strip[tile_m],
                x0,
                y0,
                "train",
                cfg,
                out_dir / "train",
                viz_dir / "train" if viz_dir else None,
            )
            if res is not None:
                results.append(res)
        limit = _limit_or_none(cfg["source_limit_train"])
        if limit is not None and len(results) >= limit:
            break
        del xyz_strip, cls_strip

    return results


def run_eclair(cfg, out_dir, viz_dir, workers):
    official = run_eclair_official(cfg, out_dir, viz_dir, workers)
    if official is not None:
        return official
    return run_eclair_legacy(cfg, out_dir, viz_dir)


def run_isprs(cfg, out_dir, viz_dir, workers):
    base = datasets_root() / "Urban" / "3DLabeling"
    split_files = {
        "train": base / "Vaihingen3D_Traininig.pts",
        "test": base / "Vaihingen3D_EVAL_WITH_REF.pts",
    }
    jobs = [
        (str(p), sp, cfg, out_dir / sp, viz_dir / sp if viz_dir else None)
        for sp, p in split_files.items()
        if p.exists()
    ]
    if not jobs:
        print(f"\n[isprs] no PTS files found in {base}")
        return []
    print(f"\n[isprs] {len(jobs)} splits  workers={min(workers, len(jobs))}", flush=True)
    return _run_pool(jobs, _worker_isprs, min(workers, len(jobs)))


# ── pool helper ────────────────────────────────────────────────────────────────


def _run_pool(jobs, fn, workers):
    results = []
    errors = 0
    skipped = 0
    t0 = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            tile_id, stats = fn(job)
            if "error" in stats:
                print(f"  ERROR {tile_id}: {stats['error'][:120]}")
                errors += 1
            elif "skipped" in stats:
                skipped += 1
            else:
                results.append((tile_id, stats))
                print(f"  ok  {tile_id}  {stats}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fn, job): job for job in jobs}
            for fut in as_completed(futs):
                tile_id, stats = fut.result()
                if "error" in stats:
                    print(f"  ERROR {tile_id}: {stats['error'][:120]}")
                    errors += 1
                elif "skipped" in stats:
                    skipped += 1
                else:
                    results.append((tile_id, stats))
                    print(f"  ok  {tile_id}  {stats}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"  -> {len(results)} tiles saved  {skipped} skipped  {errors} errors  {elapsed:.1f}s")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sources",
        default="all",
        help="comma-separated: synth,synthsparse,dales,yuto,eclair,isprs "
        "or 'all' (= the five real/dense sources; synthsparse is "
        "opt-in)",
    )
    p.add_argument(
        "--scan-profile",
        default="mix",
        choices=("mix", "yuto", "eclair", "isprs", "yuto_wide", "yuto_patchy"),
        help="synthsparse scan-degradation profile; 'mix' draws one of the three per scene",
    )
    p.add_argument(
        "--out", default=str(dataset_root()), help="output root (creates train/ and test/ subdirs)"
    )
    p.add_argument("--cell", type=float, default=0.20, help="raster cell size in metres")
    p.add_argument("--features", choices=["generic", "generic_v2", "legacy"], default="generic_v2")
    p.add_argument(
        "--workers", type=int, default=None, help="parallel worker processes (default: cpu_count-1)"
    )
    p.add_argument("--no-viz", action="store_true", help="skip PNG exports")
    p.add_argument(
        "--source-limit-train",
        type=int,
        default=-1,
        help="max train tiles per REAL source (dales/yuto/eclair); "
        "-1 = all. Synth counts are --n-train/--n-test.",
    )
    p.add_argument(
        "--source-limit-test", type=int, default=-1, help="max test tiles per REAL source; -1 = all"
    )
    p.add_argument(
        "--yuto-test-frac",
        type=float,
        default=0.2,
        help="fraction of YUTO LAS files held out as the test split "
        "(file-level split avoids spatial leakage); 0 disables",
    )
    p.add_argument(
        "--no-compress",
        action="store_true",
        help="write uncompressed NPZ tiles (larger on disk, several "
        "times faster to load during training)",
    )
    p.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="regenerate existing NPZ files; default is to skip them",
    )
    p.add_argument(
        "--wisar-dataset-dir",
        default="",
        help="existing WiSAR BEV dataset to integrate when --sources includes wisar",
    )

    # synth-specific
    sg = p.add_argument_group("synth")
    sg.add_argument("--n-train", type=int, default=40, help="synth train scenes")
    sg.add_argument("--n-test", type=int, default=10, help="synth test scenes")
    sg.add_argument("--seed", type=int, default=42)
    sg.add_argument("--tile-size", type=float, default=100.0)
    sg.add_argument("--veh-per-scene", type=int, nargs=2, default=[12, 45])
    sg.add_argument("--ground-density", type=float, nargs=2, default=[6.0, 35.0])
    sg.add_argument("--veh-density", type=float, nargs=2, default=[15.0, 45.0])
    sg.add_argument("--ground-noise", type=float, default=0.03)
    sg.add_argument("--max-slope", type=float, default=0.08)
    sg.add_argument("--trees-per-scene", type=int, nargs=2, default=[18, 45])
    sg.add_argument("--houses-per-scene", type=int, nargs=2, default=[2, 6])
    sg.add_argument("--bushes-per-scene", type=int, nargs=2, default=[8, 26])
    sg.add_argument("--roof-clutter-per-scene", type=int, nargs=2, default=[3, 12])
    sg.add_argument(
        "--sparse-density-boost",
        type=float,
        default=4.0,
        help="synthsparse: multiply ground/vehicle point densities "
        "before scan degradation. The scan profiles were "
        "calibrated on real surveys whose raw density is far "
        "higher than the synth defaults; without the boost the "
        "degraded ground occupancy is ~1%% (vs the real 13-57%%) "
        "and the ground_shadow/shadow_edt channels come out "
        "all-zero.",
    )
    return p.parse_args()


def main():
    for _s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            _s.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    if args.workers is None:
        args.workers = max(1, (os.cpu_count() or 2) - 1)
    sources = (
        [s.strip() for s in args.sources.split(",")]
        if args.sources != "all"
        else ["synth", "dales", "yuto", "eclair", "isprs"]
    )

    out_dir = Path(args.out)
    for sp in ("train", "test"):
        (out_dir / sp).mkdir(parents=True, exist_ok=True)
    viz_dir = None if args.no_viz else out_dir / "viz"
    if viz_dir:
        for sp in ("train", "test"):
            (viz_dir / sp).mkdir(parents=True, exist_ok=True)

    cfg = dict(
        cell=args.cell,
        features=args.features,
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        tile_size=args.tile_size,
        veh_per_scene=args.veh_per_scene,
        ground_density=args.ground_density,
        veh_density=args.veh_density,
        ground_noise=args.ground_noise,
        max_slope=args.max_slope,
        trees_per_scene=args.trees_per_scene,
        houses_per_scene=args.houses_per_scene,
        bushes_per_scene=args.bushes_per_scene,
        roof_clutter_per_scene=args.roof_clutter_per_scene,
        source_limit_train=args.source_limit_train,
        source_limit_test=args.source_limit_test,
        yuto_test_frac=args.yuto_test_frac,
        npz_compress=not args.no_compress,
        dataset_version=DATASET_VERSION,
        mask_policy=MASK_POLICY,
        skip_existing=not args.overwrite_existing,
    )

    print(f"Output : {out_dir}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Workers: {args.workers}  cell={args.cell}m  features={args.features}")
    print(f"Viz    : {'off' if args.no_viz else viz_dir}")
    print(f"Existing: {'overwrite' if args.overwrite_existing else 'skip'}")
    t_total = time.perf_counter()

    summary = {}
    if "synth" in sources:
        summary["synth"] = run_synth(cfg, out_dir, viz_dir, args.workers)
    if "synthsparse" in sources:
        boost = max(1.0, float(args.sparse_density_boost))
        sp_cfg = dict(
            cfg,
            scan_profile=args.scan_profile,
            synth_name="synthsparse",
            seed=cfg["seed"] + 777,
            ground_density=[d * boost for d in cfg["ground_density"]],
            veh_density=[d * boost for d in cfg["veh_density"]],
        )
        summary["synthsparse"] = run_synth(sp_cfg, out_dir, viz_dir, args.workers)
    if "dales" in sources:
        summary["dales"] = run_dales(cfg, out_dir, viz_dir, args.workers)
    if "yuto" in sources:
        summary["yuto"] = run_yuto(cfg, out_dir, viz_dir, args.workers)
    if "eclair" in sources:
        summary["eclair"] = run_eclair(cfg, out_dir, viz_dir, args.workers)
    if "isprs" in sources:
        summary["isprs"] = run_isprs(cfg, out_dir, viz_dir, args.workers)
    if "wisar" in sources:
        if not args.wisar_dataset_dir:
            raise ValueError("--sources wisar requires --wisar-dataset-dir")
        from bordereye.ml.bev.data.wisar import integrate_dataset

        stats = integrate_dataset(
            Path(args.wisar_dataset_dir),
            out_dir,
            overwrite=args.overwrite_existing,
        )
        print(f"  wisar   : copied={stats['copied']} skipped={stats['skipped']}")

    print(f"\n{'=' * 60}")
    print(f"DONE  {time.perf_counter() - t_total:.1f}s total")
    for src, res in summary.items():
        train_n = sum(
            1 for _tid, stats in res if str(stats.get("split", "train")).lower() != "test"
        )
        test_n = len(res) - train_n
        print(f"  {src:8s}: {len(res):4d} tiles total  (train={train_n} test={test_n})")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
