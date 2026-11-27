"""WiSAR3D ingestion and leakage-safe mosaic construction for BEV."""
from __future__ import annotations

import io
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.cluster.hierarchy as hierarchy

from bordereye import RasterGrid
from bordereye.ml.bev.core.footprint_masks import fp_mask_from_boxes
from bordereye.ml.bev.data.bev_channels import (
    CHANNEL_BUILDER_VERSION,
    WIDTH_HOLE_VERSION,
    build_generic_v2,
)
from bordereye.raster_ops import fill_nodata_nn
from bordereye.terrain import csf_ground_mask, height_above_ground

WISAR3D_CLASSES = (
    "Vehicle", "Mattress", "Ball", "Gazebo", "Canopy", "Chair",
    "Small_table", "Big_table", "Tripod_pipe", "Barrel", "Tent", "Box",
    "Drone_case", "Small_bin", "Ramp", "Sitting_person", "Standing_person",
    "Cone", "Flag", "Pipe", "Porta_potty", "Play_tunnel",
)
WISAR3D_CLASS_TO_ID = {name: index for index, name in enumerate(WISAR3D_CLASSES)}
WISAR3D_DISTANCE_THRESHOLDS_M = (
    (0.19, 0.95, 1.72), (0.08, 0.38, 0.69), (0.03, 0.13, 0.23),
    (0.29, 1.44, 2.59), (0.32, 1.60, 2.88), (0.05, 0.24, 0.43),
    (0.06, 0.29, 0.52), (0.08, 0.41, 0.73), (0.06, 0.29, 0.52),
    (0.05, 0.26, 0.47), (0.21, 1.07, 1.92), (0.05, 0.23, 0.41),
    (0.07, 0.36, 0.64), (0.03, 0.16, 0.29), (0.07, 0.34, 0.62),
    (0.06, 0.31, 0.57), (0.05, 0.25, 0.45), (0.01, 0.06, 0.11),
    (0.06, 0.31, 0.56), (0.02, 0.10, 0.17), (0.11, 0.57, 1.02),
    (0.03, 0.17, 0.30),
)


def integrate_dataset(source_dataset: Path, target_dataset: Path, *,
                      overwrite: bool = False, dry_run: bool = False) -> dict[str, int]:
    """Copy schema-compatible WiSAR tiles into a unified BEV training dataset."""
    source_train = Path(source_dataset) / "train"
    target_train = Path(target_dataset) / "train"
    tiles = sorted(source_train.glob("wisar_*.npz"))
    if not tiles:
        raise FileNotFoundError(f"no WiSAR tiles under {source_train}")
    if not target_train.is_dir():
        raise FileNotFoundError(f"target train directory not found: {target_train}")
    reference = next(target_train.glob("*.npz"), None)
    if reference is None:
        raise FileNotFoundError(f"target dataset has no reference tile: {target_train}")
    with np.load(tiles[0], allow_pickle=False) as src, np.load(reference, allow_pickle=False) as ref:
        src_names = tuple(map(str, src["channel_names"].tolist()))
        ref_names = tuple(map(str, ref["channel_names"].tolist()))
    if src_names != ref_names:
        raise ValueError(f"WiSAR channel schema does not match target: {src_names} != {ref_names}")
    copied = skipped = 0
    for tile in tiles:
        destination = target_train / tile.name
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        if not dry_run:
            temporary = destination.with_name(destination.stem + ".copytmp.npz")
            shutil.copy2(tile, temporary)
            temporary.replace(destination)
        copied += 1
    return {"copied": copied, "skipped": skipped}


def _read_labels(labels_tar: tarfile.TarFile, frame: str) -> np.ndarray:
    member = labels_tar.getmember(f"label/{frame}.txt")
    lines = labels_tar.extractfile(member).read().decode("utf-8").splitlines()
    boxes = []
    for line in lines:
        fields = line.split()
        if len(fields) != 8 or fields[0].lower() != "vehicle":
            continue
        cx, cy, z, length, width, height, theta = map(float, fields[1:])
        if length < width:
            length, width = width, length
            theta += np.pi / 2
        boxes.append([cx, cy, z, length, width, height, theta])
    return np.asarray(boxes, np.float32).reshape(-1, 7)


def _read_labels_multiclass(
    labels_tar: tarfile.TarFile, frame: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read all official boxes and return canonical OBBs plus zero-based IDs."""
    member = labels_tar.getmember(f"label/{frame}.txt")
    lines = labels_tar.extractfile(member).read().decode("utf-8").splitlines()
    boxes, classes = [], []
    for line in lines:
        fields = line.split()
        if len(fields) != 8 or fields[0] not in WISAR3D_CLASS_TO_ID:
            continue
        cx, cy, z, length, width, height, theta = map(float, fields[1:])
        if length < width:
            length, width = width, length
            theta += np.pi / 2
        boxes.append([cx, cy, z, length, width, height, theta])
        classes.append(WISAR3D_CLASS_TO_ID[fields[0]])
    return (np.asarray(boxes, np.float32).reshape(-1, 7),
            np.asarray(classes, np.int16))


def _frame_members(wisar_dir: Path):
    result = {}
    for tar_path in sorted(wisar_dir.glob("points_*.tar")):
        with tarfile.open(tar_path) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".npy"):
                    result[Path(member.name).stem] = (tar_path, member.name)
    return result


def _cluster_boxes(boxes: np.ndarray, max_distance: float):
    if len(boxes) <= 1:
        return [np.arange(len(boxes))]
    labels = hierarchy.fclusterdata(boxes[:, :2], t=max_distance,
                                    criterion="distance", method="single")
    return [np.flatnonzero(labels == label) for label in np.unique(labels)]


def _box_mask(boxes, origin, cell, shape):
    empty = np.zeros(shape, np.uint8)
    return fp_mask_from_boxes(empty, boxes, tuple(origin), cell,
                              solid_fill=True, sparse_box_fill=True).astype(np.uint8)


def generate(wisar_dir: Path, out_dir: Path, csf_exe: Path, *, cell: float = 0.20,
             context_m: float = 18.0, cluster_eps_m: float = 12.0,
             max_window_m: float = 80.0, split: str = "train", frames=None,
             max_frames: int = 0) -> None:
    """Stream WiSAR tar files and create vehicle-centred 18-channel tiles."""
    wisar_dir, out_dir = Path(wisar_dir), Path(out_dir)
    members = _frame_members(wisar_dir)
    selected = sorted(set(members) & set(frames)) if frames is not None else sorted(members)
    if max_frames:
        selected = selected[:max_frames]
    target = out_dir / split
    target.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".csf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from bordereye.ml.bev.data.bev_dataset import save_bev_npz

    made = vehicles = 0
    with tarfile.open(wisar_dir / "labels.tar") as labels_tar:
        for frame in selected:
            boxes = _read_labels(labels_tar, frame)
            if not len(boxes):
                continue
            tar_path, member_name = members[frame]
            with tarfile.open(tar_path) as points_tar:
                raw = points_tar.extractfile(member_name).read()
            points = np.load(io.BytesIO(raw), allow_pickle=False)
            xyz = np.asarray(points[:, :3], np.float64)
            ground = csf_ground_mask(xyz, csf_exe, cache=cache_dir / f"{frame}.npy")
            if ground.sum() < 10:
                continue
            dtm = fill_nodata_nn(RasterGrid.from_point_cloud(xyz[ground], cell_size=cell), stat="min")
            hag = height_above_ground(xyz, dtm)
            for group_id, indices in enumerate(_cluster_boxes(boxes, cluster_eps_m)):
                group_boxes = boxes[indices]
                lo = group_boxes[:, :2].min(axis=0) - context_m
                hi = group_boxes[:, :2].max(axis=0) + context_m
                size = np.minimum(hi - lo, max_window_m)
                center = (lo + hi) * 0.5
                lo, hi = center - size * 0.5, center + size * 0.5
                inside = ((xyz[:, 0] >= lo[0]) & (xyz[:, 0] < hi[0]) &
                          (xyz[:, 1] >= lo[1]) & (xyz[:, 1] < hi[1]))
                if inside.sum() < 32:
                    continue
                bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
                channels, raster, names = build_generic_v2(xyz[inside], hag[inside], cell, bounds)
                local = boxes[(boxes[:, 0] >= lo[0]) & (boxes[:, 0] < hi[0]) &
                              (boxes[:, 1] >= lo[1]) & (boxes[:, 1] < hi[1])]
                seg = _box_mask(local, raster.origin, cell, raster.shape)
                args = SimpleNamespace(cell=cell, features="generic_v2", npz_compress=True,
                                       dataset_version="bev_dataset_v7",
                                       bev_schema="generic_v2_18",
                                       mask_policy="solid_vehicle_obb",
                                       pseudo_obb_policy="wisar_labels")
                save_bev_npz(target / f"wisar_{frame}_{group_id:02d}.npz", channels, raster,
                             local, np.zeros((0, 7), np.float32),
                             np.zeros((0, 7), np.float32), seg, names, args,
                             "wisar", split, seg_box_mask=seg, fp_mask=seg,
                             extra={"source_frame": np.asarray(frame),
                                    "veh_width_hole_version": np.asarray(WIDTH_HOLE_VERSION)})
                made += 1
                vehicles += len(local)
    print(f"WiSAR: wrote {made} tiles with {vehicles} vehicle boxes to {target}")


def generate_full_strips(
    wisar_dir: Path,
    out_dir: Path,
    csf_exe: Path,
    *,
    cell: float = 0.20,
    split: str,
    frames,
    all_classes: bool = False,
    csf_cache_dir: Path | None = None,
) -> None:
    """Create one full-scene BEV tile per official WiSAR3D strip.

    Unlike :func:`generate`, this does not use GT boxes to select inference
    windows and also retains strips without vehicles. It is therefore suitable
    for the official detection benchmark, including false-positive scoring.
    """
    wisar_dir, out_dir = Path(wisar_dir), Path(out_dir)
    members = _frame_members(wisar_dir)
    selected = sorted(set(members) & {str(frame).zfill(6) for frame in frames})
    target = out_dir / split
    target.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(csf_cache_dir) if csf_cache_dir is not None else out_dir / ".csf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    from bordereye.ml.bev.data.bev_dataset import save_bev_npz

    by_archive: dict[Path, list[tuple[str, str]]] = {}
    for frame in selected:
        tar_path, member_name = members[frame]
        by_archive.setdefault(tar_path, []).append((frame, member_name))

    made = objects = 0
    with tarfile.open(wisar_dir / "labels.tar") as labels_tar:
        for tar_path, frame_members in sorted(by_archive.items()):
            with tarfile.open(tar_path) as points_tar:
                for frame, member_name in frame_members:
                    final = target / f"wisar_{frame}.npz"
                    if final.exists():
                        made += 1
                        with np.load(final, allow_pickle=False) as existing:
                            objects += len(np.asarray(existing["boxes_valid"]).reshape(-1, 7))
                        continue
                    raw = points_tar.extractfile(member_name).read()
                    points = np.load(io.BytesIO(raw), allow_pickle=False)
                    xyz = np.asarray(points[:, :3], np.float64)
                    if all_classes:
                        boxes, box_classes = _read_labels_multiclass(labels_tar, frame)
                    else:
                        boxes = _read_labels(labels_tar, frame)
                        box_classes = np.zeros(len(boxes), np.int16)
                    ground = csf_ground_mask(
                        xyz, csf_exe, cache=cache_dir / f"{frame}.npy")
                    if ground.sum() < 10:
                        print(f"[skip] {frame}: fewer than 10 ground points")
                        continue
                    dtm = fill_nodata_nn(
                        RasterGrid.from_point_cloud(xyz[ground], cell_size=cell),
                        stat="min")
                    hag = height_above_ground(xyz, dtm)
                    lo = np.floor(xyz[:, :2].min(axis=0) / cell) * cell
                    hi = np.ceil(xyz[:, :2].max(axis=0) / cell) * cell
                    bounds = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
                    channels, raster, names = build_generic_v2(xyz, hag, cell, bounds)
                    seg = _box_mask(boxes, raster.origin, cell, raster.shape)
                    args = SimpleNamespace(
                        cell=cell,
                        features="generic_v2",
                        npz_compress=True,
                        dataset_version="bev_dataset_v7",
                        bev_schema="generic_v2_18",
                        mask_policy="solid_vehicle_obb",
                        pseudo_obb_policy="wisar_labels",
                    )
                    save_bev_npz(
                        final,
                        channels,
                        raster,
                        boxes,
                        np.zeros((0, 7), np.float32),
                        np.zeros((0, 7), np.float32),
                        seg,
                        names,
                        args,
                        "wisar",
                        split,
                        seg_box_mask=seg,
                        fp_mask=seg,
                        extra={
                            "source_frame": np.asarray(frame),
                            "official_wisar_split": np.asarray(split),
                            "box_classes": box_classes,
                            "class_names": np.asarray(
                                WISAR3D_CLASSES if all_classes else ("Vehicle",)),
                            "veh_width_hole_version": np.asarray(WIDTH_HOLE_VERSION),
                        },
                    )
                    made += 1
                    objects += len(boxes)
                    print(
                        f"[{split}] {made}/{len(selected)} {frame}: "
                        f"{len(xyz):,} points, {len(boxes)} objects, "
                        f"{channels.shape[2]}x{channels.shape[1]} px",
                        flush=True,
                    )
    print(
        f"WiSAR official {split}: wrote {made} full strips with "
        f"{objects} object boxes to {target}")


def mosaic_tiles(tile_paths, out_dir: Path, *, grid: int = 3, gap_px: int = 4,
                 split: str = "train") -> None:
    """Pack tiles while validating schema/cell and retaining source provenance."""
    paths = [Path(path) for path in tile_paths]
    if not paths:
        return
    metadata = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata.append((path, tuple(data["channels"].shape), float(data["cell"]),
                             tuple(str(x) for x in data["channel_names"].tolist())))
    cells = {round(item[2], 9) for item in metadata}
    schemas = {item[3] for item in metadata}
    channel_counts = {item[1][0] for item in metadata}
    if len(cells) != 1 or len(schemas) != 1 or len(channel_counts) != 1:
        raise ValueError("WiSAR mosaic inputs have mixed cell sizes or channel schemas")
    cell = metadata[0][2]
    names = metadata[0][3]
    slot = max(max(shape[1:]) for _, shape, _, _ in metadata) + gap_px
    side = grid * slot
    target = Path(out_dir) / split
    target.mkdir(parents=True, exist_ok=True)

    for mosaic_id, offset in enumerate(range(0, len(paths), grid * grid)):
        group = paths[offset:offset + grid * grid]
        channels = np.zeros((len(names), side, side), np.float32)
        seg = np.zeros((side, side), np.uint8)
        fp = np.zeros_like(seg)
        boxes = []
        source_tiles = []
        source_frames = []
        for index, path in enumerate(group):
            row_slot, col_slot = divmod(index, grid)
            row0, col0 = row_slot * slot, col_slot * slot
            with np.load(path, allow_pickle=False) as data:
                ch = data["channels"].astype(np.float32)
                h, w = ch.shape[1:]
                channels[:, row0:row0+h, col0:col0+w] = ch
                seg[row0:row0+h, col0:col0+w] = data["seg_mask"]
                if "fp_mask" in data:
                    fp[row0:row0+h, col0:col0+w] = data["fp_mask"]
                for box in np.asarray(data["boxes_valid"], np.float32).reshape(-1, 7):
                    source_col = (box[0] - float(data["origin"][0])) / cell
                    source_row = h - (box[1] - float(data["origin"][1])) / cell
                    boxes.append([(source_col + col0) * cell,
                                  (side - source_row - row0) * cell, *box[2:]])
                source_tiles.append(path.name)
                source_frames.append(str(data["source_frame"]) if "source_frame" in data else path.stem)
        payload = dict(channels=channels, origin=np.zeros(2), cell=np.float64(cell),
                       shape=np.asarray((side, side)), boxes=np.asarray(boxes, np.float32).reshape(-1, 7),
                       boxes_valid=np.asarray(boxes, np.float32).reshape(-1, 7),
                       boxes_pseudo=np.zeros((0, 7), np.float32), boxes_ignore=np.zeros((0, 7), np.float32),
                       seg_mask=seg, fp_mask=fp, seg_box_mask=fp,
                       channel_names=np.asarray(names), features=np.asarray("generic_v2"),
                       source=np.asarray("wisar"), source_split=np.asarray(split),
                       dataset_version=np.asarray("bev_dataset_v7"),
                       bev_schema=np.asarray("generic_v2_18"),
                       channel_builder_version=np.asarray(CHANNEL_BUILDER_VERSION),
                       veh_width_hole_version=np.asarray(WIDTH_HOLE_VERSION),
                       source_tiles=np.asarray(source_tiles), source_frames=np.asarray(source_frames))
        tmp = target / f"wisar_m{mosaic_id:05d}.tmp.npz"
        final = target / f"wisar_m{mosaic_id:05d}.npz"
        np.savez_compressed(tmp, **payload)
        tmp.replace(final)
