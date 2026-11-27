"""Unified DALES vehicle instance dataset pipeline.

One pipeline, two instance sources:

* **Instance DALES** (DALESObjects ``*_new.ply``): per-point ``ins_class`` gives
  the *real* car/truck instances.
* **Semantic DALES** (``*.las``): only per-point semantic ``classification``
  (3=car, 4=truck) — instances are *derived* by per-class DBSCAN clustering
  (pseudo ground truth).

Both yield a :class:`VehicleTile` carrying ``(xyz, intensity, sem_class,
instance_id)`` per point.  The rest is identical for both:

    1. hole + watershed-blob candidate detection on the raster/DTM
    2. fit an L-shape OBB per candidate (this is the training box)
    3. match each candidate OBB against the real/derived instances
       (point coverage & purity inside the OBB)
    4. if the match is good -> extract the candidate-OBB point cloud as a
       labelled (car/truck) training sample

This module only depends on ``bordereye`` + numpy (+ sklearn via instances,
matplotlib via viz, laspy via io — all lazily).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np

from ._masks import in_classes
from .geometry import convex_hull_2d, oriented_bounding_box_2d, polygon_iou, rect_corners_2d
from .holes import find_raster_holes
from .instances import DbscanConfig, segment_instances_dbscan
from .io import read_dales_ply, read_las_data
from .lshape import LShapeResult, fit_lshape
from .raster import RasterGrid
from .raster_ops import fill_nodata_nn
from .terrain import height_above_ground
from .watershed import watershed_blobs

GROUND_CLASS = 1
CAR_CLASS = 3
TRUCK_CLASS = 4
VEHICLE_CLASSES = (CAR_CLASS, TRUCK_CLASS)
CLASS_NAMES = {CAR_CLASS: "car", TRUCK_CLASS: "truck"}
LABEL_OF = {CAR_CLASS: 1, TRUCK_CLASS: 2}     # common NN labels: non_vehicle=0, car=1, truck=2


# ─────────────────────────────────────────────────────────────────────────────
# Tile = points + semantic class + (real or derived) instance id
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class VehicleTile:
    tile: str
    split: str
    source: Literal["ply", "las"]
    xyz: np.ndarray          # (N,3) float64
    intensity: np.ndarray    # (N,)  float32
    sem_class: np.ndarray    # (N,)  int32
    instance_id: np.ndarray  # (N,)  uint32  (0 = none; vehicle instances only)

    def ground_mask(self) -> np.ndarray:
        return self.sem_class == GROUND_CLASS


def load_tile_ply(path: str | Path, split: str) -> VehicleTile:
    """Real instances from DALESObjects ins_class (vehicle points only)."""
    path = Path(path)
    data = read_dales_ply(path)
    xyz = np.column_stack([
        data["x"].astype(np.float64),
        data["y"].astype(np.float64),
        data["z"].astype(np.float64),
    ])
    sem = data["sem_class"].astype(np.int32)
    ins = data["ins_class"].astype(np.int64)
    # keep instance ids only for vehicle points (others -> 0)
    veh = in_classes(sem, VEHICLE_CLASSES) & (ins != 0)
    instance_id = np.zeros(len(sem), dtype=np.uint32)
    if np.any(veh):
        _, inv = np.unique(ins[veh], return_inverse=True)
        instance_id[veh] = inv.astype(np.uint32) + 1
    return VehicleTile(
        tile=path.stem.replace("_new", ""), split=split, source="ply",
        xyz=xyz, intensity=data["intensity"].astype(np.float32),
        sem_class=sem, instance_id=instance_id,
    )


def load_tile_las(
    path: str | Path,
    split: str,
    *,
    dbscan: DbscanConfig | None = None,
) -> VehicleTile:
    """Derived instances from LAS semantic class via per-class DBSCAN."""
    path = Path(path)
    cfg = dbscan or DbscanConfig(eps=0.5, min_samples=6, min_cluster_points=10)
    las = read_las_data(path)
    sem = np.asarray(las.classification, dtype=np.int32)
    xyz = np.column_stack([
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        np.asarray(las.z, dtype=np.float64),
    ])
    intensity = np.asarray(las.intensity, dtype=np.float32)

    instance_id = np.zeros(len(sem), dtype=np.uint32)
    next_id = 0
    for veh_cls in VEHICLE_CLASSES:
        sel = np.where(sem == veh_cls)[0]
        if len(sel) < cfg.min_cluster_points:
            continue
        ids, _stats = segment_instances_dbscan(xyz[sel], cfg)
        keep = ids > 0
        if np.any(keep):
            instance_id[sel[keep]] = ids[keep].astype(np.uint32) + next_id
            next_id += int(ids.max())
    return VehicleTile(
        tile=path.stem, split=split, source="las",
        xyz=xyz, intensity=intensity, sem_class=sem, instance_id=instance_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candidate detection: raster holes + watershed blobs  (-> L-shape OBB)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CandidateConfig:
    cell_size: float = 0.25
    hag_min: float = 0.25
    hag_max: float = 5.0
    hole_min_cells: int = 4
    veh_min_len: float = 1.2
    veh_max_len: float = 16.0
    veh_min_wid: float = 0.6
    veh_max_wid: float = 4.5
    watershed_peak_dist: float = 1.2
    watershed_peak_height: float = 0.3
    watershed_sigma: float = 0.8
    # cell-centre footprints underestimate the true extent by half a cell on
    # each side; expand the fitted OBB by +cell_size in length & width.
    expand_half_cell: bool = True


@dataclass(frozen=True)
class VehicleCandidate:
    source: Literal["hole", "blob", "rect"]
    ls: LShapeResult

    def obb_corners(self) -> np.ndarray:
        return ls_to_corners(self.ls)


def detect_candidates_rect(
    tile: VehicleTile,
    cfg=None,
    *,
    hag: np.ndarray | None = None,
) -> list[VehicleCandidate]:
    """Candidates from the rectangular shape-prior search (rect_search).

    A drop-in alternative to :func:`detect_candidates` (hole + watershed blob):
    runs :func:`bordereye.rect_search.detect_vehicles_rect` and wraps each
    rotated-rectangle detection as a ``source="rect"`` :class:`VehicleCandidate`,
    so the rest of :func:`build_tile_dataset` (instance matching, patch
    extraction, labelling) is identical.
    """
    from .rect_search import RectSearchConfig, detect_vehicles_rect

    rcfg = cfg or RectSearchConfig()
    dets = detect_vehicles_rect(tile.xyz, tile.ground_mask(), hag=hag, config=rcfg)
    out: list[VehicleCandidate] = []
    for d in dets:
        ls = LShapeResult(
            vertices=d.vertices, center=d.center, length=d.length, width=d.width,
            rotation_deg=d.rotation_deg, area=d.length * d.width,
            criterion="variance", criterion_value=d.score,
        )
        out.append(VehicleCandidate(source="rect", ls=ls))
    return out


def ls_to_corners(ls: LShapeResult) -> np.ndarray:
    """Four world-XY corners of an L-shape OBB (for drawing / extraction)."""
    return rect_corners_2d(
        ls.center[0], ls.center[1], ls.length, ls.width, np.deg2rad(ls.rotation_deg)
    )


def _hole_cells_to_world(cells: np.ndarray, raster: RasterGrid) -> np.ndarray:
    rows, cols = cells[:, 0], cells[:, 1]
    x = raster.origin[0] + (cols + 0.5) * raster.cell_size[0]
    y = raster.origin[1] + (raster.height - 1 - rows + 0.5) * raster.cell_size[1]
    return np.column_stack((x, y))


def _comp_coords_to_world(coords: np.ndarray, raster: RasterGrid) -> np.ndarray:
    cols, urows = coords[:, 0], coords[:, 1]
    x = raster.origin[0] + (cols + 0.5) * raster.cell_size[0]
    y = raster.origin[1] + (urows + 0.5) * raster.cell_size[1]
    return np.column_stack((x, y))


def compute_tile_hag(tile: VehicleTile, cfg: CandidateConfig | None = None) -> np.ndarray:
    """Per-point height-above-ground for all tile points (NaN where undefined)."""
    cfg = cfg or CandidateConfig()
    ground_xyz = tile.xyz[tile.ground_mask()]
    if len(ground_xyz) < 3:
        return np.full(len(tile.xyz), np.nan)
    dtm_raw = RasterGrid.from_point_cloud(ground_xyz, cell_size=cfg.cell_size)
    dtm_filled = fill_nodata_nn(dtm_raw, stat="min")
    return height_above_ground(tile.xyz, dtm_filled)


def detect_candidates(
    tile: VehicleTile,
    cfg: CandidateConfig | None = None,
    *,
    hag: np.ndarray | None = None,
) -> list[VehicleCandidate]:
    cfg = cfg or CandidateConfig()
    xyz = tile.xyz
    ground_xyz = xyz[tile.ground_mask()]
    if len(ground_xyz) < 3:
        return []

    dtm_raw = RasterGrid.from_point_cloud(ground_xyz, cell_size=cfg.cell_size)
    if hag is None:
        hag = compute_tile_hag(tile, cfg)
    hag_mask = np.isfinite(hag) & (hag >= cfg.hag_min) & (hag <= cfg.hag_max)
    hag_xyz = xyz[hag_mask].copy()
    hag_xyz[:, 2] = hag[hag_mask]

    pad = cfg.cell_size if cfg.expand_half_cell else 0.0  # +half cell each side

    def _correct(ls: LShapeResult) -> LShapeResult:
        if pad <= 0.0:
            return ls
        length = ls.length + pad
        width = ls.width + pad
        return replace(ls, length=length, width=width, area=length * width)

    def _accept(ls: LShapeResult) -> bool:
        return (cfg.veh_min_len <= ls.length <= cfg.veh_max_len
                and cfg.veh_min_wid <= ls.width <= cfg.veh_max_wid)

    candidates: list[VehicleCandidate] = []

    # (a) DTM holes — vehicles punch holes in the ground surface
    for hole in find_raster_holes(dtm_raw, min_cells=cfg.hole_min_cells, snap_edge=False):
        cell_xy = _hole_cells_to_world(hole.cells, dtm_raw)
        if len(cell_xy) < 2:
            continue
        try:
            ls = _correct(fit_lshape(cell_xy, criterion="variance", angle_resolution_deg=1.0))
        except ValueError:
            continue
        if _accept(ls):
            candidates.append(VehicleCandidate("hole", ls))

    # (b) watershed blobs on the HAG raster — splits side-by-side vehicles
    if len(hag_xyz) >= cfg.hole_min_cells:
        blob_raster = RasterGrid.from_point_cloud(hag_xyz, cell_size=cfg.cell_size)
        for comp in watershed_blobs(
            hag_xyz, cell_size=cfg.cell_size,
            min_peak_distance_m=cfg.watershed_peak_dist,
            min_peak_height=cfg.watershed_peak_height,
            smooth_sigma=cfg.watershed_sigma,
            min_component_size=cfg.hole_min_cells,
        ):
            cell_xy = _comp_coords_to_world(comp.coords, blob_raster)
            if len(cell_xy) < 2:
                continue
            try:
                ls = _correct(fit_lshape(cell_xy, criterion="variance", angle_resolution_deg=1.0))
            except ValueError:
                continue
            if _accept(ls):
                candidates.append(VehicleCandidate("blob", ls))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Spatial index + point-in-OBB
# ─────────────────────────────────────────────────────────────────────────────
class _GridIndex:
    """Uniform-grid XY index for fast AABB point lookup."""

    def __init__(self, xyz: np.ndarray, cell_size: float = 5.0):
        self.xyz = xyz
        self.cs = cell_size
        self.ox = float(xyz[:, 0].min())
        self.oy = float(xyz[:, 1].min())
        ci = ((xyz[:, 0] - self.ox) / cell_size).astype(np.int64)
        ri = ((xyz[:, 1] - self.oy) / cell_size).astype(np.int64)
        self.ncols = int(ci.max()) + 1
        keys = ri * self.ncols + ci
        order = np.argsort(keys, kind="stable")
        self._order = order
        sorted_keys = keys[order]
        boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_keys)]))
        self._ranges = {
            int(sorted_keys[s]): (int(s), int(e))
            for s, e in zip(starts, ends, strict=True)
        }

    def query_aabb(self, cx: float, cy: float, half: float) -> np.ndarray:
        c0 = max(0, int((cx - half - self.ox) / self.cs))
        c1 = int((cx + half - self.ox) / self.cs)
        r0 = max(0, int((cy - half - self.oy) / self.cs))
        r1 = int((cy + half - self.oy) / self.cs)
        out = []
        for r in range(r0, r1 + 1):
            base = r * self.ncols
            for c in range(c0, c1 + 1):
                rng = self._ranges.get(base + c)
                if rng is not None:
                    out.append(self._order[rng[0]:rng[1]])
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


def obb_point_indices(xyz: np.ndarray, ls: LShapeResult,
                      index: _GridIndex | None = None) -> np.ndarray:
    """Indices of points inside the OBB (no full-length allocation)."""
    cx, cy = ls.center
    half = 0.5 * float(np.hypot(ls.length, ls.width)) + 0.5
    if index is not None:
        cand = index.query_aabb(cx, cy, half)
    else:
        cand = np.where((np.abs(xyz[:, 0] - cx) <= half) & (np.abs(xyz[:, 1] - cy) <= half))[0]
    if len(cand) == 0:
        return cand
    sub = xyz[cand]
    a = np.deg2rad(ls.rotation_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    dx = sub[:, 0] - cx
    dy = sub[:, 1] - cy
    xo = dx * ca - dy * sa
    yo = dx * sa + dy * ca
    hit = (np.abs(xo) <= ls.length / 2 + 1e-3) & (np.abs(yo) <= ls.width / 2 + 1e-3)
    return cand[hit]


def points_in_obb(xyz: np.ndarray, ls: LShapeResult, index: _GridIndex | None = None) -> np.ndarray:
    out = np.zeros(len(xyz), dtype=bool)
    out[obb_point_indices(xyz, ls, index)] = True
    return out


def _sorted_obb_indices(
    xyz: np.ndarray, ls: LShapeResult, index: _GridIndex | None = None,
) -> np.ndarray:
    """Ascending in-OBB indices — the same selection, and order, as
    ``np.flatnonzero(points_in_obb(...))`` but without the full-length mask."""
    indices = obb_point_indices(xyz, ls, index)
    indices.sort()
    return indices


# ─────────────────────────────────────────────────────────────────────────────
# Match candidate OBBs against instances + extract point clouds
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MatchConfig:
    purity_min: float = 0.7      # dominant-instance pts / vehicle pts inside OBB
    coverage_min: float = 0.5    # dominant-instance pts inside OBB / instance total
    min_points: int = 20         # min vehicle pts for a positive
    iou_min: float = 0.0         # min OBB IoU vs matched instance (0 = record only)
    feature: Literal["height", "intensity"] = "height"
    # vehicle/non-vehicle dataset extras
    neg_ratio_max: float = 0.03  # vehicle-pt ratio (of off-ground pts) <= this -> non_vehicle
    hag_patch_min: float = 0.10  # min HAG to keep a point in the extracted patch
    hag_patch_max: float = 5.5   # max HAG to keep a point in the extracted patch


@dataclass
class ExtractedObject:
    xyz_norm: np.ndarray         # (M,3) float32, centred + unit-scaled
    feat_norm: np.ndarray        # (M,1) float32, min-max
    label: int                   # 0=car, 1=truck
    sem_class: int
    instance_id: int
    purity: float
    coverage: float
    obb_iou: float               # geometric IoU vs matched instance OBB (quality)
    n_points: int
    source: str                  # candidate source: hole/blob
    obb_corners: np.ndarray      # (4,2) world XY


@dataclass
class TileResult:
    tile: str
    split: str
    objects: list[ExtractedObject] = field(default_factory=list)
    n_candidates: int = 0
    n_non_vehicle: int = 0       # candidates with no vehicle instance inside
    n_ignored: int = 0           # vehicle pts present but match too weak


@dataclass
class CandidateDiag:
    """Per-candidate matching diagnostics for debug visualisation."""
    source: str                       # "hole" | "blob"
    cand_corners: np.ndarray          # (4,2) candidate OBB (after half-cell correction)
    iou: float                        # OBB IoU vs dominant instance (0 if no vehicle)
    purity: float
    coverage: float
    accepted: bool                    # passed all thresholds -> goes into the dataset
    class_name: str | None            # dominant instance class (car/truck) or None
    inst_id: int                      # matched instance id (0 if none)
    inst_corners: np.ndarray | None   # (4,2) matched real/derived instance OBB or None


def _instance_metadata(
    xyz: np.ndarray,
    sem: np.ndarray,
    inst: np.ndarray,
    *,
    fit_obbs: bool,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, int],
    dict[int, np.ndarray | None],
]:
    """Group positive instances once and derive all shared metadata."""
    positive = np.flatnonzero(inst > 0)
    if len(positive) == 0:
        return {}, {}, {}
    order = np.argsort(inst[positive], kind="stable")
    ordered = positive[order]
    labels = inst[ordered]
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    groups = {
        int(labels[start]): group
        for start, group in zip(starts, np.split(ordered, starts[1:]), strict=True)
    }
    semantics: dict[int, int] = {}
    obbs: dict[int, np.ndarray | None] = {}
    for instance_id, indices in groups.items():
        classes = sem[indices]
        positive_classes = classes[classes > 0]
        semantics[instance_id] = (
            int(np.bincount(positive_classes).argmax()) if len(positive_classes) else 0
        )
        if fit_obbs:
            try:
                obbs[instance_id] = oriented_bounding_box_2d(xyz[indices, :2]).vertices
            except ValueError:
                obbs[instance_id] = None
    return groups, semantics, obbs


def diagnose_candidates(
    tile: VehicleTile,
    candidates: list[VehicleCandidate],
    cfg: MatchConfig | None = None,
) -> tuple[list[CandidateDiag], list[tuple[str, np.ndarray]]]:
    """Match candidates and return per-candidate diagnostics + all instance OBBs.

    Unlike :func:`match_and_extract` this keeps *every* candidate (hole and
    blob, matched or not) and reports the IoU against the matched instance —
    intended for debug rasters comparing hole/blob OBB vs the real OBB.
    """
    cfg = cfg or MatchConfig()
    xyz, sem, inst = tile.xyz, tile.sem_class, tile.instance_id
    inst_groups, inst_sem, inst_obb = _instance_metadata(xyz, sem, inst, fit_obbs=True)

    index = _GridIndex(xyz, cell_size=5.0)
    diags: list[CandidateDiag] = []

    for cand in candidates:
        corners = cand.obb_corners()
        inst_in = inst[_sorted_obb_indices(xyz, cand.ls, index)]
        veh = inst_in > 0
        if not veh.any():
            diags.append(CandidateDiag(cand.source, corners, 0.0, 0.0, 0.0, False, None, 0, None))
            continue
        ids, counts = np.unique(inst_in[veh], return_counts=True)
        k = int(np.argmax(counts))
        dom_id, n_dom, n_veh = int(ids[k]), int(counts[k]), int(veh.sum())
        purity = n_dom / max(n_veh, 1)
        coverage = n_dom / max(len(inst_groups[dom_id]), 1)
        sem_cls = inst_sem.get(dom_id, 0)
        ic = inst_obb.get(dom_id)
        iou = polygon_iou(corners, ic) if ic is not None else 0.0
        accepted = (purity >= cfg.purity_min and coverage >= cfg.coverage_min
                    and n_dom >= cfg.min_points and iou >= cfg.iou_min
                    and sem_cls in VEHICLE_CLASSES)
        diags.append(CandidateDiag(
            cand.source, corners, round(iou, 4), round(purity, 4), round(coverage, 4),
            accepted, CLASS_NAMES.get(sem_cls), dom_id, ic,
        ))

    instance_obbs = [
        (CLASS_NAMES[inst_sem[iid]], inst_obb[iid])
        for iid in inst_groups
        if inst_obb[iid] is not None and inst_sem[iid] in VEHICLE_CLASSES
    ]
    return diags, instance_obbs


def to_obb_frame(xyz: np.ndarray, ls: LShapeResult) -> tuple[np.ndarray, np.ndarray]:
    """Rotate XY into the OBB frame (centred, axis-aligned). Returns (x_obb, y_obb)."""
    cx, cy = ls.center
    a = np.deg2rad(ls.rotation_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    dx = xyz[:, 0] - cx
    dy = xyz[:, 1] - cy
    return dx * ca - dy * sa, dx * sa + dy * ca


def _shape_features_xy(xy: np.ndarray, ls: LShapeResult) -> tuple[float, float]:
    """rectangularity = hull_area / OBB_area; compactness = perimeter^2 / hull_area.

    Pure-numpy convex hull + shoelace area + edge-length perimeter (no shapely).
    """
    obb_area = max(ls.length * ls.width, 1e-9)
    hull = convex_hull_2d(xy)
    if len(hull) >= 3:
        x, y = hull[:, 0], hull[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        hull_area = max(0.5 * abs(float(np.sum(x * yn - xn * y))), 1e-9)
        perimeter = float(np.sum(np.hypot(xn - x, yn - y)))
    else:
        hull_area, perimeter = obb_area, 4.0 * float(np.sqrt(obb_area))
    rectangularity = min(1.0, hull_area / obb_area)
    compactness = (perimeter ** 2) / hull_area
    return float(rectangularity), float(compactness)


@dataclass
class CandidateRecord:
    """One training row: OBB-frame patch + shape features + vehicle/non-vehicle label."""
    label: str                   # "car" | "truck" | "non_vehicle" | "ignore"
    patch: np.ndarray            # (N,3) float32 [x_obb, y_obb, z_hag]
    source: str                  # "hole" | "blob" | "rect"
    center_x: float
    center_y: float
    length: float
    width: float
    height_3d: float
    rectangularity: float
    compactness: float
    veh_ratio: float
    purity: float
    coverage: float
    obb_iou: float
    instance_id: int
    n_points_patch: int
    obb_corners: np.ndarray      # (4,2) world XY


def build_tile_dataset(
    tile: VehicleTile,
    cand_cfg: CandidateConfig | None = None,
    match_cfg: MatchConfig | None = None,
    *,
    candidates: list[VehicleCandidate] | None = None,
) -> list[CandidateRecord]:
    """Full per-tile vehicle/non-vehicle dataset rows (unified for LAS & PLY).

    For each hole/blob candidate: extract the OBB-frame point patch, compute
    shape features, match against instances and assign vehicle / non_vehicle /
    ignore.  Single HAG computation shared with detection.
    """
    cand_cfg = cand_cfg or CandidateConfig()
    match_cfg = match_cfg or MatchConfig()
    xyz, sem, inst = tile.xyz, tile.sem_class, tile.instance_id

    hag = compute_tile_hag(tile, cand_cfg)
    if candidates is None:
        candidates = detect_candidates(tile, cand_cfg, hag=hag)
    if not candidates:
        return []

    inst_groups, inst_sem, inst_obb = _instance_metadata(xyz, sem, inst, fit_obbs=True)

    index = _GridIndex(xyz, cell_size=5.0)
    records: list[CandidateRecord] = []

    hag_finite = np.nan_to_num(hag, nan=-1e9)  # NaN -> fails band test cheaply

    for cand in candidates:
        ls = cand.ls
        in_idx = obb_point_indices(xyz, ls, index)
        if len(in_idx) == 0:
            continue
        sem_in = sem[in_idx]
        inst_in = inst[in_idx]
        hag_in = hag_finite[in_idx]
        n_offg = int(np.count_nonzero(sem_in != GROUND_CLASS))

        # vehicle-point ratio over OFF-GROUND points (avoids ground dilution)
        veh_pt = in_classes(sem_in, VEHICLE_CLASSES)
        n_veh_pt = int(veh_pt.sum())
        veh_ratio = n_veh_pt / max(n_offg, 1)

        # instance match (geometry + point coverage/purity)
        purity = coverage = iou = 0.0
        dom_id = 0
        veh = inst_in > 0
        if veh.any():
            ids, counts = np.unique(inst_in[veh], return_counts=True)
            k = int(np.argmax(counts))
            dom_id = int(ids[k])
            n_dom = int(counts[k])
            purity = n_dom / max(int(veh.sum()), 1)
            coverage = n_dom / max(len(inst_groups.get(dom_id, ())), 1)
            ic = inst_obb.get(dom_id)
            iou = polygon_iou(cand.obb_corners(), ic) if ic is not None else 0.0

        is_veh = (dom_id > 0 and inst_sem.get(dom_id, 0) in VEHICLE_CLASSES
                  and purity >= match_cfg.purity_min
                  and coverage >= match_cfg.coverage_min
                  and iou >= match_cfg.iou_min
                  and n_veh_pt >= match_cfg.min_points)
        if is_veh:
            label = CLASS_NAMES[inst_sem[dom_id]]
        elif veh_ratio <= match_cfg.neg_ratio_max:
            label = "non_vehicle"
        else:
            label = "ignore"

        # OBB-frame patch (points within HAG band) — work on the in-OBB subset only
        band = (hag_in >= match_cfg.hag_patch_min) & (hag_in <= match_cfg.hag_patch_max)
        kp = in_idx[band]
        if len(kp):
            sub_xyz = xyz[kp]
            xo, yo = to_obb_frame(sub_xyz, ls)
            hg = hag[kp]
            patch = np.column_stack([
                xo.astype(np.float32), yo.astype(np.float32),
                hg.astype(np.float32),
            ])
            h3d = float(hg.max() - hg.min())
            rect, comp = _shape_features_xy(np.column_stack([xo, yo]), ls)
        else:
            patch = np.zeros((0, 3), np.float32)
            h3d, rect, comp = 0.0, 0.0, 0.0

        records.append(CandidateRecord(
            label=label, patch=patch, source=cand.source,
            center_x=round(ls.center[0], 3), center_y=round(ls.center[1], 3),
            length=round(ls.length, 4), width=round(ls.width, 4), height_3d=round(h3d, 4),
            rectangularity=round(rect, 4), compactness=round(comp, 4),
            veh_ratio=round(veh_ratio, 4), purity=round(purity, 4),
            coverage=round(coverage, 4), obb_iou=round(iou, 4),
            instance_id=dom_id, n_points_patch=len(kp), obb_corners=cand.obb_corners(),
        ))
    return records


def _normalise_object(xyz: np.ndarray, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = xyz.astype(np.float32)
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(xyz[:, :2], axis=1).max()
    if scale < 1e-6:
        scale = np.linalg.norm(xyz, axis=1).max()
    if scale > 1e-6:
        xyz = xyz / float(scale)
    feat = feat.astype(np.float32)
    lo, hi = float(feat.min()), float(feat.max())
    feat = (feat - lo) / (hi - lo) if hi > lo else np.zeros_like(feat)
    return xyz.astype(np.float32), feat.reshape(-1, 1).astype(np.float32)


def match_and_extract(
    tile: VehicleTile,
    candidates: list[VehicleCandidate],
    cfg: MatchConfig | None = None,
) -> TileResult:
    cfg = cfg or MatchConfig()
    res = TileResult(tile=tile.tile, split=tile.split, n_candidates=len(candidates))
    if not candidates:
        return res

    xyz = tile.xyz
    sem = tile.sem_class
    inst = tile.instance_id
    # total point count per (vehicle) instance, for coverage
    inst_groups, inst_sem, _ = _instance_metadata(xyz, sem, inst, fit_obbs=False)

    index = _GridIndex(xyz, cell_size=5.0)
    inst_obb_cache: dict[int, np.ndarray | None] = {}

    def _instance_obb(dom_id: int) -> np.ndarray | None:
        if dom_id not in inst_obb_cache:
            inst_xy = xyz[inst_groups[dom_id], :2]
            try:
                inst_obb_cache[dom_id] = oriented_bounding_box_2d(inst_xy).vertices
            except ValueError:
                inst_obb_cache[dom_id] = None
        return inst_obb_cache[dom_id]

    for cand in candidates:
        in_idx = _sorted_obb_indices(xyz, cand.ls, index)
        if len(in_idx) == 0:
            res.n_non_vehicle += 1
            continue
        inst_in = inst[in_idx]
        veh = inst_in > 0
        if not veh.any():
            res.n_non_vehicle += 1
            continue
        ids, counts = np.unique(inst_in[veh], return_counts=True)
        k = int(np.argmax(counts))
        dom_id = int(ids[k])
        n_dom = int(counts[k])
        n_veh = int(veh.sum())
        purity = n_dom / max(n_veh, 1)
        coverage = n_dom / max(len(inst_groups[dom_id]), 1)

        if not (purity >= cfg.purity_min and coverage >= cfg.coverage_min
                and n_dom >= cfg.min_points):
            res.n_ignored += 1
            continue

        sem_cls = inst_sem.get(dom_id, 0)
        if sem_cls not in VEHICLE_CLASSES:
            res.n_ignored += 1
            continue

        cand_corners = cand.obb_corners()
        inst_corners = _instance_obb(dom_id)
        iou = polygon_iou(cand_corners, inst_corners) if inst_corners is not None else 0.0
        if iou < cfg.iou_min:
            res.n_ignored += 1
            continue

        # extract the CANDIDATE-OBB point cloud (what inference will see)
        obb_xyz = xyz[in_idx]
        feat_src = obb_xyz[:, 2] if cfg.feature == "height" else tile.intensity[in_idx]
        xyz_n, feat_n = _normalise_object(obb_xyz, feat_src)
        res.objects.append(ExtractedObject(
            xyz_norm=xyz_n, feat_norm=feat_n,
            label=LABEL_OF[sem_cls], sem_class=sem_cls,
            instance_id=dom_id, purity=round(purity, 4), coverage=round(coverage, 4),
            obb_iou=round(iou, 4),
            n_points=len(in_idx), source=cand.source,
            obb_corners=cand_corners,
        ))

    return res
