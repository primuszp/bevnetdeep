"""Unified vehicle detection pipeline.

Two complementary detection methods are supported:

**Hole detection** (``"hole"``)
    Vehicles park on or drive over the ground, occluding the LiDAR returns
    beneath them.  A ground raster built from classified ground points will
    have NaN "shadow" cells where a vehicle body blocks the sensor.  These
    holes are detected, filtered by geometry, and classified.

**Blob detection** (``"blob"``)
    Points at vehicle height (typically 0.3–3.5 m above ground) are
    rasterised into an occupancy grid.  Each connected blob of occupied
    cells is a candidate vehicle.  Geometry and density features drive
    filtering and classification.

Both methods produce :class:`DetectedVehicle` objects with a common
interface.  When both are active the results are merged and deduplicated
with Non-Maximum Suppression (NMS) based on 2-D polygon IoU.

Typical usage::

    from bordereye import detect_vehicles, VehicleDetectionConfig

    vehicles = detect_vehicles(
        ground_points,
        all_points,
        config=VehicleDetectionConfig(use_lshape=True),
    )
    for v in vehicles:
        print(v.vehicle_class, v.length, v.width)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._nms import greedy_rotated_nms
from .candidates import GridCandidate, extract_raster_candidates
from .classify import VehicleClass, classify_vehicle
from .features import CandidateFeatures
from .geometry import oriented_bounding_box_2d
from .holes import RasterHole, find_raster_holes
from .lshape import fit_lshape
from .raster import RasterGrid
from .terrain import build_dtm, filter_by_height_above_ground

# ── Output type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectedVehicle:
    """A single detected vehicle, regardless of which method found it.

    Attributes:
        id: Sequential 1-based identifier within a detection run.
        source: ``"hole"`` = detected as gap in ground raster;
                ``"blob"`` = detected as occupied cluster above ground.
        vertices_2d: (4, 2) rectangle corners in world XY (metres).
        center_xy: World-space centroid of the 2-D footprint.
        length: Long side of the fitted rectangle (metres).
        width: Short side of the fitted rectangle (metres).
        height: Vertical extent (metres); ``nan`` for hole-based detections.
        min_z: Bottom elevation (metres); ``nan`` for holes.
        max_z: Top elevation (metres); ``nan`` for holes.
        rotation_deg: Long-axis angle, CCW from +X (degrees).
        vehicle_class: Best matching class name, or ``None`` if unclassified.
        class_score: Soft match score in [0, 1] for the best class.
        features: Full :class:`~bordereye.CandidateFeatures` descriptor.
        point_count: Number of LiDAR points in the candidate.
        fit_method: ``"lshape"`` or ``"obb"`` — which geometry fitter was used.
    """

    id: int
    source: Literal["hole", "blob"]

    # 2-D footprint
    vertices_2d: np.ndarray          # (4, 2) world XY corners
    center_xy: tuple[float, float]
    length: float
    width: float
    rotation_deg: float

    # 3-D (blob only; nan for holes)
    height: float
    min_z: float
    max_z: float

    # Classification
    vehicle_class: str | None
    class_score: float

    # Diagnostics
    features: CandidateFeatures
    point_count: int
    fit_method: Literal["lshape", "obb"]

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def footprint_area(self) -> float:
        return self.length * self.width

    @property
    def bounds_2d(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box of the 2-D footprint (min_x,min_y,max_x,max_y)."""
        v = self.vertices_2d
        return (float(v[:, 0].min()), float(v[:, 1].min()),
                float(v[:, 0].max()), float(v[:, 1].max()))

    @property
    def has_height(self) -> bool:
        return not math.isnan(self.height)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VehicleDetectionConfig:
    """Parameters for :func:`detect_vehicles`.

    All length values are in metres; all angles in degrees.
    """

    # ── Method switches ───────────────────────────────────────────────────────
    use_hole_detection: bool = True
    """Run the ground-shadow (hole) detection path."""
    use_blob_detection: bool = True
    """Run the above-ground blob detection path."""

    # ── DTM ───────────────────────────────────────────────────────────────────
    dtm_cell_size: float = 0.5
    """Cell size for the Digital Terrain Model raster (metres)."""
    dtm_stat: str = "min"
    """DTM statistic: ``"min"`` is the most conservative ground estimate."""

    # ── Hole detection ────────────────────────────────────────────────────────
    hole_cell_size: float = 0.25
    """Cell size for the ground raster used for hole detection (metres)."""
    hole_min_cells: int = 15
    """Minimum hole area in cells (smaller holes are ignored)."""
    hole_snap_edge: bool = True
    """Expand hole hulls to cell corners before convex-hull fitting."""

    # ── Blob detection ────────────────────────────────────────────────────────
    blob_min_height: float = 0.3
    """Minimum height above ground to include in the blob raster (metres)."""
    blob_max_height: float = 3.5
    """Maximum height above ground to include in the blob raster (metres)."""
    blob_cell_size: float = 0.20
    """Cell size for the above-ground occupancy raster (metres)."""
    blob_min_cells: int = 8
    """Minimum blob area in cells (smaller blobs are ignored)."""
    blob_diagonal: bool = False
    """Use 8-neighbour connectivity for blob connected components."""

    # ── Geometry fitting ──────────────────────────────────────────────────────
    use_lshape: bool = True
    """Use L-shape fitting instead of OBB (better for partial observations)."""
    lshape_criterion: Literal["area", "closeness", "variance"] = "variance"
    """L-shape optimisation criterion (see :func:`~bordereye.fit_lshape`)."""
    lshape_angle_resolution_deg: float = 1.0
    """Angular search step for L-shape fitting (degrees)."""

    # ── Vehicle geometry limits ───────────────────────────────────────────────
    min_length: float = 1.2
    max_length: float = 16.5
    min_width: float = 0.4
    max_width: float = 4.5
    min_height: float = 0.0       # 0 = no height lower bound (for holes)
    max_height: float = 4.5
    min_points: int = 5
    min_rectangularity: float = 0.35
    max_elongatedness: float = 10.0

    # ── NMS ───────────────────────────────────────────────────────────────────
    nms_iou_threshold: float = 0.35
    """IoU threshold for Non-Maximum Suppression (0 = no NMS, 1 = keep all)."""

    # ── Classification ────────────────────────────────────────────────────────
    vehicle_classes: list[VehicleClass] | None = None
    """Override the default vehicle class registry."""


# ── Main entry point ──────────────────────────────────────────────────────────

def detect_vehicles(
    ground_points: np.ndarray,
    all_points: np.ndarray | None = None,
    *,
    config: VehicleDetectionConfig | None = None,
) -> list[DetectedVehicle]:
    """Detect vehicles using ground-shadow holes and/or above-ground blobs.

    Args:
        ground_points: (N, 3+) XYZ array of ground-classified points.
            Used to build the DTM and the ground-hole raster.
        all_points: (M, 3+) full point cloud (ground + off-ground).
            Required when ``config.use_blob_detection`` is ``True``.
        config: Detection parameters.  ``None`` uses :class:`VehicleDetectionConfig`
            defaults which are tuned for typical vehicle sizes.

    Returns:
        Sorted list of :class:`DetectedVehicle` objects (highest class_score
        first), deduplicated by NMS.  IDs are 1-based and sequential.
    """
    cfg = config or VehicleDetectionConfig()
    detections: list[DetectedVehicle] = []

    # Build DTM (shared by both paths)
    dtm = build_dtm(ground_points, cell_size=cfg.dtm_cell_size, stat=cfg.dtm_stat)

    # ── Path 1: Ground-shadow hole detection ──────────────────────────────────
    if cfg.use_hole_detection:
        ground_raster = RasterGrid.from_point_cloud(
            ground_points, cell_size=cfg.hole_cell_size
        )
        holes = find_raster_holes(
            ground_raster,
            min_cells=cfg.hole_min_cells,
            snap_edge=cfg.hole_snap_edge,
        )
        for hole in holes:
            v = _hole_to_vehicle(hole, cfg)
            if v is not None:
                detections.append(v)

    # ── Path 2: Above-ground blob detection ───────────────────────────────────
    if cfg.use_blob_detection:
        pts = all_points if all_points is not None else ground_points
        above, _ = filter_by_height_above_ground(
            pts, dtm,
            min_height=cfg.blob_min_height,
            max_height=cfg.blob_max_height,
        )
        if len(above) >= cfg.min_points:
            blob_raster = RasterGrid.from_point_cloud(above, cell_size=cfg.blob_cell_size)
            candidates = extract_raster_candidates(
                blob_raster, diagonal=cfg.blob_diagonal, min_cells=cfg.blob_min_cells
            )
            for cand in candidates:
                v = _blob_to_vehicle(cand, blob_raster, above, cfg)
                if v is not None:
                    detections.append(v)

    # ── NMS + final sort ──────────────────────────────────────────────────────
    if len(detections) > 1 and cfg.nms_iou_threshold < 1.0:
        detections = _nms(detections, cfg.nms_iou_threshold)

    detections.sort(key=lambda v: v.class_score, reverse=True)

    # Re-assign sequential IDs
    return [_with_id(v, i + 1) for i, v in enumerate(detections)]


# ── Hole → DetectedVehicle ────────────────────────────────────────────────────

def _hole_to_vehicle(
    hole: RasterHole,
    cfg: VehicleDetectionConfig,
) -> DetectedVehicle | None:
    """Convert a RasterHole to a DetectedVehicle, or None if rejected."""
    world_pts = hole.hull_world
    if len(world_pts) < 2:
        return None

    # Fit rectangle to the hole's convex hull outline
    vertices, cx, cy, length, width, rotation_deg, fit_method = _fit_rectangle(
        world_pts, cfg, source="hole"
    )

    if not _passes_geometry(length, width, float("nan"), cfg):
        return None

    # Features (height = nan for holes, no 3-D info)
    feat = _geometry_to_features(
        length, width, float("nan"),
        hull_area=hole.hull_area,
        cell_count=hole.area_cells,
        point_count=0,
    )
    if not _passes_shape(feat, cfg):
        return None

    cls = classify_vehicle(feat, classes=cfg.vehicle_classes)

    return DetectedVehicle(
        id=0,
        source="hole",
        vertices_2d=vertices,
        center_xy=(cx, cy),
        length=length,
        width=width,
        rotation_deg=rotation_deg,
        height=float("nan"),
        min_z=float("nan"),
        max_z=float("nan"),
        vehicle_class=cls.name if cls else None,
        class_score=cls.score(feat) if cls else 0.0,
        features=feat,
        point_count=0,
        fit_method=fit_method,
    )


# ── Blob candidate → DetectedVehicle ─────────────────────────────────────────

def _blob_to_vehicle(
    candidate: GridCandidate,
    raster: RasterGrid,
    above_pts: np.ndarray,
    cfg: VehicleDetectionConfig,
) -> DetectedVehicle | None:
    """Convert a raster GridCandidate to a DetectedVehicle, or None if rejected."""
    # --- Actual points inside this candidate's bounding box ---
    wb = candidate.world_bounds          # (min_x, min_y, max_x, max_y)
    mask = (
        (above_pts[:, 0] >= wb[0]) & (above_pts[:, 0] <= wb[2]) &
        (above_pts[:, 1] >= wb[1]) & (above_pts[:, 1] <= wb[3])
    )
    local_pts = above_pts[mask]
    if len(local_pts) < cfg.min_points:
        return None

    min_z = float(local_pts[:, 2].min())
    max_z = float(local_pts[:, 2].max())
    height = max_z - min_z

    # Fit rectangle — use actual XY points for best L-shape result
    fit_pts = local_pts[:, :2]
    vertices, cx, cy, length, width, rotation_deg, fit_method = _fit_rectangle(
        fit_pts, cfg, source="blob"
    )

    if not _passes_geometry(length, width, height, cfg):
        return None

    hull_area: float | None = None
    if candidate.obb_2d is not None:
        hull_area = candidate.obb_2d.area

    feat = _geometry_to_features(
        length, width, height,
        hull_area=hull_area,
        cell_count=candidate.cell_count,
        point_count=candidate.point_count,
    )
    if not _passes_shape(feat, cfg):
        return None

    cls = classify_vehicle(feat, classes=cfg.vehicle_classes)

    return DetectedVehicle(
        id=0,
        source="blob",
        vertices_2d=vertices,
        center_xy=(cx, cy),
        length=length,
        width=width,
        rotation_deg=rotation_deg,
        height=height,
        min_z=min_z,
        max_z=max_z,
        vehicle_class=cls.name if cls else None,
        class_score=cls.score(feat) if cls else 0.0,
        features=feat,
        point_count=int(mask.sum()),
        fit_method=fit_method,
    )


# ── Geometry fitting helpers ──────────────────────────────────────────────────

def _fit_rectangle(
    pts_2d: np.ndarray,
    cfg: VehicleDetectionConfig,
    source: Literal["hole", "blob"],
) -> tuple[np.ndarray, float, float, float, float, float, Literal["lshape", "obb"]]:
    """Return (vertices, cx, cy, length, width, rotation_deg, method)."""
    unique = np.unique(pts_2d[:, :2], axis=0)

    if cfg.use_lshape and len(unique) >= 3:
        try:
            r = fit_lshape(
                unique,
                criterion=cfg.lshape_criterion,
                angle_resolution_deg=cfg.lshape_angle_resolution_deg,
            )
            return (
                r.vertices, r.center[0], r.center[1],
                r.length, r.width, r.rotation_deg, "lshape"
            )
        except ValueError:
            pass  # degenerate input; fall through to OBB

    # OBB fallback
    if len(unique) < 2:
        # Degenerate: single point or collinear → axis-aligned box
        mn = unique.min(axis=0)
        mx = unique.max(axis=0)
        verts = np.array([[mn[0],mn[1]],[mx[0],mn[1]],[mx[0],mx[1]],[mn[0],mx[1]]])
        cx, cy = float(verts[:,0].mean()), float(verts[:,1].mean())
        dims = mx - mn
        box_length, box_width = float(max(dims)), float(min(dims))
        return verts, cx, cy, box_length, box_width, 0.0, "obb"

    obb = oriented_bounding_box_2d(unique)
    return (
        obb.vertices, obb.center[0], obb.center[1],
        obb.length, obb.width, obb.rotation_deg, "obb"
    )


def _geometry_to_features(
    length: float,
    width: float,
    height: float,
    *,
    hull_area: float | None,
    cell_count: int,
    point_count: int,
) -> CandidateFeatures:
    L = max(length, 1e-6)
    W = max(width, 1e-6)
    H = height  # may be nan
    footprint = L * W
    vol = footprint * (H if not math.isnan(H) else 1.0)

    rectangularity = min(1.0, hull_area / footprint) if hull_area and hull_area > 0 else 1.0

    elongatedness = L / W
    aspect_hw = (H / W) if not math.isnan(H) else float("nan")
    compactness = min(1.0, cell_count / max(footprint, 1e-3))
    point_density = point_count / max(vol, 1e-3) if not math.isnan(H) else float("nan")

    return CandidateFeatures(
        length=L,
        width=W,
        height=H,
        elongatedness=elongatedness,
        rectangularity=rectangularity,
        compactness=compactness,
        point_density=point_density,
        aspect_hw=aspect_hw,
        footprint_area=footprint,
        volume=vol,
    )


# ── Geometry filters ──────────────────────────────────────────────────────────

def _passes_geometry(length: float, width: float, height: float, cfg: VehicleDetectionConfig) -> bool:
    if not (cfg.min_length <= length <= cfg.max_length):
        return False
    if not (cfg.min_width <= width <= cfg.max_width):
        return False
    return math.isnan(height) or cfg.min_height <= height <= cfg.max_height


def _passes_shape(feat: CandidateFeatures, cfg: VehicleDetectionConfig) -> bool:
    if feat.rectangularity < cfg.min_rectangularity:
        return False
    return feat.elongatedness <= cfg.max_elongatedness


# ── Non-Maximum Suppression ───────────────────────────────────────────────────

def _nms(
    vehicles: list[DetectedVehicle],
    iou_threshold: float,
) -> list[DetectedVehicle]:
    """Greedy NMS using 2-D polygon IoU (Sutherland-Hodgman intersection).

    Vehicles are processed in descending class_score order.  A candidate is
    suppressed if its IoU with any already-accepted detection exceeds the
    threshold.  When both detections are unclassified (score == 0) the blob
    source is preferred over hole source.
    """
    if not vehicles:
        return []

    return greedy_rotated_nms(
        vehicles,
        iou_threshold,
        corners=lambda v: v.vertices_2d,
        circle=lambda v: (
            v.center_xy[0],
            v.center_xy[1],
            0.5 * float(np.hypot(v.length, v.width)),
        ),
        # Higher score first; blob preferred over hole on tie.
        key=lambda v: (v.class_score, v.source == "blob"),
    )


# ── Utility ───────────────────────────────────────────────────────────────────

def _with_id(v: DetectedVehicle, new_id: int) -> DetectedVehicle:
    """Return a copy of *v* with a new id (frozen dataclass workaround)."""
    return DetectedVehicle(
        id=new_id,
        source=v.source,
        vertices_2d=v.vertices_2d,
        center_xy=v.center_xy,
        length=v.length,
        width=v.width,
        rotation_deg=v.rotation_deg,
        height=v.height,
        min_z=v.min_z,
        max_z=v.max_z,
        vehicle_class=v.vehicle_class,
        class_score=v.class_score,
        features=v.features,
        point_count=v.point_count,
        fit_method=v.fit_method,
    )
