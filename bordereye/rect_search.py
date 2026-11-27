"""Rectangular shape-prior vehicle search on a BEV occupancy grid.

The hole / blob pipeline asks *"which cells are connected?"* and then fits a
single oriented box per connected component.  On aerial LiDAR that fails in one
dominant way: vehicles parked side-by-side (or touching vegetation / kerbs /
buildings) merge into one big component, the single fitted OBB is over-sized,
and the vehicle-size filter throws it away. On DALES tile ``5080_54400`` this
accounts for 22.9 % of the holes and 54.1 % of the blobs that are lost.

This module asks a different question: *"which points can be explained by a
vehicle-sized rotated rectangle?"*  Inside every BEV component (including the
big merged ones) it iteratively grows / carves out vehicle-sized rotated
rectangles, scoring each candidate with a shape prior:

    S =  w_fill        * rectangularity        (occupied cells / rect cells)
       + w_size        * size_prior            (Gaussian around car/truck dims)
       - w_outside     * outer_ring_occupancy  (occupied cells just outside)

A rectangle that is well filled, the right size, and *isolated* (little point
mass hugging its border) scores high; a slab of a merged blob does not.  The
best rectangle is recorded, its cells are claimed, and the search repeats on the
remaining cells of the component — so one merged blob yields several vehicles.
A final greedy rotated-box NMS removes overlaps across components.

Pure numpy + :mod:`bordereye` (RasterGrid, connected components, terrain).  No
shapely / sklearn dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ._nms import greedy_rotated_nms
from .geometry import (
    OrientedBoundingBox2D,
    fit_hull_obb_2d,
    oriented_bounding_box_2d,
    polygon_iou,
    rect_corners_2d,
    to_box_frame_2d,
)
from .raster import RasterGrid
from .raster_ops import fill_nodata_nn
from .spatial import SpatialIndex
from .terrain import height_above_ground


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class VehicleTemplate:
    """A size prior (one Gaussian mode in the size penalty)."""
    name: str
    length: float
    width: float


# Default size priors: a small-car mode and a truck/bus mode.  The size_prior is
# the *best* (max) over all templates, so a candidate only needs to match one.
DEFAULT_TEMPLATES: tuple[VehicleTemplate, ...] = (
    VehicleTemplate("car", 4.5, 1.9),
    VehicleTemplate("van", 5.5, 2.1),
    VehicleTemplate("truck", 9.0, 2.6),
)


@dataclass(frozen=True)
class RectSearchConfig:
    # ── BEV raster ────────────────────────────────────────────────────────────
    cell_size: float = 0.25
    hag_min: float = 0.25          # candidate point band, height above ground (m)
    hag_max: float = 4.0
    dtm_cell_size: float = 0.5
    diagonal: bool = True          # 8-neighbour connectivity for components
    min_cells: int = 5             # ignore components / leftovers below this
    max_component_cells: int = 2500  # skip components bigger than this (buildings /
                                     # dense vegetation — never an isolated vehicle)
    max_score_cells: int = 600     # subsample cells used for seed scoring on big
                                   # components (claiming still uses every cell)

    # ── Vehicle size gate (hard limits, metres) ──────────────────────────────
    min_len: float = 1.6
    max_len: float = 16.0
    min_wid: float = 0.9
    max_wid: float = 3.2

    # ── Search resolution ─────────────────────────────────────────────────────
    angle_step_deg: float = 6.0    # orientation sweep step (0..180)
    max_seeds: int = 48            # seed centres tried per growing iteration
    max_rectangles_per_comp: int = 12
    refine: bool = True            # grow rect to the percentile extent of its cells
    refine_pct: float = 98.0       # robust central coverage used when refining L/W

    # ── Shape-prior score weights ─────────────────────────────────────────────
    w_fill: float = 1.0            # rectangularity (occupied / expected cells)
    w_size: float = 0.6            # size prior (car/truck Gaussian)
    w_outside: float = 1.2         # penalty for occupied cells hugging the border
    outer_margin: float = 0.6      # outer-ring width around the rectangle (m)
    size_sigma_len: float = 1.6    # Gaussian sigma for length prior (m)
    size_sigma_wid: float = 0.6    # Gaussian sigma for width prior (m)

    # ── Acceptance / NMS ──────────────────────────────────────────────────────
    min_fill: float = 0.45         # reject rectangles emptier than this
    score_threshold: float = 0.35
    nms_iou: float = 0.2
    physical_nms_iou: float = 0.08  # even modest vehicle footprint overlap is suspicious
    raw_point_refine: bool = True   # final OBB geometry from raw points, not BEV cells
    raw_refine_margin: float = 0.35 # collect raw points around the search rectangle
    raw_refine_pct: float = 99.0    # robust point extent; independent of cell size
    split_rectangularity: float = 0.76
    split_min_points: int = 20
    split_model_penalty: float = 0.12
    split_min_fraction: float = 0.25

    # ── MDL merge (fewer, better-explaining boxes) ────────────────────────────
    merge_enabled: bool = True     # merge fragments that one vehicle box explains
    merge_min_coverage: float = 0.75  # the single box must cover this much of the
                                      # two fragments' cells to replace them
    merge_min_fill: float = 0.40   # ...and stay this tightly filled (no loose box
                                   # spanning a gap between two real vehicles).
                                   # Tuned on 5100_54440 PLY vs GT instances: vs the
                                   # old 0.85/0.50 this cut truck over-segmentation
                                   # (recall 80->89%, prec 41->59%, pred/GT 1.36->1.19)
                                   # with cars unchanged at 98.9%/98.9%.

    # Point-explanation score.  Disabled by default to preserve the original
    # baseline; enable with ``dataclasses.replace(cfg, point_explanation=True)``
    # for density-aware fitting.
    point_explanation: bool = False
    w_point_mass: float = 0.55     # reward fraction of component points explained
    w_empty: float = 0.35          # penalise empty expected cells inside the box
    count_clip: float = 6.0        # cap per-cell density so one dense cell cannot dominate
    model_cost: float = 0.10       # MDL cost paid by every accepted rectangle
    min_point_mass: float = 0.08   # reject boxes explaining too little of a component

    templates: tuple[VehicleTemplate, ...] = DEFAULT_TEMPLATES

    # template (L, W) pairs used to *seed* the search (before refinement)
    def seed_sizes(self) -> list[tuple[float, float]]:
        return [(t.length, t.width) for t in self.templates]


@dataclass(frozen=True)
class RectDetection:
    """One vehicle-sized rotated rectangle found by the shape-prior search."""
    center: tuple[float, float]
    length: float
    width: float
    rotation_deg: float
    score: float
    fill: float                    # rectangularity in [0, 1]
    outer_occupancy: float         # occupied fraction of the outer ring
    size_prior: float              # best template match in [0, 1]
    n_cells: int                   # occupied BEV cells inside
    n_points: int                  # LiDAR points inside (cell counts summed)
    height: float                  # max cell height of inside cells (m), nan if unknown
    height_mean: float             # mean inside cell height (m), nan if unknown
    height_std: float              # std of inside cell heights (m) — roof flatness;
                                   # low for vehicles, high for vegetation
    vertices: np.ndarray = field(repr=False)  # (4, 2) world-XY corners, CCW

    @property
    def area(self) -> float:
        return self.length * self.width


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (pure numpy)
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class _CellMass:
    """Clipped per-cell point weights and their total, shared across seeds."""

    weights: np.ndarray
    total: float


def _cell_mass(cell_cnt: np.ndarray, cfg: RectSearchConfig) -> _CellMass | None:
    """Point-explanation weights for one cell set, or None when disabled."""
    if not cfg.point_explanation:
        return None
    weights = np.minimum(np.asarray(cell_cnt, dtype=float), cfg.count_clip)
    return _CellMass(weights=weights, total=max(float(weights.sum()), 1.0))


def _size_prior(length: float, width: float, cfg: RectSearchConfig) -> float:
    """Best Gaussian match over all templates, in [0, 1]."""
    best = 0.0
    for t in cfg.templates:
        d = ((length - t.length) ** 2) / (cfg.size_sigma_len ** 2) \
            + ((width - t.width) ** 2) / (cfg.size_sigma_wid ** 2)
        best = max(best, float(np.exp(-0.5 * d)))
    return best


def _score_rect(
    cx: float, cy: float, length: float, width: float, theta: float,
    cell_xy: np.ndarray, cell_cnt: np.ndarray, cfg: RectSearchConfig,
) -> tuple[float, float, float, float, np.ndarray]:
    """Score one rectangle against the component's occupied cells.

    Returns (score, fill, outer_occupancy, size_prior, inside_mask).
    """
    cell_area = cfg.cell_size ** 2
    xo, yo = to_box_frame_2d(cell_xy, cx, cy, theta)
    hl, hw = length / 2.0, width / 2.0
    m = cfg.outer_margin

    inside = (np.abs(xo) <= hl) & (np.abs(yo) <= hw)
    outer = (~inside) & (np.abs(xo) <= hl + m) & (np.abs(yo) <= hw + m)

    n_in = int(inside.sum())
    expected_cells = max(length * width / cell_area, 1.0)
    fill = min(1.0, n_in / expected_cells)

    outer_area = (length + 2 * m) * (width + 2 * m) - length * width
    outer_expected = max(outer_area / cell_area, 1.0)
    outer_occ = min(1.0, int(outer.sum()) / outer_expected)

    sp = _size_prior(length, width, cfg)

    score = cfg.w_fill * fill + cfg.w_size * sp - cfg.w_outside * outer_occ
    if cfg.point_explanation:
        weights = np.minimum(np.asarray(cell_cnt, dtype=float), cfg.count_clip)
        total_mass = max(float(weights.sum()), 1.0)
        point_mass = float(weights[inside].sum()) / total_mass
        outer_mass = float(weights[outer].sum()) / total_mass
        empty = max(0.0, 1.0 - fill)
        score += (cfg.w_point_mass * point_mass - cfg.w_empty * empty
                  - cfg.w_outside * outer_mass - cfg.model_cost)
    return float(score), float(fill), float(outer_occ), float(sp), inside


def _size_ok(length: float, width: float, cfg: RectSearchConfig) -> bool:
    return (cfg.min_len <= length <= cfg.max_len) and (cfg.min_wid <= width <= cfg.max_wid)


# ─────────────────────────────────────────────────────────────────────────────
# Per-component iterative rectangle growing
# ─────────────────────────────────────────────────────────────────────────────
def _seed_frame(
    sx: float, sy: float, angles: np.ndarray, cell_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """|x|, |y| of every cell in every candidate rectangle frame — (K, A) each.

    Depends only on the seed centre and the angle sweep, never on the template
    size, so one seed's projection is shared by all templates.
    """
    dx = cell_xy[:, 0] - sx                      # (K,)
    dy = cell_xy[:, 1] - sy
    ca = np.cos(-angles)                          # (A,)
    sa = np.sin(-angles)
    xo = np.outer(dx, ca) - np.outer(dy, sa)     # (K, A)
    yo = np.outer(dx, sa) + np.outer(dy, ca)
    return np.abs(xo, out=xo), np.abs(yo, out=yo)


def _score_seed_angles(
    axo: np.ndarray, ayo: np.ndarray, length: float, width: float,
    angles: np.ndarray, cfg: RectSearchConfig,
    size_prior: float, mass: _CellMass | None,
) -> tuple[float, float, float, float, float, np.ndarray]:
    """Score a fixed-centre, fixed-size rectangle over *all* angles at once.

    Vectorised across angles (the inner hot loop) — takes the shared per-seed
    projection from :func:`_seed_frame` and returns the best angle's
    (score, theta, fill, outer_occupancy, size_prior, inside_mask).

    *size_prior* and *mass* depend only on the template and the component, so
    they are computed once by the caller rather than per seed.
    """
    cell_area = cfg.cell_size ** 2
    hl, hw, m = length / 2.0, width / 2.0, cfg.outer_margin

    inside = (axo <= hl) & (ayo <= hw)           # (K, A)
    outer = (~inside) & (axo <= hl + m) & (ayo <= hw + m)

    expected = max(length * width / cell_area, 1.0)
    fill = np.minimum(1.0, inside.sum(0) / expected)            # (A,)
    outer_area = (length + 2 * m) * (width + 2 * m) - length * width
    outer_occ = np.minimum(1.0, outer.sum(0) / max(outer_area / cell_area, 1.0))
    sp = size_prior

    score = cfg.w_fill * fill + cfg.w_size * sp - cfg.w_outside * outer_occ
    if mass is not None:
        # weights @ mask is a matrix-vector product: it sums per angle without
        # materialising the (K, A) float product array.
        point_mass = (mass.weights @ inside) / mass.total
        outer_mass = (mass.weights @ outer) / mass.total
        score += (cfg.w_point_mass * point_mass - cfg.w_empty * (1.0 - fill)
                  - cfg.w_outside * outer_mass - cfg.model_cost)
    k = int(np.argmax(score))
    return (float(score[k]), float(angles[k]), float(fill[k]),
            float(outer_occ[k]), float(sp), inside[:, k])


def _best_rect_for_seed(
    sx: float, sy: float, cell_xy: np.ndarray,
    angles: np.ndarray, cfg: RectSearchConfig,
    templates: tuple[tuple[float, float, float], ...], mass: _CellMass | None,
) -> tuple | None:
    """Best (score, cx, cy, L, W, theta, fill, outer, sp, inside) over templates."""
    best = None
    axo, ayo = _seed_frame(sx, sy, angles, cell_xy)
    for L0, W0, size_prior in templates:
        score, theta, fill, outer_occ, sp, inside = _score_seed_angles(
            axo, ayo, L0, W0, angles, cfg, size_prior, mass
        )
        if best is None or score > best[0]:
            best = (score, sx, sy, L0, W0, theta, fill, outer_occ, sp, inside)
    return best


def _fit_best(
    seed_xy: np.ndarray, work_xy: np.ndarray,
    full_xy: np.ndarray, angles: np.ndarray, cfg: RectSearchConfig,
    work_cnt: np.ndarray | None = None, full_cnt: np.ndarray | None = None,
) -> tuple | None:
    """Best vehicle box over the seeds, grown and re-scored on *full_xy*.

    Returns (score, cx, cy, L, W, theta, fill, outer, sp, inside_full_mask).
    """
    best = None
    if work_cnt is None:
        work_cnt = np.ones(len(work_xy), dtype=float)
    if full_cnt is None:
        full_cnt = np.ones(len(full_xy), dtype=float)
    # Template size priors and the clipped cell-count weights are the same for
    # every seed, so build them once instead of per seed x template.
    templates = tuple((L, W, _size_prior(L, W, cfg)) for L, W in cfg.seed_sizes())
    work_mass = _cell_mass(work_cnt, cfg)
    for sx, sy in seed_xy:
        cand = _best_rect_for_seed(
            float(sx), float(sy), work_xy, angles, cfg, templates, work_mass)
        if cand is not None and (best is None or cand[0] > best[0]):
            best = cand
    if best is None:
        return None
    _, cx, cy, L, W, theta, *_ = best
    inside_full = _score_rect(cx, cy, L, W, theta, full_xy, full_cnt, cfg)[4]
    if cfg.refine and inside_full.sum() >= 2:
        cx, cy, L, W, theta = _grow(cx, cy, theta, full_xy[inside_full], cfg)
    score, fill, outer_occ, sp, inside_full = _score_rect(cx, cy, L, W, theta, full_xy, full_cnt, cfg)
    return score, cx, cy, L, W, theta, fill, outer_occ, sp, inside_full


def _make_detection(
    params: tuple, idx: np.ndarray, cell_cnt: np.ndarray, cell_h: np.ndarray | None,
) -> RectDetection:
    score, cx, cy, L, W, theta, fill, outer_occ, sp = params
    n_pts = int(cell_cnt[idx].sum())
    if cell_h is not None and len(idx):
        h_in = cell_h[idx]
        h_max = float(np.nanmax(h_in))
        h_mean = float(np.nanmean(h_in))
        h_std = float(np.nanstd(h_in))
    else:
        h_max = h_mean = h_std = float("nan")
    return RectDetection(
        center=(float(cx), float(cy)), length=float(L), width=float(W),
        rotation_deg=float(np.degrees(theta)), score=float(score),
        fill=float(fill), outer_occupancy=float(outer_occ), size_prior=float(sp),
        n_cells=len(idx), n_points=n_pts,
        height=h_max, height_mean=h_mean, height_std=h_std,
        vertices=rect_corners_2d(cx, cy, L, W, theta),
    )


def _merge_pass(
    cands: list[dict], cell_xy: np.ndarray, angles: np.ndarray, cfg: RectSearchConfig,
) -> list[dict]:
    """Collapse fragments: replace two boxes with one when a single vehicle box
    explains (covers tightly) the cells of both — the MDL 'fewer boxes' step."""
    changed = True
    while changed and len(cands) > 1:
        changed = False
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) > cfg.max_len:
                    continue
                union = np.union1d(a["idx"], b["idx"])
                if len(union) < cfg.min_cells:
                    continue
                cxy = cell_xy[union].mean(axis=0)
                res = _fit_best(cxy[None, :], cell_xy[union], cell_xy, angles, cfg)
                if res is None:
                    continue
                score, cx, cy, L, W, theta, fill, outer_occ, sp, inside_full = res
                merged_idx = np.flatnonzero(inside_full)
                coverage = float(np.isin(union, merged_idx).mean())   # of both fragments
                if (_size_ok(L, W, cfg) and fill >= cfg.merge_min_fill
                        and coverage >= cfg.merge_min_coverage):
                    merged = {"score": score, "cx": cx, "cy": cy, "L": L, "W": W,
                              "theta": theta, "fill": fill, "outer": outer_occ,
                              "sp": sp, "idx": merged_idx}
                    cands = [c for k, c in enumerate(cands) if k not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return cands


def search_component(
    cell_xy: np.ndarray,
    cell_cnt: np.ndarray,
    cell_h: np.ndarray | None,
    cfg: RectSearchConfig,
) -> list[RectDetection]:
    """Carve vehicle-sized rotated rectangles out of one component, then merge
    fragments that a single vehicle box explains better (MDL)."""
    n = len(cell_xy)
    if n < cfg.min_cells:
        return []

    angles = np.deg2rad(np.arange(0.0, 180.0, cfg.angle_step_deg))
    claimed = np.zeros(n, dtype=bool)
    rng = np.random.default_rng(0)
    cands: list[dict] = []

    for _ in range(cfg.max_rectangles_per_comp):
        free = np.flatnonzero(~claimed)
        if len(free) < cfg.min_cells:
            break
        free_xy = cell_xy[free]
        free_cnt = cell_cnt[free]

        # Fast path: if the remaining cells already fit in a single vehicle box,
        # one centroid seed suffices.  Only over-sized (merged) remainders pay for
        # the multi-seed search.
        # A set that fits an L x W box has diameter <= hypot(L, W), and the
        # diameter is at least the longest axis-aligned side, so an oversized
        # bounding box rules the fast path out without a convex hull.
        span = free_xy.max(axis=0) - free_xy.min(axis=0)
        if float(span.max()) > np.hypot(cfg.max_len, cfg.max_wid):
            single = False
        else:
            try:
                obb = oriented_bounding_box_2d(free_xy)
                single = obb.length <= cfg.max_len and obb.width <= cfg.max_wid
            except ValueError:
                single = False
        # Subsample the cells used to *score* candidates on big components; the
        # fit/claim below always re-scores against the full free set.
        if len(free) > cfg.max_score_cells:
            work_sel = rng.choice(len(free), size=cfg.max_score_cells, replace=False)
            work_xy = free_xy[work_sel]
            work_cnt = free_cnt[work_sel]
        else:
            work_xy = free_xy
            work_cnt = free_cnt

        if single:
            seed_xy = free_xy.mean(axis=0)[None, :]
        elif len(free) > cfg.max_seeds:
            seed_xy = work_xy[rng.choice(len(work_xy), size=cfg.max_seeds, replace=False)]
        else:
            seed_xy = free_xy

        res = _fit_best(seed_xy, work_xy, free_xy, angles, cfg,
                        work_cnt=work_cnt, full_cnt=free_cnt)
        if res is None:
            break
        score, cx, cy, L, W, theta, fill, outer_occ, sp, inside_free = res
        if not inside_free.any():
            break
        global_inside = free[inside_free]
        claimed[global_inside] = True            # always claim, so the loop ends

        point_mass = float(cell_cnt[global_inside].sum()) / max(float(cell_cnt.sum()), 1.0)
        if (score >= cfg.score_threshold and fill >= cfg.min_fill
                and (not cfg.point_explanation or point_mass >= cfg.min_point_mass)
                and _size_ok(L, W, cfg) and int(inside_free.sum()) >= cfg.min_cells):
            cands.append({"score": score, "cx": cx, "cy": cy, "L": L, "W": W,
                          "theta": theta, "fill": fill, "outer": outer_occ,
                          "sp": sp, "idx": global_inside})

    if cfg.merge_enabled and len(cands) > 1:
        cands = _merge_pass(cands, cell_xy, angles, cfg)

    return [
        _make_detection(
            (c["score"], c["cx"], c["cy"], c["L"], c["W"], c["theta"],
             c["fill"], c["outer"], c["sp"]),
            c["idx"], cell_cnt, cell_h,
        )
        for c in cands
    ]


def _grow(
    cx: float, cy: float, theta: float, inside_xy: np.ndarray, cfg: RectSearchConfig,
) -> tuple[float, float, float, float, float]:
    """Re-centre and resize the rectangle to the captured cells (the 'growing' step)."""
    xo, yo = to_box_frame_2d(inside_xy, cx, cy, theta)
    # Use asymmetric lower/upper quantiles instead of a symmetric percentile
    # around the mean.  Aerial vehicle returns are frequently one-sided, so the
    # old symmetric estimate shrank the box toward the denser roof portion.
    tail = max(0.0, min(50.0, (100.0 - cfg.refine_pct) / 2.0))
    x0, x1 = np.percentile(xo, [tail, 100.0 - tail])
    y0, y1 = np.percentile(yo, [tail, 100.0 - tail])
    # Cell centres are samples of occupied square cells, not zero-area points.
    # A square cell's full projection on either rotated rectangle axis is
    # cell_size * (|cos(theta)| + |sin(theta)|).  Adding that projection makes
    # the OBB cover the complete occupied cell footprints at every angle.
    cell_projection = cfg.cell_size * (abs(np.cos(theta)) + abs(np.sin(theta)))
    length = float(x1 - x0) + cell_projection
    width = float(y1 - y0) + cell_projection
    mx, my = float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)
    ca, sa = np.cos(theta), np.sin(theta)
    ncx = cx + mx * ca - my * sa
    ncy = cy + mx * sa + my * ca
    if width > length:  # keep length on the long axis
        length, width = width, length
        theta += np.pi / 2.0
    return ncx, ncy, length, width, theta


# ─────────────────────────────────────────────────────────────────────────────
# Rotated-box NMS
# ─────────────────────────────────────────────────────────────────────────────
def _half_diagonal(d: RectDetection) -> float:
    return 0.5 * float(np.hypot(d.length, d.width))


def _local_box_candidates(
    index: SpatialIndex,
    detection: RectDetection,
    margin: float,
) -> np.ndarray:
    """Cheap circular prefilter for one margin-expanded rectangle."""
    radius = float(np.hypot(detection.length * 0.5 + margin, detection.width * 0.5 + margin))
    return index.radius(detection.center, radius, use_3d=False)


def nms_rectangles(dets: list[RectDetection], iou_threshold: float) -> list[RectDetection]:
    """Greedy rotated-box NMS over rectangle-search detections, best score first."""
    return greedy_rotated_nms(
        dets,
        iou_threshold,
        corners=lambda d: d.vertices,
        circle=lambda d: (d.center[0], d.center[1], _half_diagonal(d)),
        key=lambda d: d.score,
    )


def _refine_rectangles_from_raw_points(
    xyz: np.ndarray,
    detections: list[RectDetection],
    cfg: RectSearchConfig,
    *,
    index: SpatialIndex | None = None,
) -> list[RectDetection]:
    """Refine final OBB geometry from raw XY points, independent of BEV cells."""
    out: list[RectDetection] = []
    index = index or SpatialIndex(xyz)
    xy = xyz[:, :2]
    for d in detections:
        candidate_idx = _local_box_candidates(index, d, cfg.raw_refine_margin)
        candidate_xy = xy[candidate_idx]
        theta = np.deg2rad(d.rotation_deg)
        xo, yo = to_box_frame_2d(candidate_xy, d.center[0], d.center[1], theta)
        m = ((np.abs(xo) <= d.length / 2 + cfg.raw_refine_margin)
             & (np.abs(yo) <= d.width / 2 + cfg.raw_refine_margin))
        if int(m.sum()) < cfg.min_cells:
            out.append(d)
            continue
        bx, by = xo[m], yo[m]
        tail = max(0., min(50., (100. - cfg.raw_refine_pct) / 2.))
        x0, x1 = np.percentile(bx, [tail, 100-tail])
        y0, y1 = np.percentile(by, [tail, 100-tail])
        length, width = float(x1-x0), float(y1-y0)
        mx, my = float((x0+x1)/2), float((y0+y1)/2)
        ca, sa = np.cos(theta), np.sin(theta)
        cx = d.center[0] + mx*ca - my*sa
        cy = d.center[1] + mx*sa + my*ca
        if width > length:
            length, width = width, length
            theta += np.pi/2
        # Reject a refinement that absorbs a neighbouring vehicle.
        if not _size_ok(length, width, cfg):
            out.append(d)
            continue
        out.append(replace(
            d, center=(cx, cy), length=length, width=width,
            rotation_deg=float(np.degrees(theta) % 180),
            vertices=rect_corners_2d(cx, cy, length, width, theta),
            n_points=int(m.sum()),
        ))
    return out


def _shape_quality(
    points: np.ndarray,
    det: RectDetection,
    *,
    hull_area: float | None = None,
) -> float:
    """Quality of one vehicle explanation; higher is better."""
    if len(points) < 3 or det.area <= 1e-9:
        return 0.0
    if hull_area is None:
        hull_area = fit_hull_obb_2d(points[:, :2]).hull_area
    rectangularity = min(1.0, hull_area / det.area)
    xo, yo = to_box_frame_2d(points[:, :2], det.center[0], det.center[1], np.deg2rad(det.rotation_deg))
    inside = (np.abs(xo) <= det.length/2 + 1e-6) & (np.abs(yo) <= det.width/2 + 1e-6)
    coverage = float(inside.mean())
    return .65 * rectangularity + .35 * coverage


def _detection_from_points(
    points: np.ndarray,
    old: RectDetection,
    cfg: RectSearchConfig,
    *,
    max_width: float | None = None,
    obb: OrientedBoundingBox2D | None = None,
) -> RectDetection | None:
    if len(points) < cfg.split_min_points:
        return None
    if obb is None:
        obb = fit_hull_obb_2d(points[:, :2]).obb
    width_limit = cfg.max_wid if max_width is None else max_width
    if not (
        cfg.min_len <= obb.length <= cfg.max_len
        and cfg.min_wid <= obb.width <= width_limit
    ):
        return None
    h = points[:, 2]
    return replace(
        old, center=obb.center, length=obb.length, width=obb.width,
        rotation_deg=obb.rotation_deg, n_points=len(points),
        height=float(np.ptp(h)), height_mean=float(np.mean(h)), height_std=float(np.std(h)),
        vertices=obb.vertices,
    )
def _split_nonrectangular_detections(
    xyz: np.ndarray, detections: list[RectDetection], cfg: RectSearchConfig,
    owner: np.ndarray | None = None,
    *, index: SpatialIndex | None = None,
) -> list[RectDetection]:
    """Split a low hull/OBB-area-ratio blob when two OBBs explain it better."""
    out: list[RectDetection] = []
    index = index or SpatialIndex(xyz)
    xy = xyz[:, :2]
    for di, d in enumerate(detections):
        candidate_idx = _local_box_candidates(index, d, cfg.raw_refine_margin)
        theta = np.deg2rad(d.rotation_deg)
        xo, yo = to_box_frame_2d(xy[candidate_idx], d.center[0], d.center[1], theta)
        mask = ((np.abs(xo) <= d.length/2 + cfg.raw_refine_margin)
                & (np.abs(yo) <= d.width/2 + cfg.raw_refine_margin))
        if owner is not None:
            mask &= owner[candidate_idx] == di
        pts = xyz[candidate_idx[mask]]
        stats = fit_hull_obb_2d(pts[:, :2]) if len(pts) >= 3 else None
        ratio = 1.0 if stats is None else stats.rectangularity
        local_obb = None if stats is None else stats.obb
        merged = _detection_from_points(pts, d, cfg, max_width=4.0, obb=local_obb)
        merged_quality = (
            _shape_quality(pts, merged, hull_area=stats.hull_area)
            if merged is not None and stats is not None
            else 0.0
        )
        physically_wide = local_obb is not None and local_obb.width > cfg.max_wid
        if (
            ratio >= cfg.split_rectangularity and not physically_wide
        ) or len(pts) < 2 * cfg.split_min_points:
            out.append(d)
            continue
        # Test both principal directions. Touching L-shaped vehicles often have
        # no visible gap, so evaluate several balanced cuts, not only max-gap.
        best = None
        for values in (xo[mask], yo[mask]):
            order = np.argsort(values)
            sv = values[order]
            gaps = np.diff(sv)
            if not len(gaps):
                continue
            cuts = {int(np.argmax(gaps)) + 1}
            cuts.update(int(len(pts) * q) for q in (.3, .4, .5, .6, .7))
            for cut in cuts:
                if cut < cfg.split_min_points or len(pts) - cut < cfg.split_min_points:
                    continue
                pa, pb = pts[order[:cut]], pts[order[cut:]]
                fraction = min(len(pa), len(pb)) / len(pts)
                if fraction < cfg.split_min_fraction:
                    continue
                stats_a = fit_hull_obb_2d(pa[:, :2]) if len(pa) >= 3 else None
                stats_b = fit_hull_obb_2d(pb[:, :2]) if len(pb) >= 3 else None
                a = _detection_from_points(
                    pa, d, cfg, max_width=4.0, obb=None if stats_a is None else stats_a.obb)
                b = _detection_from_points(
                    pb, d, cfg, max_width=4.0, obb=None if stats_b is None else stats_b.obb)
                if a is None or b is None:
                    continue
                # Both OBBs are fully recomputed above. Weight their quality by
                # explained point mass and charge a second-model penalty.
                quality = ((len(pa) * _shape_quality(pa, a, hull_area=stats_a.hull_area)
                            + len(pb) * _shape_quality(pb, b, hull_area=stats_b.hull_area))
                           / len(pts) - cfg.split_model_penalty)
                if best is None or quality > best[0]:
                    best = (quality, a, b)
        # Compare raw quality (before model-cost penalty) so the MDL cost is not
        # charged twice: once in `quality` and again as a margin here.
        if best is not None and best[0] + cfg.split_model_penalty > merged_quality + .04:
            out.extend(best[1:])
        else:
            # Re-merge/refit: if two boxes are not demonstrably better, use the
            # single OBB recomputed from the complete blob.
            out.append(merged if merged is not None else d)
    return out


def _resolve_overlaps_by_repartition(
    xyz: np.ndarray, detections: list[RectDetection], cfg: RectSearchConfig,
    *, index: SpatialIndex | None = None,
) -> list[RectDetection]:
    """Repartition strongly overlapping vehicles and refit their orientation."""
    dets = list(detections)
    index = index or SpatialIndex(xyz)
    xy = xyz[:, :2]
    for _ in range(3):
        changed = False
        for i in range(len(dets)):
            for j in range(i+1, len(dets)):
                # Bounding-circle prefilter before the exact polygon clip.
                di, dj = dets[i], dets[j]
                if (np.hypot(di.center[0]-dj.center[0], di.center[1]-dj.center[1])
                        > _half_diagonal(di) + _half_diagonal(dj)):
                    continue
                old_iou = polygon_iou(di.vertices, dj.vertices)
                if old_iou <= cfg.physical_nms_iou:
                    continue
                a, b = dets[i], dets[j]
                candidate_idx = np.union1d(
                    _local_box_candidates(index, a, cfg.raw_refine_margin),
                    _local_box_candidates(index, b, cfg.raw_refine_margin),
                )
                local_xy = xy[candidate_idx]
                union = np.zeros(len(candidate_idx), dtype=bool)
                costs = []
                for d in (a, b):
                    xo, yo = to_box_frame_2d(
                        local_xy,
                        d.center[0],
                        d.center[1],
                        np.deg2rad(d.rotation_deg),
                    )
                    union |= (
                        (np.abs(xo) <= d.length / 2 + cfg.raw_refine_margin)
                        & (np.abs(yo) <= d.width / 2 + cfg.raw_refine_margin)
                    )
                    costs.append(
                        (xo / max(d.length / 2, 0.1)) ** 2
                        + (yo / max(d.width / 2, 0.1)) ** 2
                    )
                local_idx = np.flatnonzero(union)
                idx = candidate_idx[local_idx]
                choose_b = costs[1][local_idx] < costs[0][local_idx]
                na = _detection_from_points(xyz[idx[~choose_b]], a, cfg)
                nb = _detection_from_points(xyz[idx[choose_b]], b, cfg)
                if na and nb and polygon_iou(na.vertices, nb.vertices) < old_iou:
                    dets[i], dets[j] = na, nb
                    changed = True
        if not changed:
            break
    return dets


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ─────────────────────────────────────────────────────────────────────────────
def detect_vehicles_rect(
    points: np.ndarray,
    ground_mask: np.ndarray | None = None,
    *,
    hag: np.ndarray | None = None,
    config: RectSearchConfig | None = None,
) -> list[RectDetection]:
    """Shape-prior rotated-rectangle vehicle search.

    Args:
        points: (N, 3+) XYZ point cloud.
        ground_mask: boolean (N,) marking ground points, used to build the DTM
            when *hag* is not supplied.  If both are ``None`` the lowest 30 % of
            points by Z is used as a rough ground proxy.
        hag: optional precomputed per-point height-above-ground (N,).
        config: :class:`RectSearchConfig`.

    Returns:
        List of :class:`RectDetection`, NMS-deduplicated, sorted by score.
    """
    cfg = config or RectSearchConfig()
    xyz = np.asarray(points, dtype=np.float64)[:, :3]

    if hag is None:
        if ground_mask is not None and int(np.count_nonzero(ground_mask)) >= 3:
            ground_xyz = xyz[ground_mask]
        else:
            thr = np.percentile(xyz[:, 2], 30.0)
            ground_xyz = xyz[xyz[:, 2] <= thr]
        dtm = fill_nodata_nn(
            RasterGrid.from_point_cloud(ground_xyz, cell_size=cfg.dtm_cell_size), stat="min"
        )
        hag = height_above_ground(xyz, dtm)

    band = np.isfinite(hag) & (hag >= cfg.hag_min) & (hag <= cfg.hag_max)
    if int(band.sum()) < cfg.min_cells:
        return []

    band_xyz = xyz[band].copy()
    band_xyz[:, 2] = hag[band]                         # raster Z = HAG for height feature
    return search_points(band_xyz, config=cfg)


def instance_ids_from_rect(
    points: np.ndarray,
    *,
    config: RectSearchConfig | None = None,
) -> tuple[np.ndarray, list[RectDetection]]:
    """Turn a point set into instances via the rectangle search (semantic→instance).

    Runs :func:`search_points` on *points* (intended to be a single semantic
    class, e.g. all car points of a tile) and assigns **every** point to a
    rectangle: a point inside a rectangle takes that rectangle's id (the
    highest-scoring one if it falls in several); points outside all rectangles
    go to the nearest rectangle centre — so the labelling is a full partition,
    no point is dropped.

    Returns:
        ``(instance_id, detections)`` where ``instance_id`` is a 1-based int32
        array (one id per input point; matches ``detections`` ordered by score),
        and ``detections`` is the score-sorted list of :class:`RectDetection`.
        With no detections every point is instance 1 (single fallback instance).
    """
    cfg = config or RectSearchConfig()
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    n = len(xyz)
    inst = np.zeros(n, dtype=np.int32)
    if n == 0:
        return inst, []

    dets = search_points(xyz, config=cfg)          # already score-sorted desc
    if not dets:
        inst[:] = 1
        return inst, dets

    xy = xyz[:, :2]
    assigned = np.zeros(n, dtype=bool)
    for k, d in enumerate(dets, start=1):  # highest score first → wins overlaps
        cx, cy = d.center
        xo, yo = to_box_frame_2d(xy, cx, cy, np.deg2rad(d.rotation_deg))
        ins = (np.abs(xo) <= d.length / 2 + 1e-6) & (np.abs(yo) <= d.width / 2 + 1e-6) & (~assigned)
        inst[ins] = k
        assigned[ins] = True

    left = ~assigned
    if left.any():  # outside every box → nearest centre
        centers = np.array([d.center for d in dets])           # (K, 2)
        d2 = ((xy[left][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inst[left] = d2.argmin(axis=1).astype(np.int32) + 1
    return inst, dets


def search_points(
    points: np.ndarray,
    *,
    config: RectSearchConfig | None = None,
) -> list[RectDetection]:
    """Run the rotated-rectangle search directly on a point set (no ground / HAG).

    Rasterises *points* into a BEV occupancy grid and carves vehicle-sized
    rectangles out of every connected component.  Use this when the input is
    already the candidate band — e.g. an isolated set of vehicle-only points, to
    measure the searcher's upper-bound recall decoupled from ground / clutter.

    Args:
        points: (N, 3+) XYZ.  The Z column is used only for the per-detection
            height feature; the search itself is 2-D (footprint).
        config: :class:`RectSearchConfig`.

    Returns:
        List of :class:`RectDetection`, NMS-deduplicated, sorted by score.
    """
    cfg = config or RectSearchConfig()
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    if len(xyz) < cfg.min_cells:
        return []

    raster = RasterGrid.from_point_cloud(xyz, cell_size=cfg.cell_size)
    centers = raster.cell_centers("max")               # (M, 3): world XY + max Z
    cc_all = centers[:, :2]
    ch_all = centers[:, 2]
    cnt_all = raster.aggregation.count

    detections: list[RectDetection] = []
    for comp in raster.connected_components(diagonal=cfg.diagonal, min_size=cfg.min_cells):
        if comp.size > cfg.max_component_cells:
            continue                               # building / dense vegetation
        idx = raster.aggregation.indices_for_coords(comp.coords, raster.grid_shape)
        idx = idx[idx >= 0]
        if len(idx) < cfg.min_cells:
            continue
        detections.extend(
            search_component(cc_all[idx], cnt_all[idx], ch_all[idx], cfg)
        )

    if cfg.raw_point_refine:
        raw_index = SpatialIndex(xyz)
        detections = _refine_rectangles_from_raw_points(xyz, detections, cfg, index=raw_index)
        detections = _resolve_overlaps_by_repartition(xyz, detections, cfg, index=raw_index)
        # Split must see the complete local semantic blob. Exclusive ownership
        # hides the bridge/tail that signals two touching vehicles.
        detections = _split_nonrectangular_detections(xyz, detections, cfg, index=raw_index)
        detections = _resolve_overlaps_by_repartition(xyz, detections, cfg, index=raw_index)
    # Final physical exclusion after raw-point geometry refinement.  This is
    # separate from proposal NMS because refinement can enlarge rectangles.
    detections = nms_rectangles(detections, cfg.nms_iou)
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections
