from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import convex_hull_2d, oriented_bounding_box_2d


@dataclass(frozen=True)
class InstanceBBoxLimits:
    min_length: float = 1.2
    max_length: float = 16.0
    min_width: float = 0.6
    max_width: float = 4.5
    min_height: float = 0.3
    max_height: float = 4.5

    def validate(self) -> None:
        if not (0 < self.min_length <= self.max_length):
            raise ValueError("Invalid length limits")
        if not (0 < self.min_width <= self.max_width):
            raise ValueError("Invalid width limits")
        if not (0 <= self.min_height <= self.max_height):
            raise ValueError("Invalid height limits")


@dataclass(frozen=True)
class DbscanConfig:
    eps: float = 0.45
    min_samples: int = 8
    min_cluster_points: int = 10
    cluster_space: Literal["xy", "xyz"] = "xyz"
    bbox_limits: InstanceBBoxLimits = InstanceBBoxLimits()


@dataclass(frozen=True)
class FecInstanceConfig:
    """Fast XY instance segmentation without the scikit-learn dependency.

    ``max_neighbors`` bounds the KD-tree result size.  This keeps peak memory
    predictable on dense airborne point clouds and is the main performance
    advantage over an unrestricted radius-neighbour graph.
    """

    radius: float = 0.40
    min_cluster_points: int = 10
    max_cluster_points: int | None = None
    max_neighbors: int = 50
    bbox_limits: InstanceBBoxLimits = InstanceBBoxLimits()
    deterministic_ids: bool = True


@dataclass(frozen=True)
class InstancePostprocessConfig:
    """Shape-aware cleanup applied after any instance segmenter."""

    split_max_length: float = 16.0
    split_max_width: float = 4.5
    split_min_points: int = 40
    merge_gap: float = 0.5
    merge_max_length: float = 16.0
    merge_max_width: float = 4.5
    merge_max_height: float = 4.5
    orphan_radius: float = 0.0


@dataclass(frozen=True)
class InstanceSegmentationStats:
    n_points: int
    n_instances_before_filter: int
    n_instances_after_filter: int
    n_clustered_after_filter: int
    n_noise_after_filter: int


def _require_dbscan():
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("scikit-learn is required for DBSCAN instance segmentation") from exc
    return DBSCAN


def labels_to_instance_id(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    out = np.zeros(len(labels), dtype=np.uint32)
    valid = labels >= 0
    if np.any(valid):
        _, inverse = np.unique(labels[valid], return_inverse=True)
        out[valid] = inverse.astype(np.uint32, copy=False) + 1
    return out


def reindex_instance_id(instance_id: np.ndarray) -> np.ndarray:
    ids = np.asarray(instance_id, dtype=np.uint32)
    out = np.zeros_like(ids)
    valid = ids > 0
    if np.any(valid):
        _, inverse = np.unique(ids[valid], return_inverse=True)
        out[valid] = inverse.astype(np.uint32, copy=False) + 1
    return out


def reindex_instance_id_by_centroid(
    points: np.ndarray,
    instance_id: np.ndarray,
) -> np.ndarray:
    """Assign stable IDs ordered by XY centroid.

    Cluster algorithms may return different numeric IDs when point order or
    worker scheduling changes.  Sorting centroids makes exported instance IDs
    reproducible while preserving the fast vectorised remapping path.
    """
    pts = np.asarray(points, dtype=np.float64)
    ids = np.asarray(instance_id, dtype=np.uint32)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) != len(ids):
        raise ValueError("points must be Nx2+ and match instance_id length")

    unique = np.unique(ids[ids > 0])
    if len(unique) == 0:
        return np.zeros_like(ids)

    # Reduce centroids without one boolean scan per instance.  On many small
    # instances this is substantially faster than repeatedly evaluating
    # ``ids == current_id``.
    valid = ids > 0
    mapped = np.searchsorted(unique, ids[valid])
    count = np.bincount(mapped, minlength=len(unique))
    sum_x = np.bincount(mapped, weights=pts[valid, 0], minlength=len(unique))
    sum_y = np.bincount(mapped, weights=pts[valid, 1], minlength=len(unique))
    centroid_x = sum_x / count
    centroid_y = sum_y / count
    order = np.lexsort((unique, np.round(centroid_y, 6), np.round(centroid_x, 6)))
    new_for_unique = np.empty(len(unique), dtype=np.uint32)
    new_for_unique[order] = np.arange(1, len(unique) + 1, dtype=np.uint32)
    out = np.zeros_like(ids)
    out[valid] = new_for_unique[mapped]
    return out


def _instance_length_width_height(points: np.ndarray) -> tuple[float, float, float]:
    xy = points[:, :2]
    hull = convex_hull_2d(xy)
    if len(hull) >= 3:
        obb = oriented_bounding_box_2d(hull)
        length, width = obb.length, obb.width
    else:
        span = xy.max(axis=0) - xy.min(axis=0)
        length, width = float(max(span)), float(min(span))
    height = float(points[:, 2].max() - points[:, 2].min())
    return float(length), float(width), height


def _positive_index_groups(ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Group positive labels with one sort instead of one full scan per label."""
    positive_index = np.flatnonzero(ids > 0)
    if len(positive_index) == 0:
        return np.empty(0, dtype=ids.dtype), []
    order = np.argsort(ids[positive_index])
    sorted_index = positive_index[order]
    sorted_labels = ids[sorted_index]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    stops = np.r_[starts[1:], len(sorted_index)]
    return sorted_labels[starts], [sorted_index[start:stop] for start, stop in zip(starts, stops, strict=True)]


def filter_instances_by_bbox(
    points: np.ndarray,
    instance_id: np.ndarray,
    limits: InstanceBBoxLimits,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ids = np.asarray(instance_id, dtype=np.uint32).copy()
    limits.validate()

    _, groups = _positive_index_groups(ids)
    for index in groups:
        length, width, height = _instance_length_width_height(pts[index])
        if not (
            limits.min_length <= length <= limits.max_length
            and limits.min_width <= width <= limits.max_width
            and limits.min_height <= height <= limits.max_height
        ):
            ids[index] = 0
    return reindex_instance_id(ids)


def segment_instances_dbscan(
    points: np.ndarray,
    config: DbscanConfig | None = None,
) -> tuple[np.ndarray, InstanceSegmentationStats]:
    cfg = config or DbscanConfig()
    if cfg.eps <= 0:
        raise ValueError("eps must be > 0")
    if cfg.min_samples <= 0:
        raise ValueError("min_samples must be > 0")
    if cfg.min_cluster_points <= 0:
        raise ValueError("min_cluster_points must be > 0")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")

    DBSCAN = _require_dbscan()
    coords = pts[:, :2] if cfg.cluster_space == "xy" else pts[:, :3]
    labels = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples, n_jobs=-1).fit_predict(coords)
    valid = labels >= 0
    if np.any(valid):
        counts = np.bincount(labels[valid])
        labels[valid & (counts[labels.clip(min=0)] < cfg.min_cluster_points)] = -1

    before = labels_to_instance_id(labels)
    after = filter_instances_by_bbox(pts[:, :3], before, cfg.bbox_limits)
    stats = InstanceSegmentationStats(
        n_points=len(pts),
        n_instances_before_filter=int(before.max(initial=0)),
        n_instances_after_filter=int(after.max(initial=0)),
        n_clustered_after_filter=int(np.sum(after > 0)),
        n_noise_after_filter=int(np.sum(after == 0)),
    )
    return after, stats


def segment_instances_fec(
    points: np.ndarray,
    config: FecInstanceConfig | None = None,
) -> tuple[np.ndarray, InstanceSegmentationStats]:
    """Segment object instances with fast Euclidean clustering in XY.

    This is the dependency-free, high-throughput alternative to DBSCAN.  XY
    clustering avoids vertical returns bridging neighbouring vehicles; XYZ is
    still used by the subsequent oriented bounding-box filter.
    """
    from .clustering import fec_cluster

    cfg = config or FecInstanceConfig()
    if cfg.radius <= 0 or cfg.min_cluster_points <= 0 or cfg.max_neighbors <= 0:
        raise ValueError("radius, min_cluster_points and max_neighbors must be > 0")
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")

    before = fec_cluster(
        pts,
        cfg.radius,
        min_cluster_size=cfg.min_cluster_points,
        max_cluster_size=cfg.max_cluster_points,
        use_3d=False,
        max_neighbors=cfg.max_neighbors,
    ).astype(np.uint32, copy=False)
    after = filter_instances_by_bbox(pts[:, :3], before, cfg.bbox_limits)
    if cfg.deterministic_ids:
        after = reindex_instance_id_by_centroid(pts, after)
    stats = InstanceSegmentationStats(
        n_points=len(pts),
        n_instances_before_filter=int(before.max(initial=0)),
        n_instances_after_filter=int(after.max(initial=0)),
        n_clustered_after_filter=int(np.count_nonzero(after)),
        n_noise_after_filter=int(np.count_nonzero(after == 0)),
    )
    return after, stats


def collect_instance_hulls(points: np.ndarray, instance_id: np.ndarray) -> list[tuple[int, np.ndarray]]:
    pts = np.asarray(points, dtype=np.float64)
    ids = np.asarray(instance_id)
    hulls: list[tuple[int, np.ndarray]] = []
    instances, groups = _positive_index_groups(ids)
    for instance, index in zip(instances, groups, strict=True):
        xy = pts[index, :2]
        if len(xy) < 3:
            continue
        hull = convex_hull_2d(xy)
        if len(hull) >= 3:
            hulls.append((int(instance), hull))
    return hulls


def postprocess_instances(
    points: np.ndarray,
    instance_id: np.ndarray,
    config: InstancePostprocessConfig | None = None,
) -> np.ndarray:
    """Split merged objects, merge fragments, and optionally absorb noise.

    The implementation reuses BorderEye's shape-guided cluster splitter and
    limits merge comparisons with a KD-tree.  Runtime therefore scales with
    nearby instance pairs instead of all possible pairs.
    """
    from scipy.spatial import cKDTree

    from .clustering import refine_cluster_labels

    cfg = config or InstancePostprocessConfig()
    pts = np.asarray(points, dtype=np.float64)
    ids = np.asarray(instance_id, dtype=np.uint32)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) != len(ids):
        raise ValueError("points must be Nx3+ and match instance_id length")
    if min(cfg.split_max_length, cfg.split_max_width, cfg.merge_max_length,
           cfg.merge_max_width, cfg.merge_max_height) <= 0:
        raise ValueError("box limits must be > 0")
    if cfg.merge_gap < 0 or cfg.orphan_radius < 0:
        raise ValueError("merge_gap and orphan_radius must be >= 0")

    # Split L-shaped/chained and physically oversized clusters.  The shared
    # splitter is faster and better tested than OmniEye's separate DBSCAN
    # fallback implementation.
    out = refine_cluster_labels(
        pts,
        ids,
        max_length=cfg.split_max_length,
        max_width=cfg.split_max_width,
        min_split_points=cfg.split_min_points,
    ).astype(np.uint32, copy=False)

    active, point_groups = _positive_index_groups(out)
    if len(active) > 1 and cfg.merge_gap > 0:
        centroids = np.vstack([pts[index, :2].mean(axis=0) for index in point_groups])
        # Any valid merged box has centroids closer than its maximum length
        # plus the allowed gap, so distant pairs can be discarded safely.
        candidate_pairs = cKDTree(centroids).query_pairs(cfg.merge_max_length + cfg.merge_gap)
        parent = np.arange(len(active), dtype=np.int64)

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for ia, ib in sorted(candidate_pairs):
            ra, rb = find(ia), find(ib)
            if ra == rb:
                continue
            index_a, index_b = point_groups[ra], point_groups[rb]
            a, b = pts[index_a], pts[index_b]
            # Fast AABB lower bound before the more expensive OBB fit.
            amin, amax = a[:, :2].min(0), a[:, :2].max(0)
            bmin, bmax = b[:, :2].min(0), b[:, :2].max(0)
            gap = np.linalg.norm(np.maximum(0.0, np.maximum(amin, bmin) - np.minimum(amax, bmax)))
            if gap > cfg.merge_gap:
                continue
            length, width, height = _instance_length_width_height(np.vstack((a, b)))
            if (length <= cfg.merge_max_length and width <= cfg.merge_max_width
                    and height <= cfg.merge_max_height):
                parent[rb] = ra
                point_groups[ra] = np.concatenate((index_a, index_b))
                point_groups[rb] = np.empty(0, dtype=np.int64)

        for i, index in enumerate(point_groups):
            if len(index):
                out[index] = active[find(i)]

    if cfg.orphan_radius > 0 and np.any(out == 0) and np.any(out > 0):
        active, point_groups = _positive_index_groups(out)
        centroids = np.vstack([pts[index, :2].mean(axis=0) for index in point_groups])
        noise_index = np.flatnonzero(out == 0)
        distance, nearest = cKDTree(centroids).query(pts[noise_index, :2], workers=-1)
        accepted = distance <= cfg.orphan_radius
        out[noise_index[accepted]] = active[nearest[accepted]]

    return reindex_instance_id_by_centroid(pts, out)


def attach_instance_id_to_las(
    las: object,
    instance_id: np.ndarray,
    *,
    dimension_name: str = "instance_id",
    overwrite: bool = True,
) -> object:
    from .io import _require_laspy
    laspy = _require_laspy()

    ids = np.asarray(instance_id, dtype=np.uint32)
    if len(ids) != len(las.points):
        raise ValueError("instance_id length must match LAS point count")

    header = las.header.copy()
    exists = dimension_name in header.point_format.dimension_names
    if exists and not overwrite:
        raise ValueError(f"LAS already has '{dimension_name}' dimension")
    if not exists:
        header.add_extra_dim(laspy.ExtraBytesParams(name=dimension_name, type=np.uint32))

    out = laspy.LasData(header)
    out.points = laspy.ScaleAwarePointRecord.zeros(len(las.points), header=header)
    for dim in las.point_format.dimension_names:
        if dim in header.point_format.dimension_names:
            out[dim] = las[dim]
    out[dimension_name] = ids
    return out
