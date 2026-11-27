"""Reusable spatial index for common point-cloud neighbourhood queries."""
from __future__ import annotations

from threading import Lock

import numpy as np
from scipy.spatial import cKDTree


def points_in_polygon_2d(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Return a boundary-inclusive mask for points inside a simple polygon."""
    xy = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(vertices, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("points must be an Nx2+ array")
    if polygon.ndim != 2 or polygon.shape[1] < 2 or len(polygon) < 3:
        raise ValueError("vertices must be an Nx2+ array with at least three rows")

    x = xy[:, 0]
    y = xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    boundary = np.zeros(len(xy), dtype=bool)
    scale = max(float(np.ptp(polygon[:, :2], axis=0).max()), 1.0)
    eps = np.finfo(np.float64).eps * scale * 16.0

    previous = polygon[-1, :2]
    for current in polygon[:, :2]:
        x1, y1 = previous
        x2, y2 = current
        dx, dy = x2 - x1, y2 - y1

        # Include points on an edge; ray casting alone is ambiguous there.
        cross = (x - x1) * dy - (y - y1) * dx
        on_segment = (
            (np.abs(cross) <= eps)
            & (x >= min(x1, x2) - eps)
            & (x <= max(x1, x2) + eps)
            & (y >= min(y1, y2) - eps)
            & (y <= max(y1, y2) + eps)
        )
        boundary |= on_segment

        crosses_y = (y1 > y) != (y2 > y)
        x_intersection = dx * (y - y1) / (dy if abs(dy) > eps else 1.0) + x1
        inside ^= crosses_y & (x < x_intersection)
        previous = current
    return inside | boundary


class SpatialIndex:
    """One point-cloud index reused by radius, KNN, box and polygon queries.

    XY and XYZ KD-trees are built lazily and cached independently. This keeps
    construction cheap when a workflow only needs planar or only 3-D queries.
    All query methods return indices into the original point order.
    """

    def __init__(self, points: np.ndarray, *, leaf_size: int = 16) -> None:
        self._source = points
        cloud = np.asarray(points, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] < 2:
            raise ValueError("points must be an Nx2+ array")
        # Only the queryable coordinate columns must be finite. Extra columns
        # (intensity, classification, ...) are never indexed and legitimately
        # carry NaN, so validating them would reject valid clouds.
        if not np.isfinite(cloud[:, :min(3, cloud.shape[1])]).all():
            raise ValueError("point coordinates must be finite")
        if leaf_size <= 0:
            raise ValueError("leaf_size must be > 0")
        self._points = cloud
        self.leaf_size = int(leaf_size)
        self._trees: dict[int, cKDTree] = {}
        self._tree_lock = Lock()

    @property
    def points(self) -> np.ndarray:
        """Read-only-by-convention point array used to build the index."""
        return self._points

    @property
    def size(self) -> int:
        return len(self._points)

    @property
    def built_dimensions(self) -> tuple[int, ...]:
        """Dimensions whose lazy KD-tree has already been materialized."""
        return tuple(sorted(self._trees))

    def indexes(self, points: np.ndarray) -> bool:
        """Whether this index was created for the supplied point container."""
        if points is self._source:
            return True
        candidate = np.asarray(points)
        return candidate.shape == self._points.shape and np.shares_memory(candidate, self._points)

    def _dimensions(self, use_3d: bool) -> int:
        dimensions = 3 if use_3d else 2
        if self._points.shape[1] < dimensions:
            raise ValueError("3-D queries require an Nx3+ point array")
        return dimensions

    def _tree(self, dimensions: int) -> cKDTree:
        tree = self._trees.get(dimensions)
        if tree is None:
            with self._tree_lock:
                tree = self._trees.get(dimensions)
                if tree is None:
                    tree = cKDTree(
                        self._points[:, :dimensions],
                        leafsize=self.leaf_size,
                        compact_nodes=True,
                        # Sliding-midpoint construction is substantially
                        # faster on LiDAR clouds and measured no query penalty.
                        balanced_tree=False,
                    )
                    self._trees[dimensions] = tree
        return tree

    def radius(
        self,
        center: np.ndarray | tuple[float, ...],
        radius: float,
        *,
        use_3d: bool = True,
        workers: int = -1,
    ) -> np.ndarray:
        """Indices within *radius* of one XY or XYZ center."""
        if radius < 0:
            raise ValueError("radius must be >= 0")
        dimensions = self._dimensions(use_3d)
        query = np.asarray(center, dtype=np.float64)
        if query.shape != (dimensions,):
            raise ValueError(f"center must contain exactly {dimensions} coordinates")
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        result = np.asarray(
            self._tree(dimensions).query_ball_point(query, radius, workers=workers),
            dtype=np.int64,
        )
        result.sort()
        return result

    def radius_batch(
        self,
        centers: np.ndarray,
        radius: float,
        *,
        use_3d: bool = True,
        workers: int = -1,
    ) -> list[np.ndarray]:
        """Radius-query several centers while reusing one cached tree."""
        if radius < 0:
            raise ValueError("radius must be >= 0")
        dimensions = self._dimensions(use_3d)
        queries = np.asarray(centers, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != dimensions:
            raise ValueError(f"centers must have shape (N, {dimensions})")
        if self.size == 0:
            return [np.empty(0, dtype=np.int64) for _ in queries]
        groups = self._tree(dimensions).query_ball_point(queries, radius, workers=workers)
        result = []
        for group in groups:
            indices = np.asarray(group, dtype=np.int64)
            indices.sort()
            result.append(indices)
        return result

    def radius_counts(
        self,
        queries: np.ndarray,
        radius: float,
        *,
        use_3d: bool = True,
        workers: int = -1,
    ) -> np.ndarray:
        """Number of neighbours per query without materializing index lists."""
        if radius < 0:
            raise ValueError("radius must be >= 0")
        dimensions = self._dimensions(use_3d)
        query = np.asarray(queries, dtype=np.float64)
        if query.ndim == 1:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != dimensions:
            raise ValueError(f"queries must have shape ({dimensions},) or (N, {dimensions})")
        if self.size == 0:
            return np.zeros(len(query), dtype=np.int64)
        return np.asarray(
            self._tree(dimensions).query_ball_point(
                query,
                radius,
                workers=workers,
                return_length=True,
            ),
            dtype=np.int64,
        )

    def knn(
        self,
        queries: np.ndarray,
        k: int = 1,
        *,
        use_3d: bool = True,
        max_distance: float = np.inf,
        workers: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbour distances and original point indices.

        Missing neighbours beyond *max_distance* receive distance ``inf`` and
        index ``-1`` rather than SciPy's out-of-range sentinel.
        """
        if k <= 0:
            raise ValueError("k must be > 0")
        if max_distance < 0:
            raise ValueError("max_distance must be >= 0")
        dimensions = self._dimensions(use_3d)
        query = np.asarray(queries, dtype=np.float64)
        single = query.ndim == 1
        if single:
            query = query[None, :]
        if query.ndim != 2 or query.shape[1] != dimensions:
            raise ValueError(f"queries must have shape ({dimensions},) or (N, {dimensions})")
        distances, indices = self._tree(dimensions).query(
            query,
            k=k,
            distance_upper_bound=max_distance,
            workers=workers,
        )
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        indices[indices == self.size] = -1
        if single:
            return np.atleast_1d(distances[0]), np.atleast_1d(indices[0])
        return distances, indices

    def box(self, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        """Indices inside a boundary-inclusive 2-D or 3-D axis-aligned box."""
        lower = np.asarray(minimum, dtype=np.float64)
        upper = np.asarray(maximum, dtype=np.float64)
        if lower.ndim != 1 or lower.shape != upper.shape or len(lower) not in (2, 3):
            raise ValueError("minimum and maximum must be matching length-2 or length-3 arrays")
        if np.any(upper < lower):
            raise ValueError("maximum must be >= minimum")
        dimensions = len(lower)
        if self._points.shape[1] < dimensions:
            raise ValueError("3-D boxes require an Nx3+ point array")
        if self.size == 0:
            return np.empty(0, dtype=np.int64)
        center = (lower + upper) * 0.5
        half = (upper - lower) * 0.5
        candidates = self.radius(center, float(np.linalg.norm(half)), use_3d=dimensions == 3)
        selected = self._points[candidates, :dimensions]
        inside = np.all((selected >= lower) & (selected <= upper), axis=1)
        return candidates[inside]

    def polygon(
        self,
        vertices: np.ndarray,
        *,
        z_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Indices inside a simple XY polygon and optional inclusive Z band."""
        polygon = np.asarray(vertices, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] < 2 or len(polygon) < 3:
            raise ValueError("vertices must be an Nx2+ array with at least three rows")
        candidates = self.box(polygon[:, :2].min(axis=0), polygon[:, :2].max(axis=0))
        inside = points_in_polygon_2d(self._points[candidates, :2], polygon[:, :2])
        if z_bounds is not None:
            if self._points.shape[1] < 3:
                raise ValueError("z_bounds requires an Nx3+ point array")
            z_min, z_max = z_bounds
            if z_max < z_min:
                raise ValueError("z_bounds maximum must be >= minimum")
            z = self._points[candidates, 2]
            inside &= (z >= z_min) & (z <= z_max)
        return candidates[inside]

    def oriented_box(
        self,
        center: tuple[float, float] | np.ndarray,
        length: float,
        width: float,
        rotation_degrees: float = 0.0,
        *,
        z_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Indices inside a boundary-inclusive oriented XY rectangle."""
        if length < 0 or width < 0:
            raise ValueError("length and width must be >= 0")
        center_xy = np.asarray(center, dtype=np.float64)
        if center_xy.shape != (2,):
            raise ValueError("center must contain exactly two coordinates")
        radius = 0.5 * float(np.hypot(length, width))
        candidates = self.radius(center_xy, radius, use_3d=False)
        delta = self._points[candidates, :2] - center_xy
        angle = np.deg2rad(rotation_degrees)
        cosine, sine = np.cos(angle), np.sin(angle)
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        scale = max(length, width, 1.0)
        eps = np.finfo(np.float64).eps * scale * 16.0
        inside = (
            (np.abs(local_x) <= length * 0.5 + eps)
            & (np.abs(local_y) <= width * 0.5 + eps)
        )
        if z_bounds is not None:
            if self._points.shape[1] < 3:
                raise ValueError("z_bounds requires an Nx3+ point array")
            z_min, z_max = z_bounds
            if z_max < z_min:
                raise ValueError("z_bounds maximum must be >= minimum")
            z = self._points[candidates, 2]
            inside &= (z >= z_min - eps) & (z <= z_max + eps)
        return candidates[inside]

    def select(self, indices: np.ndarray) -> np.ndarray:
        """Return point rows for indices produced by a query method."""
        return self._points[np.asarray(indices, dtype=np.int64)]

    def mask(self, indices: np.ndarray) -> np.ndarray:
        """Convert query indices into a full-length boolean mask."""
        result = np.zeros(self.size, dtype=bool)
        result[np.asarray(indices, dtype=np.int64)] = True
        return result


def spatial_index(points: np.ndarray, *, leaf_size: int = 16) -> SpatialIndex:
    """Create one reusable index for common XY and XYZ point-cloud queries."""
    return SpatialIndex(points, leaf_size=leaf_size)
