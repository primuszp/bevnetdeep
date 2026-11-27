from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

SUPPORTED_STATS = ("min", "max", "mean", "count")
_MAX_DENSE_AGGREGATION_CELLS = 5_000_000
# Measurements on synthetic and real LAS data put the dense/sort crossover
# around 1.7 cells per point. Stay just above it while the absolute cell cap
# below keeps the four dense accumulator arrays bounded.
_MAX_DENSE_TO_POINT_RATIO = 1.75
# Sampling uses an int32 cell -> sparse-row lookup. Keep the cap conservative:
# ten million cells cost 40 MB and avoid allocating one dense float array per
# requested statistic.
MAX_DENSE_SAMPLE_CELLS = 10_000_000
DENSE_SAMPLE_TO_POINT_RATIO = 8


@dataclass(frozen=True)
class SparseAggregation:
    coords: np.ndarray
    count: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    mean: np.ndarray
    linear_index: np.ndarray | None = None

    def values(self, stat: str) -> np.ndarray:
        if stat == "min":
            return self.minimum
        if stat == "max":
            return self.maximum
        if stat == "mean":
            return self.mean
        if stat == "count":
            return self.count
        raise ValueError(f"Unsupported stat '{stat}'. Use one of: {', '.join(SUPPORTED_STATS)}")

    def indices_for_coords(self, coords: np.ndarray, grid_shape: tuple[int, ...]) -> np.ndarray:
        if self.linear_index is None:
            raise ValueError("SparseAggregation has no linear index")
        query = np.asarray(coords, dtype=np.int64)
        out = np.full(len(query), -1, dtype=np.int64)
        if len(self.linear_index) == 0 or len(query) == 0:
            return out
        shape = tuple(int(v) for v in grid_shape)
        query_linear = linearize_coords(query, shape)
        pos = np.searchsorted(self.linear_index, query_linear)
        n = len(self.linear_index)
        valid = (pos < n) & (self.linear_index[np.minimum(pos, n - 1)] == query_linear)
        out[valid] = pos[valid]
        return out


def normalize_cell_size(cell_size: float | Iterable[float], dims: int) -> np.ndarray:
    arr = np.asarray(cell_size, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(dims, float(arr), dtype=np.float64)
    if arr.shape != (dims,):
        raise ValueError(f"cell_size must be a scalar or a length-{dims} sequence")
    if np.any(arr <= 0):
        raise ValueError("cell_size values must be > 0")
    return arr


def validate_points(points: np.ndarray, dims: int = 3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < dims:
        raise ValueError(f"points must be an Nx{dims}+ array")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(pts[:, :dims]).all():
        raise ValueError("point coordinates must be finite")
    return pts


def aggregate_sparse(
    coords: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, ...] | None = None,
) -> SparseAggregation:
    coords = np.asarray(coords, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must be an NxD integer array")
    if values.ndim != 1 or len(values) != len(coords):
        raise ValueError("values must be a 1D array with the same length as coords")
    if len(coords) == 0:
        dims = coords.shape[1] if coords.ndim == 2 else 0
        empty_i = np.empty((0, dims), dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        empty_c = np.empty(0, dtype=np.int64)
        empty_linear = np.empty(0, dtype=np.int64) if grid_shape is not None else None
        return SparseAggregation(empty_i, empty_c, empty_f, empty_f, empty_f, empty_linear)

    # With a known shape the integer cell ID is sufficient for grouping. The
    # dedicated linear path avoids sorting/copying the full NxD coordinate
    # matrix, which is especially important for large voxel clouds.
    if grid_shape is None:
        order = np.lexsort(tuple(coords[:, dim] for dim in range(coords.shape[1] - 1, -1, -1)))
    else:
        shape = tuple(int(v) for v in grid_shape)
        if len(shape) != coords.shape[1]:
            raise ValueError("grid_shape dimensionality must match coords")
        linear = linearize_coords(coords, shape)
        return aggregate_linear(linear, values, shape)

    sorted_coords = coords[order]
    sorted_values = values[order]
    starts = np.flatnonzero(
        np.r_[True, np.any(sorted_coords[1:] != sorted_coords[:-1], axis=1)]
    )
    unique_coords = sorted_coords[starts]
    counts = np.diff(np.r_[starts, len(sorted_coords)])
    sums = np.add.reduceat(sorted_values, starts)
    minimum = np.minimum.reduceat(sorted_values, starts)
    maximum = np.maximum.reduceat(sorted_values, starts)

    return SparseAggregation(
        coords=unique_coords,
        count=counts.astype(np.int64, copy=False),
        minimum=minimum,
        maximum=maximum,
        mean=sums / counts,
        linear_index=None,
    )


def use_dense_aggregation(
    grid_size: int,
    point_count: int,
    *,
    max_cells: int = _MAX_DENSE_AGGREGATION_CELLS,
) -> bool:
    """Return whether dense accumulators are faster within the memory budget."""
    return (
        grid_size <= max_cells
        and grid_size <= _MAX_DENSE_TO_POINT_RATIO * point_count
    )


def grid_cell_count(shape: tuple[int, ...]) -> int:
    """Exact grid size, rejecting domains that cannot use int64 cell IDs."""
    total = math.prod(int(size) for size in shape)
    if total > np.iinfo(np.int64).max:
        raise ValueError("grid is too large for int64 linear cell IDs")
    return total


def linearize_coords(coords: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Convert integer grid coordinates to C-independent x-fast cell IDs."""
    points = np.asarray(coords)
    if points.ndim != 2 or points.shape[1] != len(shape):
        raise ValueError("coordinate dimensionality must match shape")
    grid_cell_count(shape)
    # Horner form avoids the temporary NxD multiplication used by sum().
    linear = points[:, -1].astype(np.int64, copy=True)
    for dim in range(len(shape) - 2, -1, -1):
        linear *= shape[dim]
        linear += points[:, dim]
    return linear


def _coords_from_linear(linear: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    dtype = np.int32 if max(shape, default=0) <= np.iinfo(np.int32).max else np.int64
    coords = np.empty((len(linear), len(shape)), dtype=dtype)
    remainder = linear
    for dim, size in enumerate(shape[:-1]):
        remainder, coordinate = np.divmod(remainder, size)
        coords[:, dim] = coordinate
    coords[:, -1] = remainder
    return coords


@dataclass(frozen=True)
class GroupedStats:
    """Per-cell count/sum/min/max keyed by sorted linear cell ID."""

    cell_ids: np.ndarray
    count: np.ndarray
    total: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def group_by_cell(
    cell_ids: np.ndarray,
    values: np.ndarray,
    grid_size: int,
    *,
    max_dense_cells: int = _MAX_DENSE_AGGREGATION_CELLS,
) -> GroupedStats:
    """Group values by linear cell ID, choosing the cheaper of two strategies.

    Dense accumulators (bincount plus unbuffered min/max) win for compact
    grids; sparse grids use one integer sort and a reduceat pass. This is the
    single grouping core behind both in-memory and streaming aggregation.
    """
    ids = np.asarray(cell_ids, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    if use_dense_aggregation(grid_size, len(ids), max_cells=max_dense_cells):
        count = np.bincount(ids, minlength=grid_size)
        total = np.bincount(ids, weights=vals, minlength=grid_size)
        minimum = np.full(grid_size, np.inf, dtype=np.float64)
        maximum = np.full(grid_size, -np.inf, dtype=np.float64)
        np.minimum.at(minimum, ids, vals)
        np.maximum.at(maximum, ids, vals)
        occupied = np.flatnonzero(count)
        return GroupedStats(
            cell_ids=occupied,
            count=count[occupied],
            total=total[occupied],
            minimum=minimum[occupied],
            maximum=maximum[occupied],
        )

    order = np.argsort(ids)
    sorted_ids = ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    sorted_values = vals[order]
    return GroupedStats(
        cell_ids=sorted_ids[starts],
        count=np.diff(np.r_[starts, len(sorted_ids)]).astype(np.int64, copy=False),
        total=np.add.reduceat(sorted_values, starts),
        minimum=np.minimum.reduceat(sorted_values, starts),
        maximum=np.maximum.reduceat(sorted_values, starts),
    )


def aggregate_linear(
    linear: np.ndarray,
    values: np.ndarray,
    grid_shape: tuple[int, ...],
) -> SparseAggregation:
    """Aggregate values by precomputed linear cell ID.

    Dense accumulation wins for compact grids; sparse grids use one integer
    sort and reconstruct coordinates only for occupied cells.
    """
    cell_ids = np.asarray(linear, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    shape = tuple(int(v) for v in grid_shape)
    if cell_ids.ndim != 1 or vals.ndim != 1 or len(cell_ids) != len(vals):
        raise ValueError("linear and values must be same-length 1D arrays")
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("grid_shape must contain positive dimensions")
    if len(cell_ids) == 0:
        empty_i = np.empty((0, len(shape)), dtype=np.int32)
        empty_f = np.empty(0, dtype=np.float64)
        empty_c = np.empty(0, dtype=np.int64)
        return SparseAggregation(empty_i, empty_c, empty_f, empty_f, empty_f, empty_c)

    grid_size = grid_cell_count(shape)
    if cell_ids.min() < 0 or cell_ids.max() >= grid_size:
        raise ValueError("linear cell IDs must be inside grid_shape")

    grouped = group_by_cell(cell_ids, vals, grid_size)
    return SparseAggregation(
        coords=_coords_from_linear(grouped.cell_ids, shape),
        count=grouped.count,
        minimum=grouped.minimum,
        maximum=grouped.maximum,
        mean=grouped.total / grouped.count,
        linear_index=grouped.cell_ids,
    )


def dense_row_lookup(aggregation: SparseAggregation, grid_size: int) -> np.ndarray:
    """Map each linear cell ID to its row in a sparse aggregation (-1 = empty)."""
    if aggregation.linear_index is None:
        raise ValueError("SparseAggregation has no linear index")
    dtype = np.int32 if len(aggregation.coords) <= np.iinfo(np.int32).max else np.int64
    lookup = np.full(grid_size, -1, dtype=dtype)
    lookup[aggregation.linear_index] = np.arange(len(aggregation.coords), dtype=dtype)
    return lookup
