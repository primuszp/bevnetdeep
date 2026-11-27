"""Bounded-memory LAS/LAZ aggregation for raster and voxel grids."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._aggregation import (
    GroupedStats,
    SparseAggregation,
    _coords_from_linear,
    grid_cell_count,
    group_by_cell,
    linearize_coords,
    normalize_cell_size,
    use_dense_aggregation,
)

_MAX_STREAMING_DENSE_CELLS = 10_000_000


def _merge_reduced(left: GroupedStats, right: GroupedStats) -> GroupedStats:
    """Merge two sorted partial aggregations without returning to point level."""
    cell_ids = np.concatenate((left.cell_ids, right.cell_ids))
    order = np.argsort(cell_ids)
    sorted_ids = cell_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    count = np.concatenate((left.count, right.count))[order]
    total = np.concatenate((left.total, right.total))[order]
    minimum = np.concatenate((left.minimum, right.minimum))[order]
    maximum = np.concatenate((left.maximum, right.maximum))[order]
    return GroupedStats(
        cell_ids=sorted_ids[starts],
        count=np.add.reduceat(count, starts),
        total=np.add.reduceat(total, starts),
        minimum=np.minimum.reduceat(minimum, starts),
        maximum=np.maximum.reduceat(maximum, starts),
    )


class _StreamingAggregation:
    """Incrementally combine point chunks with predictable peak memory."""

    def __init__(self, shape: tuple[int, ...], estimated_points: int) -> None:
        self.shape = shape
        self.grid_size = grid_cell_count(shape)
        # Streaming can devote more space to the final accumulators because
        # the full point matrix is never resident. At the cap, count/sum/min/
        # max consume about 320 MB in total.
        self._dense = use_dense_aggregation(
            self.grid_size,
            estimated_points,
            max_cells=_MAX_STREAMING_DENSE_CELLS,
        )
        self._levels: list[GroupedStats | None] = []
        if self._dense:
            self._count = np.zeros(self.grid_size, dtype=np.int64)
            self._total = np.zeros(self.grid_size, dtype=np.float64)
            self._minimum = np.full(self.grid_size, np.inf, dtype=np.float64)
            self._maximum = np.full(self.grid_size, -np.inf, dtype=np.float64)

    def update(self, cell_ids: np.ndarray, values: np.ndarray) -> None:
        if len(cell_ids) == 0:
            return
        reduced = group_by_cell(cell_ids, values, self.grid_size)
        if self._dense:
            ids = reduced.cell_ids
            self._count[ids] += reduced.count
            self._total[ids] += reduced.total
            self._minimum[ids] = np.minimum(self._minimum[ids], reduced.minimum)
            self._maximum[ids] = np.maximum(self._maximum[ids], reduced.maximum)
            return

        # Binary-carry merging bounds the number of resident partial arrays.
        # Each occupied cell participates in at most log2(chunks) merges.
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            reduced = _merge_reduced(self._levels[level], reduced)
            self._levels[level] = None
            level += 1
        if level == len(self._levels):
            self._levels.append(reduced)
        else:
            self._levels[level] = reduced

    def finish(self) -> SparseAggregation:
        if self._dense:
            cell_ids = np.flatnonzero(self._count)
            count = self._count[cell_ids]
            total = self._total[cell_ids]
            minimum = self._minimum[cell_ids]
            maximum = self._maximum[cell_ids]
        else:
            parts = [part for part in self._levels if part is not None]
            if not parts:
                cell_ids = np.empty(0, dtype=np.int64)
                count = np.empty(0, dtype=np.int64)
                total = minimum = maximum = np.empty(0, dtype=np.float64)
            else:
                combined = parts[0]
                for part in parts[1:]:
                    combined = _merge_reduced(combined, part)
                cell_ids = combined.cell_ids
                count = combined.count
                total = combined.total
                minimum = combined.minimum
                maximum = combined.maximum

        return SparseAggregation(
            coords=_coords_from_linear(cell_ids, self.shape),
            count=count,
            minimum=minimum,
            maximum=maximum,
            mean=total / count if len(count) else np.empty(0, dtype=np.float64),
            linear_index=cell_ids,
        )


def _validate_file(path: Path | str, chunk_size: int) -> Path:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("streaming grids currently support LAS and LAZ files")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return source


def _chunk_arrays(chunk) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize scaled coordinates once per LAS chunk."""
    return (
        np.asarray(chunk.x, dtype=np.float64),
        np.asarray(chunk.y, dtype=np.float64),
        np.asarray(chunk.z, dtype=np.float64),
    )


def _chunk_cell_ids(
    axes: tuple[np.ndarray, ...],
    values: np.ndarray,
    origin: np.ndarray,
    cell_size: np.ndarray,
    shape: tuple[int, ...],
    *,
    clip_to_bounds: bool,
) -> tuple[np.ndarray, np.ndarray]:
    coord_dtype = (
        np.int32
        if not clip_to_bounds and max(shape) <= np.iinfo(np.int32).max
        else np.int64
    )
    coords = np.empty((len(axes[0]), len(shape)), dtype=coord_dtype)
    for dimension, axis in enumerate(axes):
        coords[:, dimension] = np.floor(
            (axis - origin[dimension]) / cell_size[dimension]
        ).astype(coord_dtype)
    if clip_to_bounds:
        valid = np.ones(len(coords), dtype=bool)
        for dimension, size in enumerate(shape):
            valid &= (coords[:, dimension] >= 0) & (coords[:, dimension] < size)
        coords = coords[valid]
        values = values[valid]
    return linearize_coords(coords, shape), values


def rasterize_las(
    path: Path | str,
    cell_size: float | tuple[float, float],
    *,
    chunk_size: int = 2_000_000,
    bounds: tuple[float, float, float, float] | None = None,
):
    """Build a RasterGrid from LAS/LAZ without loading the full point cloud."""
    from .io import _require_laspy
    from .raster import RasterGrid

    laspy = _require_laspy()
    source = _validate_file(path, chunk_size)
    size = normalize_cell_size(cell_size, dims=2)
    with laspy.open(source) as reader:
        header = reader.header
        if header.point_count == 0:
            raise ValueError("point cloud must not be empty")
        if bounds is None:
            origin = np.asarray(header.mins[:2], dtype=np.float64)
            maximum = np.asarray(header.maxs[:2], dtype=np.float64)
            shape_xy = np.floor((maximum - origin) / size).astype(np.int64) + 1
        else:
            min_x, min_y, max_x, max_y = bounds
            origin = np.array([min_x, min_y], dtype=np.float64)
            maximum = np.array([max_x, max_y], dtype=np.float64)
            shape_xy = np.ceil((maximum - origin) / size).astype(np.int64)
        if np.any(shape_xy <= 0):
            raise ValueError("Invalid raster bounds or cell_size")
        shape = tuple(map(int, shape_xy))
        accumulator = _StreamingAggregation(shape, int(header.point_count))
        for chunk in reader.chunk_iterator(chunk_size):
            x, y, z = _chunk_arrays(chunk)
            cell_ids, values = _chunk_cell_ids(
                (x, y), z, origin, size, shape, clip_to_bounds=bounds is not None
            )
            accumulator.update(cell_ids, values)
    return RasterGrid(
        cell_size=size,
        origin=origin,
        shape=(shape[1], shape[0]),
        _aggregation=accumulator.finish(),
    )


def voxelize_las(
    path: Path | str,
    voxel_size: float | tuple[float, float, float],
    *,
    chunk_size: int = 2_000_000,
    bounds: tuple[float, float, float, float, float, float] | None = None,
):
    """Build a VoxelGrid from LAS/LAZ without loading the full point cloud."""
    from .io import _require_laspy
    from .voxel import VoxelGrid

    laspy = _require_laspy()
    source = _validate_file(path, chunk_size)
    size = normalize_cell_size(voxel_size, dims=3)
    with laspy.open(source) as reader:
        header = reader.header
        if header.point_count == 0:
            raise ValueError("point cloud must not be empty")
        if bounds is None:
            origin = np.asarray(header.mins[:3], dtype=np.float64)
            maximum = np.asarray(header.maxs[:3], dtype=np.float64)
            shape_xyz = np.floor((maximum - origin) / size).astype(np.int64) + 1
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = bounds
            origin = np.array([min_x, min_y, min_z], dtype=np.float64)
            maximum = np.array([max_x, max_y, max_z], dtype=np.float64)
            shape_xyz = np.ceil((maximum - origin) / size).astype(np.int64)
        if np.any(shape_xyz <= 0):
            raise ValueError("Invalid voxel bounds or voxel_size")
        shape = tuple(map(int, shape_xyz))
        accumulator = _StreamingAggregation(shape, int(header.point_count))
        for chunk in reader.chunk_iterator(chunk_size):
            axes = _chunk_arrays(chunk)
            cell_ids, values = _chunk_cell_ids(
                axes, axes[2], origin, size, shape, clip_to_bounds=bounds is not None
            )
            accumulator.update(cell_ids, values)
    return VoxelGrid(
        voxel_size=size,
        origin=origin,
        shape=(shape[2], shape[1], shape[0]),
        aggregation=accumulator.finish(),
    )
