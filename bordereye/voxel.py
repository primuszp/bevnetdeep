from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._aggregation import (
    SparseAggregation,
    aggregate_sparse,
    grid_cell_count,
    linearize_coords,
    normalize_cell_size,
    use_dense_aggregation,
    validate_points,
)
from ._grid import GridStatsMixin
from .components import GridComponent


@dataclass(frozen=True)
class VoxelGrid(GridStatsMixin):
    """Sparse 3D voxel grid with min, max, mean, and count per voxel."""

    voxel_size: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int, int]
    aggregation: SparseAggregation

    @classmethod
    def from_las(
        cls,
        path: str | Path,
        voxel_size: float | Iterable[float],
        *,
        chunk_size: int = 2_000_000,
        bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> VoxelGrid:
        """Stream a LAS/LAZ file into a sparse voxel grid in bounded memory."""
        from .streaming import voxelize_las

        return voxelize_las(path, voxel_size, chunk_size=chunk_size, bounds=bounds)

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        voxel_size: float | Iterable[float],
        values: np.ndarray | None = None,
        bounds: tuple[float, float, float, float, float, float] | None = None,
    ) -> VoxelGrid:
        pts = validate_points(points, dims=3)
        vs = normalize_cell_size(voxel_size, dims=3)

        if bounds is None:
            min_xyz = pts[:, :3].min(axis=0)
            max_xyz = pts[:, :3].max(axis=0)
            shape_xyz = np.floor((max_xyz - min_xyz) / vs).astype(np.int64) + 1
            mask = None
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = bounds
            min_xyz = np.array([min_x, min_y, min_z], dtype=np.float64)
            max_xyz = np.array([max_x, max_y, max_z], dtype=np.float64)
            shape_xyz = np.ceil((max_xyz - min_xyz) / vs).astype(np.int64)

        if np.any(shape_xyz <= 0):
            raise ValueError("Invalid voxel bounds or voxel_size")

        coord_dtype = (
            np.int32
            if bounds is None and shape_xyz.max() <= np.iinfo(np.int32).max
            else np.int64
        )
        coords = np.floor((pts[:, :3] - min_xyz) / vs).astype(coord_dtype)
        if bounds is not None:
            mask = np.all((coords >= 0) & (coords < shape_xyz), axis=1)
        if values is None:
            stat_values = pts[:, 2]
        else:
            stat_values = np.asarray(values, dtype=np.float64)
            if stat_values.ndim != 1 or len(stat_values) != len(pts):
                raise ValueError("values must be a 1D array with one value per point")

        if mask is not None:
            coords = coords[mask]
            stat_values = stat_values[mask]
        agg = aggregate_sparse(coords, stat_values, tuple(shape_xyz))
        return cls(
            voxel_size=vs,
            origin=min_xyz,
            shape=(int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0])),
            aggregation=agg,
        )

    @property
    def ndim(self) -> int:
        return 3

    # Narrow the mixin's variadic return types back to the 3-D contract, so
    # callers can still unpack them into named coordinates.
    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """World extent as ``(min_x, min_y, min_z, max_x, max_y, max_z)``."""
        return super().bounds  # type: ignore[return-value]

    def component_world_bounds(
        self, component: GridComponent
    ) -> tuple[float, float, float, float, float, float]:
        """Component extent as ``(min_x, min_y, min_z, max_x, max_y, max_z)``."""
        return super().component_world_bounds(component)  # type: ignore[return-value]

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        """Grid shape in coordinate order: (x, y, z)."""
        return (self.shape[2], self.shape[1], self.shape[0])

    @property
    def cell_size(self) -> np.ndarray:
        """Alias matching RasterGrid.cell_size."""
        return self.voxel_size


    def index(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be an Nx3+ array")
        return np.floor((pts[:, :3] - self.origin) / self.voxel_size).astype(np.int64)

    def to_dense(self, stat: str = "mean", fill_value: float | int | None = None) -> np.ndarray:
        if stat == "count":
            fill = 0 if fill_value is None else fill_value
            dense = np.full(self.shape, fill, dtype=np.int64)
        else:
            fill = np.nan if fill_value is None else fill_value
            dense = np.full(self.shape, fill, dtype=np.float64)

        if len(self.aggregation.coords) == 0:
            return dense

        linear = self.aggregation.linear_index
        if linear is not None:
            dense.ravel()[linear] = self.aggregation.values(stat)
        else:
            coords = self.aggregation.coords
            dense[coords[:, 2], coords[:, 1], coords[:, 0]] = self.aggregation.values(stat)
        return dense

    def occupancy(self) -> np.ndarray:
        occupied = np.zeros(self.shape, dtype=bool)
        coords = self.aggregation.coords
        if len(coords):
            occupied[coords[:, 2], coords[:, 1], coords[:, 0]] = True
        return occupied


    def sample(self, points: np.ndarray, stat: str = "mean") -> np.ndarray:
        coords = self.index(points)
        values = self.aggregation.values(stat)
        out_dtype = np.float64 if stat != "count" else np.int64
        missing = np.nan if stat != "count" else 0
        out = np.full(len(coords), missing, dtype=out_dtype)
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < self.shape[2]) &
            (coords[:, 1] >= 0) & (coords[:, 1] < self.shape[1]) &
            (coords[:, 2] >= 0) & (coords[:, 2] < self.shape[0])
        )
        if not np.any(valid) or len(self.aggregation.coords) == 0:
            return out
        valid_indices = np.flatnonzero(valid)
        all_valid = bool(np.all(valid))
        valid_coords = coords if all_valid else coords[valid]
        lookup = self._sample_lookup
        if lookup is None:
            agg_idx = self.aggregation.indices_for_coords(valid_coords, self.grid_shape)
        else:
            linear = (
                valid_coords[:, 0]
                + self.shape[2] * (valid_coords[:, 1] + self.shape[1] * valid_coords[:, 2])
            )
            agg_idx = lookup[linear]
        hit = agg_idx >= 0
        if all_valid and np.all(hit):
            return values[agg_idx]
        out[valid_indices[hit]] = values[agg_idx[hit]]
        return out

    def voxel_centers(self, stat: str = "mean") -> np.ndarray:
        coords = self.aggregation.coords
        if len(coords) == 0:
            return np.empty((0, 4), dtype=np.float64)
        xyz = self.origin + (coords.astype(np.float64) + 0.5) * self.voxel_size
        values = self.aggregation.values(stat).astype(np.float64)
        return np.column_stack((xyz, values))


def voxelize_points(
    points: np.ndarray,
    voxel_size: float | Iterable[float],
    *,
    values: np.ndarray | None = None,
    bounds: tuple[float, float, float, float, float, float] | None = None,
) -> VoxelGrid:
    return VoxelGrid.from_point_cloud(
        points=points,
        voxel_size=voxel_size,
        values=values,
        bounds=bounds,
    )


def voxel_downsample(
    points: np.ndarray,
    voxel_size: float | Iterable[float],
    *,
    return_counts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Downsample a point cloud to one centroid per occupied voxel.

    Extra point columns are averaged together with XYZ, which preserves common
    continuous attributes such as intensity. Use ``return_counts`` to obtain
    the number of source points represented by each centroid.
    """
    pts = validate_points(points, dims=3)
    vs = normalize_cell_size(voxel_size, dims=3)
    origin = pts[:, :3].min(axis=0)
    coords = np.floor((pts[:, :3] - origin) / vs).astype(np.int64)
    shape = tuple((coords.max(axis=0) + 1).tolist())
    linear = linearize_coords(coords, shape)
    grid_size = grid_cell_count(shape)
    if use_dense_aggregation(grid_size, len(pts)):
        all_counts = np.bincount(linear, minlength=grid_size)
        occupied = np.flatnonzero(all_counts)
        counts = all_counts[occupied]
        centroids = np.empty((len(occupied), pts.shape[1]), dtype=np.float64)
        # bincount runs in linear time and is much faster than sorting for a
        # compact voxel domain. Process columns one-by-one to cap peak memory.
        for column in range(pts.shape[1]):
            centroids[:, column] = np.bincount(
                linear,
                weights=pts[:, column],
                minlength=grid_size,
            )[occupied]
        centroids /= counts[:, None]
    else:
        order = np.argsort(linear)
        sorted_linear = linear[order]
        starts = np.flatnonzero(np.r_[True, sorted_linear[1:] != sorted_linear[:-1]])
        counts = np.diff(np.r_[starts, len(order)]).astype(np.int64, copy=False)
        centroids = np.add.reduceat(pts[order], starts, axis=0) / counts[:, None]
    if return_counts:
        return centroids, counts
    return centroids
