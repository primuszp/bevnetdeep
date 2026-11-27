from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._aggregation import (
    SparseAggregation,
    aggregate_sparse,
    normalize_cell_size,
    validate_points,
)
from ._grid import GridStatsMixin
from .components import GridComponent


@dataclass(frozen=True)
class RasterGrid(GridStatsMixin):
    """2.5D raster with per-cell min, max, mean, and count statistics.

    Coordinates are stored sparsely as unflipped (col, row) integer cells.
    Dense arrays returned by to_dense() use image convention: row 0 is max Y.
    """

    cell_size: np.ndarray
    origin: np.ndarray
    shape: tuple[int, int]
    _aggregation: SparseAggregation | None

    @property
    def aggregation(self) -> SparseAggregation:
        aggregation = self._aggregation
        if aggregation is None:
            dense = self.__dict__.get("mean")
            if dense is None:
                raise ValueError("Lazy raster has no dense source")
            eager = type(self).from_dense(
                dense,
                cell_size=self.cell_size,
                origin=self.origin,
                lazy=False,
            )
            aggregation = eager.aggregation
            object.__setattr__(self, "_aggregation", aggregation)
        return aggregation

    @classmethod
    def from_las(
        cls,
        path: str | Path,
        cell_size: float | Iterable[float],
        *,
        chunk_size: int = 2_000_000,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> RasterGrid:
        """Stream a LAS/LAZ file into a raster with bounded peak memory."""
        from .streaming import rasterize_las

        return rasterize_las(path, cell_size, chunk_size=chunk_size, bounds=bounds)

    @classmethod
    def from_dense(
        cls,
        data: np.ndarray,
        cell_size: float | Iterable[float],
        origin: tuple[float, float] | np.ndarray,
        *,
        lazy: bool = False,
    ) -> RasterGrid:
        """Build a raster from a dense 2D array (image convention, row 0 = max Y).

        The dense array is the single source value per cell, so the per-cell
        statistics collapse: ``count`` is 1 and ``min == max == mean`` for every
        occupied (non-NaN) cell. With ``lazy=True`` the sparse aggregation is
        only materialised on first access.
        """
        arr = np.asarray(data) if lazy else np.asarray(data, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError("data must be a 2D array")
        cs = normalize_cell_size(cell_size, dims=2)
        org = np.asarray(origin, dtype=np.float64)
        if org.shape != (2,):
            raise ValueError("origin must be a length-2 coordinate")

        agg = None
        if not lazy:
            rows, cols = np.where(~np.isnan(arr))
            unflipped_rows = arr.shape[0] - 1 - rows
            coords = np.column_stack((cols, unflipped_rows)).astype(np.int64)
            values = arr[rows, cols]
            linear = cols.astype(np.int64) + unflipped_rows.astype(np.int64) * arr.shape[1]
            order = np.argsort(linear)
            sorted_values = values[order]
            agg = SparseAggregation(
                coords=coords[order],
                count=np.ones(len(coords), dtype=np.int64),
                minimum=sorted_values,
                maximum=sorted_values,
                mean=sorted_values,
                linear_index=linear[order],
            )
        raster = cls(
            cell_size=cs,
            origin=org,
            shape=(int(arr.shape[0]), int(arr.shape[1])),
            _aggregation=agg,
        )
        raster.__dict__["mean"] = arr
        return raster

    @classmethod
    def from_point_cloud(
        cls,
        points: np.ndarray,
        cell_size: float | Iterable[float],
        values: np.ndarray | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> RasterGrid:
        pts = validate_points(points, dims=3)
        cs = normalize_cell_size(cell_size, dims=2)

        mask = None
        if bounds is None:
            min_xy = pts[:, :2].min(axis=0)
            max_xy = pts[:, :2].max(axis=0)
            shape_xy = np.floor((max_xy - min_xy) / cs).astype(np.int64) + 1
        else:
            min_x, min_y, max_x, max_y = bounds
            min_xy = np.array([min_x, min_y], dtype=np.float64)
            max_xy = np.array([max_x, max_y], dtype=np.float64)
            shape_xy = np.ceil((max_xy - min_xy) / cs).astype(np.int64)

        if np.any(shape_xy <= 0):
            raise ValueError("Invalid raster bounds or cell_size")

        coord_dtype = (
            np.int32
            if bounds is None and shape_xy.max() <= np.iinfo(np.int32).max
            else np.int64
        )
        coords = np.floor((pts[:, :2] - min_xy) / cs).astype(coord_dtype)
        if bounds is not None:
            mask = np.all((coords >= 0) & (coords < shape_xy), axis=1)
        if values is None:
            stat_values = pts[:, 2]
        else:
            stat_values = np.asarray(values, dtype=np.float64)
            if stat_values.ndim != 1 or len(stat_values) != len(pts):
                raise ValueError("values must be a 1D array with one value per point")

        if mask is not None:
            coords = coords[mask]
            stat_values = stat_values[mask]
        agg = aggregate_sparse(coords, stat_values, tuple(shape_xy))
        return cls(
            cell_size=cs,
            origin=min_xy,
            shape=(int(shape_xy[1]), int(shape_xy[0])),
            _aggregation=agg,
        )

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def ndim(self) -> int:
        return 2

    # Narrow the mixin's variadic return types back to the 2-D contract, so
    # callers can still unpack them into named coordinates.
    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """World extent as ``(min_x, min_y, max_x, max_y)``."""
        return super().bounds  # type: ignore[return-value]

    def component_world_bounds(self, component: GridComponent) -> tuple[float, float, float, float]:
        """Component extent as ``(min_x, min_y, max_x, max_y)``."""
        return super().component_world_bounds(component)  # type: ignore[return-value]

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Grid shape in coordinate order: (x, y)."""
        return (self.width, self.height)


    def index(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must be an Nx2+ array")
        dtype = np.int32 if max(self.grid_shape) <= np.iinfo(np.int32).max else np.int64
        # Divide rather than multiply by the reciprocal: 1/cell_size is inexact
        # for most cell sizes, so the reciprocal form floors points that sit on
        # a cell boundary into a different cell than from_point_cloud does.
        return np.floor((pts[:, :2] - self.origin) / self.cell_size).astype(dtype)

    def row_col(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coords = self.index(points)
        cols = coords[:, 0]
        rows = self.height - 1 - coords[:, 1]
        return rows, cols

    def to_dense(self, stat: str = "mean", fill_value: float | int | None = None) -> np.ndarray:
        lazy_dense = self.__dict__.get("mean") if self._aggregation is None else None
        if lazy_dense is not None:
            if stat not in {"min", "max", "mean", "count"}:
                raise ValueError(f"Unknown statistic: {stat}")
            valid = np.isfinite(lazy_dense)
            if stat == "count":
                fill = 0 if fill_value is None else fill_value
                return np.where(valid, 1, fill).astype(np.int64, copy=False)
            fill = np.nan if fill_value is None else fill_value
            return np.where(valid, lazy_dense, fill).astype(np.float64, copy=False)

        if stat == "count":
            fill = 0 if fill_value is None else fill_value
            dense = np.full(self.shape, fill, dtype=np.int64)
        else:
            fill = np.nan if fill_value is None else fill_value
            dense = np.full(self.shape, fill, dtype=np.float64)

        if len(self.aggregation.coords) == 0:
            return dense

        cols = self.aggregation.coords[:, 0]
        unflipped_rows = self.aggregation.coords[:, 1]
        rows = self.height - 1 - unflipped_rows
        dense[rows, cols] = self.aggregation.values(stat)
        return dense

    def occupancy(self) -> np.ndarray:
        occupied = np.zeros(self.shape, dtype=bool)
        coords = self.aggregation.coords
        if len(coords):
            occupied[self.height - 1 - coords[:, 1], coords[:, 0]] = True
        return occupied


    def sample(self, points: np.ndarray, stat: str = "mean") -> np.ndarray:
        lazy_dense = self.__dict__.get("mean") if self._aggregation is None else None
        if lazy_dense is not None:
            if stat not in {"min", "max", "mean", "count"}:
                raise ValueError(f"Unknown statistic: {stat}")
            coords = self.index(points)
            valid = (
                (coords[:, 0] >= 0) & (coords[:, 0] < self.width) &
                (coords[:, 1] >= 0) & (coords[:, 1] < self.height)
            )
            if stat == "count":
                out = np.zeros(len(coords), dtype=np.int64)
            else:
                out = np.full(len(coords), np.nan, dtype=np.float64)
            if np.any(valid):
                selected = coords[valid]
                sampled = lazy_dense[self.height - 1 - selected[:, 1], selected[:, 0]]
                if stat == "count":
                    out[valid] = np.isfinite(sampled)
                else:
                    out[valid] = sampled
            return out

        values = self.aggregation.values(stat)
        out_dtype = np.float64 if stat != "count" else np.int64
        missing = np.nan if stat != "count" else 0
        coords = self.index(points)
        out = np.full(len(coords), missing, dtype=out_dtype)
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < self.width) &
            (coords[:, 1] >= 0) & (coords[:, 1] < self.height)
        )
        if not np.any(valid) or len(self.aggregation.coords) == 0:
            return out
        all_valid = bool(np.all(valid))
        valid_coords = coords if all_valid else coords[valid]
        lookup = self._sample_lookup
        if lookup is not None:
            agg_idx = lookup[valid_coords[:, 0] + self.width * valid_coords[:, 1]]
        else:
            agg_idx = self.aggregation.indices_for_coords(valid_coords, self.grid_shape)
        hit = agg_idx >= 0
        if all_valid and np.all(hit):
            return values[agg_idx]
        out[np.flatnonzero(valid)[hit]] = values[agg_idx[hit]]
        return out

    def cell_centers(self, stat: str = "mean") -> np.ndarray:
        coords = self.aggregation.coords
        if len(coords) == 0:
            return np.empty((0, 3), dtype=np.float64)
        xy = self.origin + (coords[:, :2].astype(np.float64) + 0.5) * self.cell_size
        z = self.aggregation.values(stat).astype(np.float64)
        return np.column_stack((xy, z))
