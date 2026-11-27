"""Statistics and geometry shared by :class:`RasterGrid` and :class:`VoxelGrid`.

Both grids store the same sparse per-cell aggregation and differ only in
dimensionality and in how a dense array is laid out (rasters use the image
convention with row 0 at max Y; voxels index in plain C order). Everything that
does not depend on that difference lives here, so the 2-D and 3-D grids cannot
drift apart.

Implementers must provide ``origin``, ``cell_size``, ``grid_shape`` (in
coordinate order), ``aggregation``, and ``to_dense``.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ._aggregation import (
    DENSE_SAMPLE_TO_POINT_RATIO,
    MAX_DENSE_SAMPLE_CELLS,
    dense_row_lookup,
    grid_cell_count,
)
from .components import GridComponent, connected_components_sparse


class GridStatsMixin:
    """Dimension-agnostic accessors over a sparse cell aggregation."""

    @property
    def max_coords(self) -> np.ndarray:
        """Upper world corner of the grid extent."""
        return self.origin + np.asarray(self.grid_shape, dtype=np.float64) * self.cell_size

    @property
    def bounds(self) -> tuple[float, ...]:
        """World extent as ``(*min_corner, *max_corner)``."""
        return tuple(float(v) for v in self.origin) + tuple(
            float(v) for v in self.max_coords
        )

    @property
    def occupied_count(self) -> int:
        return len(self.aggregation.coords)

    @property
    def occupied_coords(self) -> np.ndarray:
        """Occupied cell coordinates as a read-only view (no copy)."""
        coords = self.aggregation.coords.view()
        coords.flags.writeable = False
        return coords

    @cached_property
    def count(self) -> np.ndarray:
        return self.to_dense("count")

    @cached_property
    def mean(self) -> np.ndarray:
        return self.to_dense("mean")

    @cached_property
    def minimum(self) -> np.ndarray:
        return self.to_dense("min")

    @cached_property
    def maximum(self) -> np.ndarray:
        return self.to_dense("max")

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.mean, dtype=dtype)

    def values(self, stat: str = "mean") -> np.ndarray:
        return self.aggregation.values(stat)

    @cached_property
    def _sample_lookup(self) -> np.ndarray | None:
        """Compact cell-to-sparse-row index shared by every sampled statistic."""
        dense_size = grid_cell_count(self.grid_shape)
        if (
            self.aggregation.linear_index is not None
            and dense_size <= MAX_DENSE_SAMPLE_CELLS
            and dense_size <= DENSE_SAMPLE_TO_POINT_RATIO * self.occupied_count
        ):
            return dense_row_lookup(self.aggregation, dense_size)
        return None

    def connected_components(
        self,
        *,
        diagonal: bool = False,
        min_size: int = 1,
    ) -> list[GridComponent]:
        return connected_components_sparse(
            self.aggregation.coords,
            diagonal=diagonal,
            min_size=min_size,
        )

    def component_world_bounds(self, component: GridComponent) -> tuple[float, ...]:
        """World extent of one component as ``(*min_corner, *max_corner)``."""
        low = self.origin + component.min_coord.astype(np.float64) * self.cell_size
        high = self.origin + (component.max_coord.astype(np.float64) + 1.0) * self.cell_size
        return tuple(float(v) for v in low) + tuple(float(v) for v in high)
