from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .components import GridComponent
from .geometry import OrientedBoundingBox2D, convex_hull_2d, oriented_bounding_box_2d
from .raster import RasterGrid
from .voxel import VoxelGrid

GridSource = Literal["raster", "voxel"]


@dataclass(frozen=True)
class GridCandidate:
    id: int
    source: GridSource
    coords: np.ndarray
    world_bounds: tuple[float, ...]
    dimensions: tuple[float, ...]
    cell_count: int
    point_count: int
    value_min: float
    value_max: float
    value_mean: float
    obb_2d: OrientedBoundingBox2D | None = None

    @property
    def length(self) -> float:
        if self.obb_2d is not None:
            return float(self.obb_2d.length)
        return float(max(self.dimensions[:2]))

    @property
    def width(self) -> float:
        if self.obb_2d is not None:
            return float(self.obb_2d.width)
        return float(min(self.dimensions[:2]))

    @property
    def height(self) -> float:
        if self.source == "voxel":
            return float(self.dimensions[2])
        return float(self.value_max - self.value_min)

    @property
    def footprint_area(self) -> float:
        if self.obb_2d is not None:
            return float(self.obb_2d.area)
        return float(self.dimensions[0] * self.dimensions[1])

    @property
    def volume(self) -> float:
        return float(self.footprint_area * max(self.height, 0.0))

    @property
    def rotation_deg(self) -> float:
        return 0.0 if self.obb_2d is None else float(self.obb_2d.rotation_deg)


@dataclass(frozen=True)
class VehicleGeometryLimits:
    min_length: float = 1.2
    max_length: float = 16.0
    min_width: float = 0.6
    max_width: float = 4.5
    min_height: float = 0.3
    max_height: float = 4.5
    min_points: int = 1

    def validate(self) -> None:
        if not (0 < self.min_length <= self.max_length):
            raise ValueError("Invalid length limits")
        if not (0 < self.min_width <= self.max_width):
            raise ValueError("Invalid width limits")
        if not (0 <= self.min_height <= self.max_height):
            raise ValueError("Invalid height limits")
        if self.min_points <= 0:
            raise ValueError("min_points must be > 0")


def _component_stats(grid: RasterGrid | VoxelGrid, component: GridComponent) -> tuple[int, float, float, float]:
    idx = grid.aggregation.indices_for_coords(component.coords, grid.grid_shape)
    idx = idx[idx >= 0]
    counts = grid.aggregation.count[idx]
    values = grid.aggregation.mean[idx]
    point_count = int(counts.sum())
    weighted_mean = float(np.average(values, weights=counts)) if point_count else float("nan")
    return (
        point_count,
        float(grid.aggregation.minimum[idx].min()),
        float(grid.aggregation.maximum[idx].max()),
        weighted_mean,
    )


def _obb_from_grid_coords(
    coords: np.ndarray,
    origin: np.ndarray,
    cell_size: np.ndarray,
) -> OrientedBoundingBox2D | None:
    if len(coords) == 0:
        return None
    # Vectorised: build all 4 corners per cell without Python loop
    c = coords[:, :2].astype(np.float64)
    x0 = origin[0] + c[:, 0] * cell_size[0]
    y0 = origin[1] + c[:, 1] * cell_size[1]
    x1 = x0 + cell_size[0]
    y1 = y0 + cell_size[1]
    corner_points = np.concatenate([
        np.column_stack((x0, y0)),
        np.column_stack((x1, y0)),
        np.column_stack((x1, y1)),
        np.column_stack((x0, y1)),
    ])
    # For large components reduce to convex hull before OBB fitting
    if len(corner_points) > 16:
        corner_points = convex_hull_2d(corner_points)
    return oriented_bounding_box_2d(corner_points)


def candidate_from_raster_component(
    raster: RasterGrid,
    component: GridComponent,
) -> GridCandidate:
    bounds = raster.component_world_bounds(component)
    dims = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    point_count, value_min, value_max, value_mean = _component_stats(raster, component)
    return GridCandidate(
        id=component.id,
        source="raster",
        coords=component.coords,
        world_bounds=bounds,
        dimensions=(float(dims[0]), float(dims[1])),
        cell_count=component.size,
        point_count=point_count,
        value_min=value_min,
        value_max=value_max,
        value_mean=value_mean,
        obb_2d=_obb_from_grid_coords(component.coords, raster.origin, raster.cell_size),
    )


def candidate_from_voxel_component(
    voxel: VoxelGrid,
    component: GridComponent,
) -> GridCandidate:
    bounds = voxel.component_world_bounds(component)
    dims = (bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    point_count, value_min, value_max, value_mean = _component_stats(voxel, component)
    return GridCandidate(
        id=component.id,
        source="voxel",
        coords=component.coords,
        world_bounds=bounds,
        dimensions=(float(dims[0]), float(dims[1]), float(dims[2])),
        cell_count=component.size,
        point_count=point_count,
        value_min=value_min,
        value_max=value_max,
        value_mean=value_mean,
        obb_2d=_obb_from_grid_coords(component.coords, voxel.origin, voxel.voxel_size),
    )


def extract_raster_candidates(
    raster: RasterGrid,
    *,
    diagonal: bool = False,
    min_cells: int = 1,
) -> list[GridCandidate]:
    return [
        candidate_from_raster_component(raster, component)
        for component in raster.connected_components(
            diagonal=diagonal,
            min_size=min_cells,
        )
    ]


def extract_voxel_candidates(
    voxel: VoxelGrid,
    *,
    diagonal: bool = False,
    min_cells: int = 1,
) -> list[GridCandidate]:
    return [
        candidate_from_voxel_component(voxel, component)
        for component in voxel.connected_components(
            diagonal=diagonal,
            min_size=min_cells,
        )
    ]


def is_vehicle_like(
    candidate: GridCandidate,
    limits: VehicleGeometryLimits | None = None,
) -> bool:
    lim = limits or VehicleGeometryLimits()
    lim.validate()
    return (
        lim.min_length <= candidate.length <= lim.max_length
        and lim.min_width <= candidate.width <= lim.max_width
        and lim.min_height <= candidate.height <= lim.max_height
        and candidate.point_count >= lim.min_points
    )


def filter_vehicle_candidates(
    candidates: list[GridCandidate],
    limits: VehicleGeometryLimits | None = None,
) -> list[GridCandidate]:
    return [candidate for candidate in candidates if is_vehicle_like(candidate, limits)]
