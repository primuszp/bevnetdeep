from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import spatial
from scipy.ndimage import find_objects, label

from .raster import RasterGrid


@dataclass(frozen=True)
class RasterHole:
    id: int
    cells: np.ndarray
    hull_pixels: np.ndarray
    hull_world: np.ndarray
    area_cells: int
    hull_area: float


def _pixel_to_world(raster: RasterGrid, pixels: np.ndarray) -> np.ndarray:
    cols = pixels[:, 0]
    rows = pixels[:, 1]
    x = raster.origin[0] + (cols + 0.5) * raster.cell_size[0]
    unflipped_y = raster.height - 1 - rows
    y = raster.origin[1] + (unflipped_y + 0.5) * raster.cell_size[1]
    return np.column_stack((x, y))


def find_raster_holes(
    raster: RasterGrid,
    *,
    stat: str = "mean",
    min_cells: int = 1,
    snap_edge: bool = False,
    exclude_border: bool = False,
) -> list[RasterHole]:
    """Find connected NoData/NaN components in a raster statistic layer.

    Set ``exclude_border=True`` to return only enclosed holes and ignore the
    outside NoData region touching the raster boundary.
    """
    if min_cells <= 0:
        raise ValueError("min_cells must be > 0")

    dense = raster.to_dense(stat)
    hole_mask = np.isnan(dense)
    labels, n_labels = label(hole_mask)
    objects = find_objects(labels)
    holes: list[RasterHole] = []

    for hole_id in range(1, n_labels + 1):
        obj = objects[hole_id - 1]
        if obj is None:
            continue
        local = labels[obj] == hole_id
        local_rows_cols = np.argwhere(local)
        if len(local_rows_cols) == 0:
            continue
        row_offset = obj[0].start
        col_offset = obj[1].start
        rows_cols = local_rows_cols + np.array([row_offset, col_offset])
        if len(rows_cols) < min_cells:
            continue

        rows = rows_cols[:, 0]
        cols = rows_cols[:, 1]
        if exclude_border and (
            np.any(rows == 0) or np.any(rows == raster.height - 1)
            or np.any(cols == 0) or np.any(cols == raster.width - 1)
        ):
            continue
        pixels = np.column_stack((cols, rows)).astype(np.float64)

        if snap_edge:
            offsets = np.array(
                [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                dtype=np.float64,
            )
            hull_input = (pixels[:, None, :] + offsets).reshape(-1, 2)
        else:
            hull_input = pixels

        try:
            if len(hull_input) >= 3:
                hull = spatial.ConvexHull(hull_input)
                hull_pixels = hull_input[hull.vertices]
                hull_area = float(hull.volume * np.prod(raster.cell_size))
            else:
                hull_pixels = hull_input
                hull_area = 0.0
        except spatial.QhullError:
            hull_pixels = hull_input
            hull_area = 0.0

        holes.append(
            RasterHole(
                id=hole_id,
                cells=rows_cols.astype(np.int64),
                hull_pixels=hull_pixels,
                hull_world=_pixel_to_world(raster, hull_pixels),
                area_cells=len(rows_cols),
                hull_area=hull_area,
            )
        )

    return holes
