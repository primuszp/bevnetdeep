"""Raster post-processing operations: filtering, interpolation, crop, merge.

Note: every operation in this module returns a raster rebuilt through
``RasterGrid.from_dense``, which collapses the per-cell statistics: in the
result ``count`` is 1 and ``min == max == mean`` for every occupied cell.
Query the *input* raster if you need the original point statistics.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.ndimage as ndi

from .raster import RasterGrid


def pit_filter(raster: RasterGrid, kernel_size: int = 3) -> RasterGrid:
    """Remove single-cell pit/spike artefacts with a median filter.

    Replaces each cell's minimum-height value with the local median.
    Only occupied cells are updated; empty cells remain empty.

    Args:
        raster: Input height raster (typically a DTM built from ground points).
        kernel_size: Neighbourhood size (odd integer, default 3 = 3×3 window).
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")

    dense_min = raster.to_dense("min")
    valid = ~np.isnan(dense_min)
    if not np.any(valid):
        return raster

    # Nearest valid values prevent both NoData bleeding and the artificial
    # zero border introduced by scipy.signal.medfilt2d's fixed padding.
    if valid.all():
        filled = dense_min
    else:
        _, nearest = ndi.distance_transform_edt(~valid, return_indices=True)
        filled = np.where(valid, dense_min, dense_min[nearest[0], nearest[1]])
    filtered = ndi.median_filter(filled, size=kernel_size, mode="nearest")
    filtered[~valid] = np.nan  # restore empty cells

    return RasterGrid.from_dense(
        filtered, cell_size=raster.cell_size, origin=raster.origin, lazy=True
    )


def fill_nodata(
    raster: RasterGrid,
    *,
    stat: str = "mean",
    max_search_radius: float | None = None,
) -> RasterGrid:
    """Fill NaN/empty cells using Delaunay triangulation interpolation.

    Slow but accurate: uses scipy LinearNDInterpolator (linear interpolation
    between the triangulated occupied cell centres).  Cells outside the
    convex hull of occupied cells remain NaN.

    For a fast alternative see fill_nodata_nn().

    Args:
        raster: Input raster with potential empty cells.
        stat: Which statistic to interpolate ("min", "max", "mean").
        max_search_radius: If set, only fills holes whose distance to the
            nearest occupied cell is <= this value in world units.
    """
    from scipy.interpolate import LinearNDInterpolator

    dense = raster.to_dense(stat)
    valid_mask = ~np.isnan(dense)

    if max_search_radius is not None and max_search_radius < 0:
        raise ValueError("max_search_radius must be >= 0 or None")

    if valid_mask.all():
        return raster

    rows_known, cols_known = np.where(valid_mask)
    values_known = dense[rows_known, cols_known]

    # Linear triangulation needs at least three non-collinear samples.  Fall
    # back to the defined nearest-neighbour behaviour for sparse edge cases.
    known_xy = np.column_stack(
        (cols_known * raster.cell_size[0], rows_known * raster.cell_size[1])
    )
    if len(known_xy) < 3 or np.linalg.matrix_rank(known_xy - known_xy[0]) < 2:
        return fill_nodata_nn(raster, stat=stat)

    rows_fill, cols_fill = np.where(~valid_mask)

    interpolator = LinearNDInterpolator(
        known_xy,
        values_known,
    )
    fill_xy = np.column_stack(
        (cols_fill * raster.cell_size[0], rows_fill * raster.cell_size[1])
    )
    filled_values = interpolator(fill_xy)

    fillable = np.isfinite(filled_values)
    if max_search_radius is not None:
        distances = ndi.distance_transform_edt(
            ~valid_mask,
            sampling=(float(raster.cell_size[1]), float(raster.cell_size[0])),
        )
        fillable &= distances[rows_fill, cols_fill] <= max_search_radius

    result = dense.copy()
    result[rows_fill[fillable], cols_fill[fillable]] = filled_values[fillable]
    return RasterGrid.from_dense(
        result, cell_size=raster.cell_size, origin=raster.origin, lazy=True
    )


def fill_nodata_nn(
    raster: RasterGrid,
    *,
    stat: str = "mean",
) -> RasterGrid:
    """Fill NaN/empty cells with the value of the nearest occupied cell.

    Uses scipy distance_transform_edt for O(n) nearest-neighbour lookup.
    Much faster than fill_nodata() (Delaunay interpolation) and sufficient
    for terrain-following use cases such as Height-Above-Ground computation.

    Args:
        raster: Input raster with potential empty cells.
        stat:   Which statistic to propagate ("min", "max", "mean").
    """
    dense = raster.to_dense(stat)
    valid = ~np.isnan(dense)
    if valid.all():
        return raster
    if not np.any(valid):
        return raster
    _, nearest_idx = ndi.distance_transform_edt(~valid, return_indices=True)
    filled = np.where(valid, dense, dense[nearest_idx[0], nearest_idx[1]])
    return RasterGrid.from_dense(
        filled, cell_size=raster.cell_size, origin=raster.origin, lazy=True
    )


def crop_to_bounds(
    raster: RasterGrid,
    bounds: tuple[float, float, float, float],
) -> RasterGrid:
    """Crop a raster to a world-coordinate bounding box.

    The output origin snaps to the nearest cell boundary of the input raster.
    Cells outside the bounds are dropped; no padding is added.

    Args:
        bounds: (min_x, min_y, max_x, max_y) in world coordinates.
    """
    min_x, min_y, max_x, max_y = bounds
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Invalid bounds: min must be < max")

    # Clip to raster extent
    rb = raster.bounds
    min_x = max(min_x, rb[0])
    min_y = max(min_y, rb[1])
    max_x = min(max_x, rb[2])
    max_y = min(max_y, rb[3])

    if min_x >= max_x or min_y >= max_y:
        raise ValueError("Crop bounds do not overlap with raster extent")

    # Which grid columns/rows fall inside?
    cs = raster.cell_size
    col_min = int(np.floor((min_x - raster.origin[0]) / cs[0]))
    col_max = int(np.ceil((max_x - raster.origin[0]) / cs[0]))
    row_min_world = int(np.floor((min_y - raster.origin[1]) / cs[1]))
    row_max_world = int(np.ceil((max_y - raster.origin[1]) / cs[1]))

    col_min = max(col_min, 0)
    col_max = min(col_max, raster.width)
    row_min_world = max(row_min_world, 0)
    row_max_world = min(row_max_world, raster.height)

    # image rows run top-down; world rows run bottom-up
    img_row_min = raster.height - row_max_world
    img_row_max = raster.height - row_min_world

    dense = raster.to_dense("mean")
    cropped = dense[img_row_min:img_row_max, col_min:col_max]
    new_origin = np.array([
        raster.origin[0] + col_min * cs[0],
        raster.origin[1] + row_min_world * cs[1],
    ], dtype=np.float64)

    return RasterGrid.from_dense(
        cropped, cell_size=raster.cell_size, origin=new_origin, lazy=True
    )


def merge_rasters(rasters: Iterable[RasterGrid], *, prefer: str = "mean") -> RasterGrid:
    """Merge multiple adjacent or overlapping rasters into one.

    All rasters must share the same cell_size. Overlapping cells are resolved
    using the 'prefer' statistic (first non-NaN value in raster order, or mean
    if prefer="mean_overlap").

    Args:
        rasters: Iterable of RasterGrid objects with identical cell_size.
        prefer: "first" keeps the first raster's value where tiles overlap.
                "last"  keeps the last raster's value.
                "mean"  averages overlapping values (default).
    """
    raster_list = list(rasters)
    if prefer not in {"mean", "first", "last"}:
        raise ValueError("prefer must be 'mean', 'first', or 'last'")
    if not raster_list:
        raise ValueError("No rasters to merge")
    if len(raster_list) == 1:
        return raster_list[0]

    cs = raster_list[0].cell_size
    for r in raster_list[1:]:
        if not np.allclose(r.cell_size, cs, rtol=1e-6):
            raise ValueError("All rasters must have the same cell_size")

    # Origins must sit on a common cell grid, otherwise the integer pixel
    # offsets below would silently shift tiles by up to half a cell.
    ref_origin = raster_list[0].origin
    for r in raster_list[1:]:
        frac = (r.origin - ref_origin) / cs
        if not np.allclose(frac, np.round(frac), atol=1e-6):
            raise ValueError(
                "Raster origins are not aligned to a common cell grid "
                f"(offset {frac} cells); resample before merging"
            )

    # Compute the combined world extent
    all_bounds = [r.bounds for r in raster_list]
    global_min_x = min(b[0] for b in all_bounds)
    global_min_y = min(b[1] for b in all_bounds)
    global_max_x = max(b[2] for b in all_bounds)
    global_max_y = max(b[3] for b in all_bounds)

    global_origin = np.array([global_min_x, global_min_y], dtype=np.float64)
    total_cols = int(np.ceil((global_max_x - global_min_x) / cs[0]))
    total_rows = int(np.ceil((global_max_y - global_min_y) / cs[1]))

    if prefer == "mean":
        accum = np.zeros((total_rows, total_cols), dtype=np.float64)
        weight = np.zeros((total_rows, total_cols), dtype=np.int64)
    else:
        canvas = np.full((total_rows, total_cols), np.nan, dtype=np.float64)

    for r in raster_list:
        dense = r.to_dense("mean")  # (H, W)
        # Offset of this tile in global canvas (world → pixel)
        col_off = int(np.round((r.origin[0] - global_min_x) / cs[0]))
        row_off_world = int(np.round((r.origin[1] - global_min_y) / cs[1]))
        # canvas rows are flipped: row 0 = max Y
        img_row_off = total_rows - row_off_world - r.height

        valid = ~np.isnan(dense)
        rr = np.where(valid)
        tile_rows = rr[0] + img_row_off
        tile_cols = rr[1] + col_off

        # Clamp to canvas bounds
        in_bounds = (
            (tile_rows >= 0) & (tile_rows < total_rows) &
            (tile_cols >= 0) & (tile_cols < total_cols)
        )
        tr = tile_rows[in_bounds]
        tc = tile_cols[in_bounds]
        vals = dense[rr[0][in_bounds], rr[1][in_bounds]]

        if prefer == "mean":
            # Flattened bincount instead of np.add.at on a 2-D index pair.
            flat = tr.astype(np.int64) * total_cols + tc
            accum += np.bincount(flat, weights=vals, minlength=accum.size).reshape(accum.shape)
            weight += np.bincount(flat, minlength=weight.size).reshape(weight.shape)
        elif prefer == "last":
            canvas[tr, tc] = vals
        else:  # "first"
            empty = np.isnan(canvas[tr, tc])
            canvas[tr[empty], tc[empty]] = vals[empty]

    if prefer == "mean":
        canvas = np.full_like(accum, np.nan)
        np.divide(accum, weight, out=canvas, where=weight > 0)

    return RasterGrid.from_dense(canvas, cell_size=cs, origin=global_origin, lazy=True)
