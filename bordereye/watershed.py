"""Watershed-based blob splitting for merged vehicle detection.

When multiple vehicles are parked side-by-side they form a single connected
blob in the occupancy raster.  This module uses a height-priority watershed on
the HAG (Height-Above-Ground) max-height raster to split such merged blobs back
into individual vehicle-sized components.

Algorithm
---------
1. Build a max-HAG height raster from the off-ground point cloud.
2. Smooth with a small Gaussian to suppress single-point spikes.
3. Find local height maxima (vehicle roof peaks) whose neighbourhood
   distance exceeds *min_peak_distance_m* (prevents duplicate seeds per
   vehicle).
4. Label each local maximum as a unique seed.
5. Run a priority-queue watershed that floods outward from the seeds in
   decreasing height order, stopping at unoccupied cells.
6. Each flooded region becomes a GridComponent.

Only scipy and numpy are required (no scikit-image dependency).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .components import GridComponent
from .raster import RasterGrid

# ── Public API ────────────────────────────────────────────────────────────────

def watershed_blobs(
    hag_pts: np.ndarray,
    cell_size: float,
    *,
    min_peak_distance_m: float = 1.2,
    min_peak_height: float = 0.3,
    smooth_sigma: float = 0.8,
    min_component_size: int = 4,
    bounds: tuple[float, float, float, float] | None = None,
) -> list[GridComponent]:
    """Split merged vehicle blobs using watershed on the HAG height raster.

    Vehicles parked side-by-side produce a single connected component in a
    plain occupancy raster.  This function uses the height structure (vehicle
    roofs are local height maxima separated by lower inter-vehicle gaps) to
    seed a watershed and recover individual vehicles.

    Args:
        hag_pts:             (N, 3) array [x, y, hag_height] of HAG-filtered
                             off-ground points.
        cell_size:           Raster cell size in metres.
        min_peak_distance_m: Minimum distance between two detected vehicle-
                             roof peaks (metres).  Set ~= shortest vehicle
                             dimension; default 1.2 m.
        min_peak_height:     Minimum HAG height to be considered a peak
                             (metres).  Filters ground-level noise.
        smooth_sigma:        Sigma (in cells) for Gaussian smoothing applied
                             before peak detection.  Suppresses single-point
                             spikes without destroying the roof shape.
        min_component_size:  Minimum number of raster cells for a returned
                             component (smaller ones are discarded).
        bounds:              Optional world-coordinate bounding box.

    Returns:
        List of GridComponent objects.  Each component's ``coords`` array has
        shape (M, 2) in (col, unflipped_row) convention — the same convention
        used by ``connected_components_sparse``.
    """
    pts = np.asarray(hag_pts, dtype=np.float64)
    if len(pts) == 0:
        return []

    # ── 1. Build max-HAG height raster ───────────────────────────────────────
    height_raster = RasterGrid.from_point_cloud(pts, cell_size=cell_size, bounds=bounds)
    dense = height_raster.to_dense("max")  # (H, W), NaN = empty
    occupied = ~np.isnan(dense)

    if not occupied.any():
        return []

    # ── 2. Fill NaN with 0 for filtering, smooth ─────────────────────────────
    h = np.where(occupied, dense, 0.0)
    h_smooth = ndi.gaussian_filter(h.astype(np.float32), sigma=smooth_sigma).astype(np.float64)
    h_smooth = np.where(occupied, h_smooth, 0.0)

    # ── 3. Local maxima: cells equal to the neighbourhood max ─────────────────
    peak_radius = max(1, int(np.ceil(min_peak_distance_m / cell_size)))
    peak_size   = 2 * peak_radius + 1
    local_max   = ndi.maximum_filter(h_smooth, size=peak_size)
    is_peak     = occupied & (h_smooth == local_max) & (h_smooth >= min_peak_height)

    # Label each isolated peak group as one seed
    seeds, n_seeds = ndi.label(is_peak)
    if n_seeds == 0:
        # No peaks found — fall back to a single component per connected blob
        return _connected_components_from_mask(occupied, height_raster, min_component_size)

    # ── 4. Watershed via scipy.ndimage.watershed_ift ─────────────────────────
    # watershed_ift works on uint8 image where 0=background; seeds must be
    # positive integers. It minimises the input image, so we invert heights
    # (high HAG → low cost → expands first).
    H, W = dense.shape

    # Build uint8 cost image: invert smoothed heights, scale to [1,255]
    h_pos = h_smooth - h_smooth[occupied].min() + 1e-6
    h_max = h_pos[occupied].max()
    cost = np.zeros((H, W), dtype=np.uint8)
    cost[occupied] = np.clip(
        255 - (h_pos[occupied] / h_max * 254).astype(np.uint8), 1, 255
    )

    # Use 4-connectivity structure for watershed
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labels = ndi.watershed_ift(cost, seeds, structure=struct4)
    labels[~occupied] = 0  # clear background

    # ── 5. Convert labelled regions to GridComponent list (vectorised) ────────
    flat_labels = labels.ravel()
    flat_rows   = np.repeat(np.arange(H, dtype=np.int32), W)
    flat_cols   = np.tile(np.arange(W, dtype=np.int32), H)
    occ_flat    = flat_labels > 0

    flat_labels = flat_labels[occ_flat]
    flat_rows   = flat_rows[occ_flat]
    flat_cols   = flat_cols[occ_flat]

    # Sort by label so we can use np.split
    order       = np.argsort(flat_labels, kind="stable")
    flat_labels = flat_labels[order]
    flat_rows   = flat_rows[order]
    flat_cols   = flat_cols[order]

    split_idx   = np.flatnonzero(np.diff(flat_labels)) + 1
    rows_groups = np.split(flat_rows, split_idx)
    cols_groups = np.split(flat_cols, split_idx)

    components: list[GridComponent] = []
    for comp_id, (img_rows, img_cols) in enumerate(
        zip(rows_groups, cols_groups, strict=True)
    ):
        n_cells = len(img_rows)
        if n_cells < min_component_size:
            continue
        urows  = (height_raster.height - 1 - img_rows).astype(np.int32)
        coords = np.column_stack((img_cols, urows))
        components.append(GridComponent(id=comp_id, coords=coords))

    return components


# ── Internal helpers ──────────────────────────────────────────────────────────

def _connected_components_from_mask(
    occupied: np.ndarray,
    raster: RasterGrid,
    min_size: int,
) -> list[GridComponent]:
    """Simple 4-connected labelling fallback when no peaks are found."""
    labelled, _ = ndi.label(occupied, structure=np.ones((3, 3), dtype=int))
    H, W = occupied.shape
    flat = labelled.ravel()
    fr   = np.repeat(np.arange(H, dtype=np.int32), W)
    fc   = np.tile(np.arange(W, dtype=np.int32), H)
    m    = flat > 0
    flat, fr, fc = flat[m], fr[m], fc[m]
    order = np.argsort(flat, kind="stable")
    flat, fr, fc = flat[order], fr[order], fc[order]
    split_idx = np.flatnonzero(np.diff(flat)) + 1
    components = []
    for cid, (ir, ic) in enumerate(
        zip(np.split(fr, split_idx), np.split(fc, split_idx), strict=True)
    ):
        if len(ir) < min_size:
            continue
        urows = (raster.height - 1 - ir).astype(np.int32)
        components.append(GridComponent(id=cid, coords=np.column_stack((ic, urows))))
    return components
