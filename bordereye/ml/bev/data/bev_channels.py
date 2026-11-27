"""Canonical BEV feature construction.

This module is the single source of truth for stored channel names and
deterministic raster-derived features. The schema is model/dataset-version
specific and therefore lives inside :mod:`bordereye.ml.bev`.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from bordereye import RasterGrid

CHANNEL_BUILDER_VERSION = "bev_channels_2026_08_r2"
WIDTH_HOLE_VERSION = "component_gate_2026_07_r1"

GENERIC_V2_CHANNEL_NAMES = (
    "ground_occ", "ground_shadow", "ground_edt", "shadow_edt", "dtm_slope",
    "veh_h_p05", "veh_h_p50", "veh_h_p95", "veh_h_span", "veh_density",
    "veh_dog", "veh_sobel", "veh_h_p95_hillshade_multi",
    "veh_h_p95_aargau_like", "all_h_p95", "all_sobel", "all_density",
    "veh_width_hole",
)

GENERIC_V2_MODEL_CHANNELS_12 = (
    "ground_occ", "ground_shadow", "ground_edt", "shadow_edt",
    "veh_width_hole", "veh_h_p05", "veh_h_p95", "veh_h_span",
    "veh_sobel", "veh_h_p95_hillshade_multi", "dtm_slope", "all_density",
)

CHANNEL_DESCRIPTIONS = {
    "ground_occ": "ground-return occupancy",
    "ground_shadow": "smoothed ground-occlusion holes",
    "ground_edt": "distance inside observed ground",
    "shadow_edt": "distance inside ground-shadow regions",
    "dtm_slope": "local terrain slope",
    "veh_h_p05": "5th percentile vehicle-band height above ground",
    "veh_h_p50": "median vehicle-band height above ground",
    "veh_h_p95": "95th percentile vehicle-band height above ground",
    "veh_h_span": "vehicle-band p95 minus p05",
    "veh_density": "vehicle-band point density",
    "veh_dog": "difference-of-Gaussians vehicle relief",
    "veh_sobel": "vehicle-surface gradient magnitude",
    "veh_h_p95_hillshade_multi": "multi-azimuth shaded vehicle relief",
    "veh_h_p95_aargau_like": "multi-scale local relief",
    "all_h_p95": "95th percentile height of all non-ground returns",
    "all_sobel": "all-return height gradient magnitude",
    "all_density": "all-return point density",
    "veh_width_hole": "vehicle-width gated ground-occlusion response",
}


def robust01(values: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Robustly scale a raster to ``[0, 1]`` while preserving zero background."""
    arr = np.asarray(values, np.float32)
    finite = np.isfinite(arr)
    sample = arr[finite & (arr != 0)]
    if sample.size == 0:
        return np.zeros_like(arr, np.float32)
    a, b = np.percentile(sample, [lo, hi])
    if b <= a + 1e-8:
        return np.zeros_like(arr, np.float32)
    return np.where(finite, np.clip((arr - a) / (b - a), 0, 1), 0).astype(np.float32)


def _smooth_surface(surface: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    s = np.nan_to_num(np.asarray(surface, np.float32), nan=0.0)
    return ndi.gaussian_filter(s, sigma=sigma).astype(np.float32)


def surface_sobel(surface: np.ndarray) -> np.ndarray:
    s = _smooth_surface(surface, 0.7)
    return np.hypot(ndi.sobel(s, axis=0), ndi.sobel(s, axis=1)).astype(np.float32)


def surface_dog(surface: np.ndarray, cell: float = 0.25) -> np.ndarray:
    s = np.nan_to_num(surface, nan=0.0).astype(np.float32)
    near = max(0.6, 0.25 / float(cell))
    far = max(2.4, 1.0 / float(cell))
    return (ndi.gaussian_filter(s, near) - ndi.gaussian_filter(s, far)).astype(np.float32)


def surface_local_relief(surface: np.ndarray, cell: float = 0.25,
                         radius_m: float = 3.0) -> np.ndarray:
    s = _smooth_surface(surface, 0.7)
    sigma = max(radius_m / max(float(cell), 1e-6) / 3.0, 1.0)
    return (s - ndi.gaussian_filter(s, sigma)).astype(np.float32)


def surface_hillshade(surface: np.ndarray, azimuth_deg: float = 315.0,
                      altitude_deg: float = 45.0) -> np.ndarray:
    s = _smooth_surface(surface, 0.8)
    gy, gx = np.gradient(s)
    return _hillshade_from_gradient(gx, gy, azimuth_deg, altitude_deg)


def _hillshade_from_gradient(
    gx: np.ndarray,
    gy: np.ndarray,
    azimuth_deg: float,
    altitude_deg: float,
) -> np.ndarray:
    """Shade one already-smoothed surface gradient."""
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az = np.deg2rad(azimuth_deg)
    alt = np.deg2rad(altitude_deg)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip((shade + 1.0) * 0.5, 0, 1).astype(np.float32)


def surface_hillshade_multi(surface: np.ndarray, cell: float = 0.25) -> np.ndarray:
    del cell
    # Smoothing and gradient construction dominate this operation. They do not
    # depend on illumination azimuth, so compute them once for all eight views.
    smoothed = _smooth_surface(surface, 0.8)
    gy, gx = np.gradient(smoothed)
    shades = [
        _hillshade_from_gradient(gx, gy, az, 45.0)
        for az in (0, 45, 90, 135, 180, 225, 270, 315)
    ]
    return np.std(np.stack(shades), axis=0).astype(np.float32)


def surface_aargau_like(surface: np.ndarray, cell: float = 0.25) -> np.ndarray:
    fine = np.abs(surface_local_relief(surface, cell, 1.5))
    coarse = np.abs(surface_local_relief(surface, cell, 4.0))
    return robust01(0.65 * fine + 0.35 * coarse)


def shadow_from_ground_occ(ground_occ: np.ndarray, cell: float,
                           mode: str = "auto") -> np.ndarray:
    """Return enclosed, vehicle-scale holes in ground observations."""
    occ = np.asarray(ground_occ) > 0
    if mode not in {"auto", "raw", "smooth"}:
        raise ValueError(f"unknown shadow mode: {mode}")
    fill = float(occ.mean())
    use_smooth = mode == "smooth" or (mode == "auto" and fill < 0.35)
    if use_smooth:
        radius = max(1, round(0.7 / max(float(cell), 1e-6)))
        occ_for_holes = ndi.binary_closing(occ, iterations=radius)
    else:
        occ_for_holes = occ
    holes = ndi.binary_fill_holes(occ_for_holes) & ~occ
    labels, count = ndi.label(holes)
    if count:
        sizes = ndi.sum(holes, labels, range(1, count + 1))
        min_cells = max(4, round(0.8 / (cell * cell)))
        max_cells = max(min_cells + 1, round(120.0 / (cell * cell)))
        keep = np.flatnonzero((sizes >= min_cells) & (sizes <= max_cells)) + 1
        holes = np.isin(labels, keep)
    return holes.astype(np.float32)


def width_gated_holes(shadow: np.ndarray, cell: float,
                      min_width_m: float = 1.0, max_width_m: float = 4.0) -> np.ndarray:
    """Select complete shadow components with vehicle-scale width and area.

    The output is deliberately binary, matching the stored v18 channel
    contract. Component-level gating avoids retaining only the centreline of a
    plausible vehicle hole.
    """
    mask = np.asarray(shadow) > 0
    if not mask.any():
        return np.zeros(mask.shape, np.float32)
    labels, count = ndi.label(mask)
    keep = np.zeros(count + 1, bool)
    for component_id, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            continue
        component = labels[slices] == component_id
        height_m, width_m = (component.shape[0] * cell, component.shape[1] * cell)
        short_m, long_m = sorted((height_m, width_m))
        area_m2 = float(component.sum()) * cell * cell
        keep[component_id] = (
            min_width_m <= short_m <= max_width_m
            and 1.5 <= area_m2 <= 15.0
            and long_m <= 8.0
        )
    return keep[labels].astype(np.float32)


def _coords(xyz: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]):
    col = np.floor((xyz[:, 0] - origin[0]) / cell).astype(np.int64)
    row_up = np.floor((xyz[:, 1] - origin[1]) / cell).astype(np.int64)
    ok = (col >= 0) & (col < shape[1]) & (row_up >= 0) & (row_up < shape[0])
    return shape[0] - 1 - row_up[ok], col[ok], ok


def _cell_percentiles(
    xyz: np.ndarray,
    values: np.ndarray,
    raster: RasterGrid,
    quantiles: tuple[float, ...],
) -> dict[float, np.ndarray]:
    """Compute several linear-interpolated cell percentiles in one sort."""
    requested = tuple(float(q) for q in quantiles)
    if not requested:
        return {}
    if any(q < 0.0 or q > 100.0 for q in requested):
        raise ValueError("quantiles must be in [0, 100]")
    outputs = {q: np.zeros(raster.shape, np.float32) for q in requested}
    if len(xyz) == 0:
        return outputs
    rows, cols, ok = _coords(xyz, raster.origin, float(raster.cell_size[0]), raster.shape)
    vals = np.asarray(values, dtype=np.float64)[ok]
    linear = rows * raster.shape[1] + cols
    # ``linear`` is the primary key and value the secondary key. Thus every
    # cell is contiguous and its samples are already sorted for all requested
    # quantiles; the old implementation called np.percentile once per cell.
    order = np.lexsort((vals, linear))
    linear, vals = linear[order], vals[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear)) + 1]
    counts = np.diff(np.r_[starts, len(linear)])
    cells = linear[starts]
    for q, out in outputs.items():
        rank = (counts - 1) * (q / 100.0)
        lower = np.floor(rank).astype(np.int64)
        upper = np.ceil(rank).astype(np.int64)
        fraction = rank - lower
        low_values = vals[starts + lower]
        high_values = vals[starts + upper]
        out.ravel()[cells] = low_values + (high_values - low_values) * fraction
    return outputs


def _cell_percentile(xyz, values, raster: RasterGrid, q: float) -> np.ndarray:
    """Single-quantile compatibility wrapper around the batched implementation."""
    return _cell_percentiles(xyz, values, raster, (q,))[float(q)]


def _density(xyz, raster: RasterGrid) -> np.ndarray:
    out = np.zeros(raster.shape, np.float32)
    if len(xyz):
        rows, cols, _ = _coords(xyz, raster.origin, float(raster.cell_size[0]), raster.shape)
        # Flattened bincount instead of np.add.at on a 2-D index pair: the
        # latter walks the index arrays element by element.
        counts = np.bincount(rows * out.shape[1] + cols, minlength=out.size)
        out = counts.reshape(out.shape).astype(np.float32)
    return np.log1p(out).astype(np.float32)


def build_generic_v2(
    xyz: np.ndarray,
    hag: np.ndarray,
    cell: float,
    bounds,
    channel_names=None,
):
    """Build only the requested canonical channels and a reference raster.

    ``channel_names=None`` retains the complete stored-dataset contract. During
    inference the checkpoint's ordered channel names should be passed so unused
    percentile rasters and surface operators are never evaluated.
    """
    xyz = np.asarray(xyz, np.float64)
    hag = np.asarray(hag, np.float32)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(hag) != len(xyz):
        raise ValueError("xyz must be (N,>=3) and hag must contain N values")
    requested = tuple(GENERIC_V2_CHANNEL_NAMES if channel_names is None else channel_names)
    if not requested:
        raise ValueError("at least one BEV channel must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError(f"duplicate BEV channel names: {list(requested)}")
    unknown = set(requested) - set(GENERIC_V2_CHANNEL_NAMES)
    if unknown:
        raise ValueError(f"unknown BEV channel names: {sorted(unknown)}")

    raster = RasterGrid.from_point_cloud(xyz, cell_size=cell, bounds=bounds)
    values = {}

    ground_dependents = {
        "ground_occ", "ground_shadow", "ground_edt", "shadow_edt",
        "dtm_slope", "veh_width_hole",
    }
    if set(requested) & ground_dependents:
        ground = np.isfinite(hag) & (hag < 0.30)
        ground_occ = (_density(xyz[ground], raster) > 0).astype(np.float32)
        values["ground_occ"] = ground_occ
        if set(requested) & {"ground_shadow", "shadow_edt", "veh_width_hole"}:
            shadow = shadow_from_ground_occ(ground_occ, cell)
            values["ground_shadow"] = shadow
            if "shadow_edt" in requested:
                values["shadow_edt"] = (
                    ndi.distance_transform_edt(shadow > 0) * cell
                ).astype(np.float32)
            if "veh_width_hole" in requested:
                values["veh_width_hole"] = width_gated_holes(shadow, cell)
        if "ground_edt" in requested:
            values["ground_edt"] = (
                ndi.distance_transform_edt(ground_occ > 0) * cell
            ).astype(np.float32)
        if "dtm_slope" in requested:
            ground_z = _cell_percentile(xyz[ground], xyz[ground, 2], raster, 50)
            filled_z = ndi.distance_transform_edt(
                ground_z == 0, return_distances=False, return_indices=True
            )
            if ground.any():
                ground_z = ground_z[tuple(filled_z)]
            values["dtm_slope"] = (
                surface_sobel(ground_z) / max(8.0 * cell, 1e-6)
            ).astype(np.float32)

    vehicle_dependents = {
        "veh_h_p05", "veh_h_p50", "veh_h_p95", "veh_h_span",
        "veh_density", "veh_dog", "veh_sobel",
        "veh_h_p95_hillshade_multi", "veh_h_p95_aargau_like",
    }
    if set(requested) & vehicle_dependents:
        vehicle = np.isfinite(hag) & (hag >= 0.25) & (hag <= 4.0)
        need_p05 = bool(set(requested) & {"veh_h_p05", "veh_h_span"})
        need_p95 = bool(set(requested) & {
            "veh_h_p95", "veh_h_span", "veh_dog", "veh_sobel",
            "veh_h_p95_hillshade_multi", "veh_h_p95_aargau_like",
        })
        needed_quantiles = tuple(
            q for q, needed in ((5.0, need_p05), (50.0, "veh_h_p50" in requested),
                                (95.0, need_p95)) if needed
        )
        vehicle_percentiles = _cell_percentiles(
            xyz[vehicle], hag[vehicle], raster, needed_quantiles
        )
        if need_p05:
            p05 = vehicle_percentiles[5.0]
            values["veh_h_p05"] = p05
        if "veh_h_p50" in requested:
            values["veh_h_p50"] = vehicle_percentiles[50.0]
        if need_p95:
            p95 = vehicle_percentiles[95.0]
            values["veh_h_p95"] = p95
        if "veh_h_span" in requested:
            values["veh_h_span"] = np.clip(p95 - p05, 0, None).astype(np.float32)
        if "veh_density" in requested:
            values["veh_density"] = _density(xyz[vehicle], raster)
        if "veh_dog" in requested:
            values["veh_dog"] = surface_dog(p95, cell)
        if "veh_sobel" in requested:
            values["veh_sobel"] = surface_sobel(p95)
        if "veh_h_p95_hillshade_multi" in requested:
            values["veh_h_p95_hillshade_multi"] = surface_hillshade_multi(p95, cell)
        if "veh_h_p95_aargau_like" in requested:
            values["veh_h_p95_aargau_like"] = surface_aargau_like(p95, cell)

    if set(requested) & {"all_h_p95", "all_sobel", "all_density"}:
        above = np.isfinite(hag) & (hag >= 0.10)
        if set(requested) & {"all_h_p95", "all_sobel"}:
            all_p95 = _cell_percentile(
                xyz[above], np.clip(hag[above], 0, 30), raster, 95
            )
            values["all_h_p95"] = all_p95
            if "all_sobel" in requested:
                values["all_sobel"] = surface_sobel(all_p95)
        if "all_density" in requested:
            values["all_density"] = _density(xyz[above], raster)

    channels = np.stack([values[name] for name in requested]).astype(np.float32)
    return channels, raster, list(requested)
