from __future__ import annotations

from pathlib import Path

import numpy as np

from .raster import RasterGrid


def _validate_xyz(points: np.ndarray) -> np.ndarray:
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or len(xyz) == 0:
        raise ValueError("points must be a non-empty Nx3+ array")
    if not np.isfinite(xyz[:, :3]).all():
        raise ValueError("point coordinates must be finite")
    return xyz[:, :3]


def _window_radii(max_radius: int) -> list[int]:
    """Exponentially growing pixel radii, always including *max_radius*."""
    radii: list[int] = []
    radius = 1
    while radius < max_radius:
        radii.append(radius)
        radius *= 2
    radii.append(max_radius)
    return radii


def _progressive_surface(
    surface: np.ndarray,
    *,
    radii: list[int],
    cell_size: float,
    slope: float,
    initial_distance: float,
    max_distance: float,
) -> np.ndarray:
    """Fill and filter one dense elevation tile."""
    from scipy import ndimage as ndi

    valid = np.isfinite(surface)
    if not np.any(valid):
        raise ValueError("terrain tile has no occupied cells")
    if not valid.all():
        _, nearest = ndi.distance_transform_edt(~valid, return_indices=True)
        surface = np.where(valid, surface, surface[nearest[0], nearest[1]])

    previous_radius = 0
    for radius in radii:
        size = 2 * radius + 1
        eroded = ndi.minimum_filter(surface, size=size, mode="nearest")
        opened = ndi.maximum_filter(eroded, size=size, mode="nearest")
        distance = initial_distance + slope * (radius - previous_radius) * cell_size
        surface = np.where(surface - opened > min(max_distance, distance), opened, surface)
        previous_radius = radius
    return surface


def progressive_ground_dtm(
    points: np.ndarray,
    cell_size: float = 1.0,
    *,
    max_window_size: float = 20.0,
    slope: float = 0.15,
    initial_distance: float = 0.15,
    max_distance: float = 2.5,
    max_raster_cells: int = 50_000_000,
) -> RasterGrid:
    """Build a bare-earth DTM with a progressive morphological filter.

    Minimum cell elevations form the initial surface. Increasing morphological
    openings remove objects while a distance threshold that grows with window
    radius preserves genuine terrain slopes. Empty cells are filled from their
    nearest occupied neighbour before filtering.

    ``max_window_size`` is the largest object footprint (metres) the filter is
    expected to remove. Use a larger value for buildings and a smaller value
    when preserving abrupt terrain is more important.
    """
    xyz = _validate_xyz(points)
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if max_window_size <= 0:
        raise ValueError("max_window_size must be > 0")
    if slope < 0 or initial_distance < 0 or max_distance < 0:
        raise ValueError("slope and distance thresholds must be >= 0")
    if initial_distance > max_distance:
        raise ValueError("initial_distance must be <= max_distance")
    if max_raster_cells <= 0:
        raise ValueError("max_raster_cells must be > 0")

    raster = RasterGrid.from_point_cloud(xyz, cell_size=cell_size)
    cell_count = raster.width * raster.height
    if cell_count > max_raster_cells:
        raise ValueError(
            f"terrain raster would contain {cell_count:,} cells; increase "
            "cell_size, process tiles, or raise max_raster_cells"
        )

    surface = raster.to_dense("min")
    max_radius = max(1, int(np.ceil(max_window_size / (2.0 * cell_size))))
    surface = _progressive_surface(
        surface,
        radii=_window_radii(max_radius),
        cell_size=cell_size,
        slope=slope,
        initial_distance=initial_distance,
        max_distance=max_distance,
    )

    return RasterGrid.from_dense(
        surface,
        cell_size=raster.cell_size,
        origin=raster.origin,
        lazy=True,
    )


def _streaming_ground_mask(
    path: str | Path,
    cell_size: float,
    *,
    max_window_size: float,
    slope: float,
    initial_distance: float,
    max_distance: float,
    classify_threshold: float,
    below_tolerance: float,
    chunk_size: int,
    tile_size: float,
) -> np.ndarray:
    """Bounded-workspace PMF using overlapping dense tiles over a sparse raster."""
    from .io import _require_laspy
    from .streaming import rasterize_las

    laspy = _require_laspy()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    raster = rasterize_las(path, cell_size, chunk_size=chunk_size)
    coords = raster.aggregation.coords.astype(np.int64, copy=False)
    minimum = raster.aggregation.minimum
    tile_cells = max(1, int(np.ceil(tile_size / cell_size)))
    tile_count_x = (raster.width + tile_cells - 1) // tile_cells
    tile_count_y = (raster.height + tile_cells - 1) // tile_cells
    tile_xy = coords // tile_cells
    tile_ids = tile_xy[:, 0] + tile_count_x * tile_xy[:, 1]
    order = np.argsort(tile_ids)
    sorted_ids = tile_ids[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    groups = {
        int(sorted_ids[start]): order[start:stop]
        for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True)
    }

    radii = _window_radii(max(1, int(np.ceil(max_window_size / (2.0 * cell_size)))))
    # Successive openings compound their receptive fields. This overlap makes
    # tile-core results identical to a dense global filter away from data gaps.
    overlap = sum(radii)
    neighbour_tiles = (overlap + tile_cells - 1) // tile_cells
    layout: dict[int, tuple[int, int, int, int, int]] = {}
    terrain_cells = 0
    for tile_id in groups:
        tx, ty = tile_id % tile_count_x, tile_id // tile_count_x
        x0, y0 = tx * tile_cells, ty * tile_cells
        x1 = min(x0 + tile_cells, raster.width)
        y1 = min(y0 + tile_cells, raster.height)
        layout[tile_id] = (x0, y0, x1, y1, terrain_cells)
        terrain_cells += (x1 - x0) * (y1 - y0)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        terrain = np.memmap(
            Path(temp_dir) / "pmf-terrain.bin",
            mode="w+",
            dtype=np.float32,
            shape=(terrain_cells,),
        )
        for tile_id, (x0, y0, x1, y1, terrain_offset) in layout.items():
            tx, ty = tile_id % tile_count_x, tile_id // tile_count_x
            ex0, ey0 = max(0, x0 - overlap), max(0, y0 - overlap)
            ex1 = min(raster.width, x1 + overlap)
            ey1 = min(raster.height, y1 + overlap)
            expanded = np.full((ey1 - ey0, ex1 - ex0), np.nan, dtype=np.float64)
            for nty in range(max(0, ty - neighbour_tiles), min(tile_count_y, ty + neighbour_tiles + 1)):
                for ntx in range(max(0, tx - neighbour_tiles), min(tile_count_x, tx + neighbour_tiles + 1)):
                    selected = groups.get(ntx + tile_count_x * nty)
                    if selected is None:
                        continue
                    selected = selected[
                        (coords[selected, 0] >= ex0) & (coords[selected, 0] < ex1)
                        & (coords[selected, 1] >= ey0) & (coords[selected, 1] < ey1)
                    ]
                    expanded[coords[selected, 1] - ey0, coords[selected, 0] - ex0] = minimum[selected]
            filtered = _progressive_surface(
                expanded,
                radii=radii,
                cell_size=cell_size,
                slope=slope,
                initial_distance=initial_distance,
                max_distance=max_distance,
            )
            core = filtered[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0]
            terrain[terrain_offset:terrain_offset + core.size] = core.ravel()
        terrain.flush()

        surface = None
        with laspy.open(path) as reader:
            mask = np.empty(reader.header.point_count, dtype=bool)
            offset = 0
            for chunk in reader.chunk_iterator(chunk_size):
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)
                cx = np.floor((x - raster.origin[0]) / cell_size).astype(np.int64)
                cy = np.floor((y - raster.origin[1]) / cell_size).astype(np.int64)
                tx, ty = cx // tile_cells, cy // tile_cells
                ids = tx + tile_count_x * ty
                current = np.zeros(len(x), dtype=bool)
                for tile_id in np.unique(ids):
                    selected = ids == tile_id
                    x0, y0, x1, y1, terrain_offset = layout[int(tile_id)]
                    surface = terrain[
                        terrain_offset:terrain_offset + (x1 - x0) * (y1 - y0)
                    ].reshape(y1 - y0, x1 - x0)
                    ground_z = surface[cy[selected] - y0, cx[selected] - x0]
                    height = z[selected] - ground_z
                    current[selected] = (
                        (height >= -below_tolerance) & (height <= classify_threshold)
                    )
                mask[offset:offset + len(x)] = current
                offset += len(x)
        del surface
        del terrain  # close the Windows mapping before TemporaryDirectory cleanup
    return mask


def ground_mask(
    points: np.ndarray | str | Path,
    cell_size: float = 1.0,
    *,
    max_window_size: float = 20.0,
    slope: float = 0.15,
    initial_distance: float = 0.15,
    max_distance: float = 2.5,
    classify_threshold: float = 0.25,
    below_tolerance: float = 0.10,
    max_raster_cells: int = 50_000_000,
    chunk_size: int = 2_000_000,
    tile_size: float = 250.0,
) -> np.ndarray:
    """Classify an array or stream LAS/LAZ with the internal PMF pipeline."""
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    if max_window_size <= 0:
        raise ValueError("max_window_size must be > 0")
    if slope < 0 or initial_distance < 0 or max_distance < 0:
        raise ValueError("slope and distance thresholds must be >= 0")
    if initial_distance > max_distance:
        raise ValueError("initial_distance must be <= max_distance")
    if classify_threshold < 0 or below_tolerance < 0:
        raise ValueError("classification thresholds must be >= 0")
    if isinstance(points, (str, Path)):
        return _streaming_ground_mask(
            points,
            cell_size,
            max_window_size=max_window_size,
            slope=slope,
            initial_distance=initial_distance,
            max_distance=max_distance,
            classify_threshold=classify_threshold,
            below_tolerance=below_tolerance,
            chunk_size=chunk_size,
            tile_size=tile_size,
        )
    xyz = _validate_xyz(points)
    dtm = progressive_ground_dtm(
        xyz,
        cell_size,
        max_window_size=max_window_size,
        slope=slope,
        initial_distance=initial_distance,
        max_distance=max_distance,
        max_raster_cells=max_raster_cells,
    )
    height = height_above_ground(xyz, dtm)
    return (
        np.isfinite(height)
        & (height >= -below_tolerance)
        & (height <= classify_threshold)
    )


def segment_ground(
    points: np.ndarray,
    cell_size: float = 1.0,
    **kwargs: float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ground_points, ground_mask)`` using the internal PMF filter."""
    source = np.asarray(points)
    mask = ground_mask(source, cell_size, **kwargs)
    return source[mask], mask


def build_dtm(
    ground_points: np.ndarray,
    cell_size: float | tuple[float, float],
    *,
    stat: str = "min",
) -> RasterGrid:
    """Build a terrain surface from one selected per-cell elevation statistic.

    Use ``stat="min"`` for a conservative ground level or ``"mean"`` for a
    smoother surface. The selected statistic becomes the DTM's sample value;
    later consumers cannot silently revert to another statistic.
    """
    if stat not in {"min", "max", "mean"}:
        raise ValueError("stat must be 'min', 'max', or 'mean'")
    raster = RasterGrid.from_point_cloud(ground_points, cell_size=cell_size)
    if stat == "mean":
        return raster
    return RasterGrid.from_dense(
        raster.to_dense(stat),
        cell_size=raster.cell_size,
        origin=raster.origin,
        lazy=True,
    )


def height_above_ground(
    points: np.ndarray,
    dtm: RasterGrid,
    *,
    terrain_stat: str = "mean",
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")
    ground_z = dtm.sample(pts, terrain_stat)
    return pts[:, 2] - ground_z


def filter_by_height_above_ground(
    points: np.ndarray,
    dtm: RasterGrid,
    *,
    min_height: float,
    max_height: float,
    terrain_stat: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    if min_height > max_height:
        raise ValueError("min_height must be <= max_height")
    hag = height_above_ground(points, dtm, terrain_stat=terrain_stat)
    mask = np.isfinite(hag) & (hag >= min_height) & (hag <= max_height)
    return np.asarray(points)[mask], mask


def csf_ground_mask(
    points: np.ndarray,
    csf_exe: str | Path,
    *,
    cloth_resolution: float = 0.5,
    classify_threshold: float = 0.15,
    b_slope_smooth: bool = False,
    rigidness: int | None = None,
    cache: str | Path | None = None,
) -> np.ndarray:
    """Boolean ground mask via the CSF (Cloth Simulation Filter) external tool.

    Proper slope-aware ground/off-ground segmentation for point clouds with no
    ground class. Unlike a per-cell min-z DTM — which keeps object-bottom
    points as 'ground' and fills the occlusion holes the shadow channels need —
    CSF drapes a simulated cloth to recover true terrain even under objects.

    ``csf_exe`` is the path to csfground.exe (input LAS, output classified LAS,
    ASPRS class 2 = ground). Requires laspy. Result cached to ``cache`` (.npy).
    """
    import hashlib
    import json
    import subprocess
    import tempfile
    from pathlib import Path as _P

    from .io import _require_laspy
    laspy = _require_laspy()
    raw_points = np.asarray(points, np.float64)
    if raw_points.ndim != 2 or raw_points.shape[1] < 3 or len(raw_points) == 0:
        raise ValueError("points must be a non-empty array with at least three columns")
    xyz = raw_points[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("points contain non-finite XYZ coordinates")

    cache = _P(cache) if cache is not None else None
    cache_meta = cache.with_suffix(cache.suffix + ".json") if cache is not None else None
    digest = hashlib.sha256(np.ascontiguousarray(xyz).view(np.uint8)).hexdigest()
    expected_meta = {
        "version": 1,
        "xyz_sha256": digest,
        "point_count": len(xyz),
        "cloth_resolution": float(cloth_resolution),
        "classify_threshold": float(classify_threshold),
        "b_slope_smooth": bool(b_slope_smooth),
        "rigidness": None if rigidness is None else int(rigidness),
    }
    if cache is not None and cache.exists() and cache_meta.exists():
        try:
            saved_meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            saved_mask = np.load(cache, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError):
            saved_meta = None
            saved_mask = None
        if (saved_meta == expected_meta and saved_mask is not None
                and saved_mask.shape == (len(xyz),)):
            return saved_mask.astype(bool, copy=False)

    with tempfile.TemporaryDirectory() as td:
        td = _P(td)
        inp, outp = td / "in.las", td / "out.las"
        header = laspy.LasHeader(point_format=3)
        header.offsets = xyz.min(0)
        header.scales = [0.001, 0.001, 0.001]
        las = laspy.LasData(header)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.write(inp)

        cmd = [str(csf_exe), "-i", str(inp), "-o", str(outp),
               "-cloth_resolution", str(cloth_resolution),
               "-classify_threshold", str(classify_threshold),
               "-save_mode", "all"]
        if b_slope_smooth:
            cmd += ["-bSlopeSmooth", "true"]
        if rigidness is not None:
            cmd += ["-rigidness", str(rigidness)]
        subprocess.run(cmd, check=True, capture_output=True)

        out = laspy.read(outp)
        out_xyz = np.column_stack([out.x, out.y, out.z])
        is_ground = np.asarray(out.classification) == 2
        if len(out_xyz) == len(xyz) and np.allclose(out_xyz, xyz, atol=0.01):
            ground_mask = is_ground
        else:
            # CSF may reorder; map ground points back to the input by nearest xyz.
            from scipy.spatial import cKDTree
            ground_mask = np.zeros(len(xyz), bool)
            gpts = out_xyz[is_ground]
            if len(gpts):
                _, idx = cKDTree(xyz).query(gpts, k=1)
                ground_mask[idx] = True

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, ground_mask)
        cache_meta.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return ground_mask
