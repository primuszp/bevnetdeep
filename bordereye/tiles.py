"""LAS/LAZ tile utilities: metadata scanning, merging, and layout queries."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import _require_laspy


@dataclass(frozen=True)
class TileInfo:
    """Metadata for one LAS/LAZ tile, read from the file header only."""

    path: Path
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    point_count: int
    crs_wkt: str | None = None

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    @property
    def bounds_xyz(self) -> tuple[float, float, float, float, float, float]:
        return (self.min_x, self.min_y, self.min_z, self.max_x, self.max_y, self.max_z)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height_xy(self) -> float:
        return self.max_y - self.min_y

    @property
    def z_range(self) -> float:
        return self.max_z - self.min_z

    def overlaps(self, other: TileInfo) -> bool:
        """True if the XY footprints of two tiles overlap (touching = False)."""
        return not (
            self.max_x <= other.min_x
            or other.max_x <= self.min_x
            or self.max_y <= other.min_y
            or other.max_y <= self.min_y
        )


def _tile_info(path: Path, header) -> TileInfo:
    crs_wkt = None
    try:
        crs = header.parse_crs()
        crs_wkt = crs.to_wkt() if crs is not None else None
    except Exception:
        # Invalid/missing CRS metadata must not hide otherwise valid tile bounds.
        pass
    return TileInfo(
        path=path,
        min_x=float(header.mins[0]),
        min_y=float(header.mins[1]),
        min_z=float(header.mins[2]),
        max_x=float(header.maxs[0]),
        max_y=float(header.maxs[1]),
        max_z=float(header.maxs[2]),
        point_count=int(header.point_count),
        crs_wkt=crs_wkt,
    )


def scan_tiles(
    directory: Path | str,
    *,
    pattern: str = "*.la[sz]",
    recursive: bool = False,
) -> list[TileInfo]:
    """Scan a directory for LAS/LAZ files and return their header metadata.

    No points are loaded. Reading is done via laspy header-only access, which
    is very fast even for files with hundreds of millions of points.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern (default matches both .las and .laz).
        recursive: If True, recurse into sub-directories.

    Returns:
        List of TileInfo sorted by (min_x, min_y).
    """
    laspy = _require_laspy()

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    if pattern == "*.la[sz]":
        files = list_las_files(root, recursive=recursive)
    else:
        glob_fn = root.rglob if recursive else root.glob
        files = sorted(glob_fn(pattern))

    infos: list[TileInfo] = []
    for path in files:
        try:
            with laspy.open(path) as f:
                infos.append(_tile_info(path, f.header))
        except Exception as exc:
            import warnings
            warnings.warn(f"Could not read header for {path}: {exc}", stacklevel=2)

    return sorted(infos, key=lambda t: (t.min_x, t.min_y))


def list_las_files(
    directory: Path | str,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return naturally sorted LAS and LAZ paths without opening the files."""
    import re

    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in {".las", ".laz"}]

    def natural_key(path: Path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", path.name)]

    return sorted(files, key=natural_key)


def read_tile_info(path: Path | str) -> TileInfo:
    """Read one tile header through the same API used by :func:`scan_tiles`."""
    path = Path(path)
    result = _read_tile_headers([path])
    if not result:
        raise ValueError(f"Could not read LAS/LAZ header: {path}")
    return result[0]


def _read_tile_headers(paths: Iterable[Path]) -> list[TileInfo]:
    """Internal header reader shared by single-file and directory APIs."""
    laspy = _require_laspy()

    result: list[TileInfo] = []
    for path in paths:
        with laspy.open(path) as reader:
            result.append(_tile_info(path, reader.header))
    return result


def summarize_tiles(tiles: Iterable[TileInfo]) -> str:
    """Create a compact human-readable summary without loading point data."""
    items = list(tiles)
    if not items:
        return "Tiles: 0"
    widths = np.fromiter((t.width for t in items), dtype=np.float64)
    heights = np.fromiter((t.height_xy for t in items), dtype=np.float64)
    return (
        f"Tiles: {len(items)}\n"
        f"Total points: {sum(t.point_count for t in items)}\n"
        f"Global XY bounds: [{min(t.min_x for t in items):.3f}, {min(t.min_y for t in items):.3f}] - "
        f"[{max(t.max_x for t in items):.3f}, {max(t.max_y for t in items):.3f}]\n"
        f"Median tile size: {np.median(widths):.3f} x {np.median(heights):.3f}"
    )


def export_tile_layout_csv(tiles: Iterable[TileInfo], output: Path | str) -> Path:
    """Write header-only tile metadata as a GIS-friendly CSV table."""
    import csv

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("file", "point_count", "min_x", "min_y", "min_z",
                         "max_x", "max_y", "max_z", "width", "height", "z_range"))
        for tile in tiles:
            writer.writerow((tile.path.as_posix(), tile.point_count, *tile.bounds_xyz,
                             tile.width, tile.height_xy, tile.z_range))
    return path


def plot_tile_layout(
    tiles: Iterable[TileInfo],
    output: Path | str,
    *,
    color_by: str = "point_count",
    dpi: int = 180,
) -> Path:
    """Render a tile footprint overview without reading point records.

    ``color_by`` may be ``"point_count"``, ``"z_range"``, or ``"none"``.
    Matplotlib is imported lazily so headless/core workflows stay lightweight.
    """
    if color_by not in {"point_count", "z_range", "none"}:
        raise ValueError("color_by must be 'point_count', 'z_range', or 'none'")
    items = list(tiles)
    if not items:
        raise ValueError("tiles must not be empty")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import Rectangle

    values = np.array([t.point_count if color_by == "point_count" else t.z_range for t in items], float)
    if color_by == "point_count":
        values = np.maximum(values, 1)
        norm = LogNorm(float(values.min()), float(values.max()) if values.max() > values.min() else values.min() + 1)
    else:
        norm = Normalize(float(values.min()), float(values.max()) if values.max() > values.min() else values.min() + 1)
    cmap = plt.get_cmap("viridis")
    figure, axis = plt.subplots(figsize=(10, 10), dpi=dpi)
    for tile, value in zip(items, values, strict=True):
        color = (0.25, 0.5, 0.75, 0.45) if color_by == "none" else cmap(norm(value))
        axis.add_patch(Rectangle((tile.min_x, tile.min_y), tile.width, tile.height_xy,
                                 facecolor=color, edgecolor="black", linewidth=0.3))
    axis.autoscale()
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_title(f"LAS/LAZ tile layout ({len(items)} tiles)")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path



def merge_tiles(
    tiles: Iterable[TileInfo | Path | str],
    output: Path | str,
    *,
    chunk_size: int = 5_000_000,
) -> Path:
    """Merge LAS/LAZ tiles into a single output file (streaming, low memory).

    Points are written in tile order. The output LAS version and point format
    are taken from the first tile. The bounding-box header fields are updated
    from the actual merged extents.

    Args:
        tiles: TileInfo list (from scan_tiles) or bare file paths.
        output: Destination .las or .laz path.
        chunk_size: Number of points to buffer per tile during streaming.

    Returns:
        Path to the written output file.
    """
    tile_paths: list[Path] = []
    for t in tiles:
        if isinstance(t, TileInfo):
            tile_paths.append(t.path)
        else:
            tile_paths.append(Path(t))

    if not tile_paths:
        raise ValueError("No tiles provided")

    from .las_merge import merge_las
    return merge_las(tile_paths, output, chunk_size=chunk_size).output_file


def tile_point_cloud(
    xyz: np.ndarray,
    tile_size: float,
    *,
    overlap: float = 0.0,
    min_points: int = 100,
):
    """Split a point cloud into regular spatial tiles and yield each non-empty tile.

    The tiling is based on the XY extent of the input array.  Tiles are
    non-overlapping by default; set ``overlap`` to a positive value (metres) to
    add a border so adjacent tiles share a strip of points.

    Args:
        xyz:        (N, 3+) array — only the first two columns (X, Y) are used
                    for spatial binning; the full rows are included in the yield.
        tile_size:  Edge length of each tile in the same unit as the coordinates.
        overlap:    Width of overlap border on all four sides (default 0).
        min_points: Tiles with fewer points than this are silently skipped.

    Yields:
        (tile_xyz, bounds) where
            tile_xyz  is the subset of ``xyz`` inside the tile (including overlap),
            bounds    is ``(xmin, ymin, xmax, ymax)`` of the *core* tile (no overlap).
    """
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("xyz must be an Nx2+ array")
    if len(points) == 0:
        return
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("XY coordinates must be finite")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")
    if min_points <= 0:
        raise ValueError("min_points must be > 0")

    x = points[:, 0]
    y = points[:, 1]
    x0, y0 = float(x.min()), float(y.min())
    x1, y1 = float(x.max()), float(y.max())
    count_x = max(1, int(np.ceil((x1 - x0) / tile_size)))
    count_y = max(1, int(np.ceil((y1 - y0) / tile_size)))
    core_x = np.minimum(((x - x0) / tile_size).astype(np.int64), count_x - 1)
    core_y = np.minimum(((y - y0) / tile_size).astype(np.int64), count_y - 1)
    core_id = core_x * count_y + core_y
    order = np.argsort(core_id)
    sorted_ids = core_id[order]
    starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    groups = {
        int(sorted_ids[start]): indices
        for start, indices in zip(starts, np.split(order, starts[1:]), strict=True)
    }

    for ix in range(count_x):
        tx = x0 + ix * tile_size
        for iy in range(count_y):
            ty = y0 + iy * tile_size
            qx0 = tx - overlap
            qy0 = ty - overlap
            qx1 = tx + tile_size + overlap
            qy1 = ty + tile_size + overlap
            neighbours = [
                groups[nx * count_y + ny]
                for nx in range(max(0, ix - 1), min(count_x, ix + 2))
                for ny in range(max(0, iy - 1), min(count_y, iy + 2))
                if nx * count_y + ny in groups
            ]
            if not neighbours:
                continue
            selected = np.sort(np.concatenate(neighbours))
            selected = selected[
                (x[selected] >= qx0)
                & (y[selected] >= qy0)
                & (x[selected] <= qx1)
                & (y[selected] <= qy1)
            ]
            # Keep core-boundary ownership half-open while ensuring the global
            # maximum still belongs to the final tile.
            if ix < count_x - 1:
                selected = selected[x[selected] < qx1]
            if iy < count_y - 1:
                selected = selected[y[selected] < qy1]
            if len(selected) >= min_points:
                bounds = (tx, ty, tx + tile_size, ty + tile_size)
                yield points[selected], bounds


def tiles_covering(
    tiles: Iterable[TileInfo],
    bounds: tuple[float, float, float, float],
) -> list[TileInfo]:
    """Return tiles whose XY footprint overlaps the given bounding box.

    Args:
        tiles: Sequence of TileInfo to filter.
        bounds: (min_x, min_y, max_x, max_y) query rectangle.
    """
    min_x, min_y, max_x, max_y = bounds
    if not np.isfinite(bounds).all() or max_x < min_x or max_y < min_y:
        raise ValueError("bounds must be finite and ordered min-to-max")
    query = TileInfo(
        path=Path(""),
        min_x=min_x, min_y=min_y, min_z=0.0,
        max_x=max_x, max_y=max_y, max_z=0.0,
        point_count=0,
    )
    return [t for t in tiles if t.overlaps(query)]
