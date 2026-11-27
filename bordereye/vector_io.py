"""Minimal polygon Shapefile I/O with no GeoPandas dependency."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .geometry import polygon_area
from .spatial import points_in_polygon_2d


def _require_pyshp():
    try:
        import shapefile  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Shapefile support requires the vector extra: pip install 'bordereye[vector]'"
        ) from exc
    return shapefile


@dataclass(frozen=True)
class PolygonFeature:
    """One polygon feature with an outer ring, optional holes, and DBF attributes."""
    vertices: np.ndarray
    properties: dict[str, object]
    holes: tuple[np.ndarray, ...] = ()


def _signed_area(ring: np.ndarray) -> float:
    open_ring = ring[:-1] if len(ring) > 1 and np.array_equal(ring[0], ring[-1]) else ring
    return 0.5 * float(
        np.dot(open_ring[:, 0], np.roll(open_ring[:, 1], -1))
        - np.dot(open_ring[:, 1], np.roll(open_ring[:, 0], -1))
    )


def _shapefile_ring(vertices: np.ndarray, *, clockwise: bool) -> np.ndarray:
    ring = np.asarray(vertices, dtype=np.float64)
    if ring.ndim != 2 or ring.shape[1] < 2 or len(ring) < 3:
        raise ValueError("each polygon ring must be an Nx2+ array with at least three vertices")
    ring = ring[:, :2]
    if not np.isfinite(ring).all():
        raise ValueError("polygon coordinates must be finite")
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack((ring, ring[0]))
    if (_signed_area(ring) < 0) != clockwise:
        ring = ring[::-1]
    return ring


def read_shapefile(path: Path | str) -> list[PolygonFeature]:
    """Read polygon SHP geometry and attributes into lightweight objects."""
    shapefile = _require_pyshp()

    result: list[PolygonFeature] = []
    with shapefile.Reader(str(path)) as reader:
        fields = [field[0] for field in reader.fields[1:]]
        for shape_record in reader.iterShapeRecords():
            shape = shape_record.shape
            if shape.shapeType not in {shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM}:
                continue
            starts = [*shape.parts, len(shape.points)]
            properties = dict(zip(fields, shape_record.record, strict=True))
            rings = [
                np.asarray(shape.points[start:stop], dtype=np.float64)[:, :2]
                for start, stop in pairwise(starts)
                if stop - start >= 3
            ]
            outer = [ring for ring in rings if _signed_area(ring) < 0]
            holes = [ring for ring in rings if _signed_area(ring) >= 0]
            # Non-conforming writers sometimes emit every ring CCW. Preserve
            # such data as separate polygons rather than dropping geometry.
            if not outer:
                result.extend(PolygonFeature(ring, properties.copy()) for ring in rings)
                continue
            assigned: list[list[np.ndarray]] = [[] for _ in outer]
            for hole in holes:
                containers = [
                    i for i, shell in enumerate(outer)
                    if points_in_polygon_2d(hole[:1], shell)[0]
                ]
                if containers:
                    owner = min(containers, key=lambda i: polygon_area(outer[i]))
                    assigned[owner].append(hole)
                else:
                    outer.append(hole)
                    assigned.append([])
            result.extend(
                PolygonFeature(shell, properties.copy(), tuple(shell_holes))
                for shell, shell_holes in zip(outer, assigned, strict=True)
            )
    return result


def _dbf_field_names(names: list[str]) -> dict[str, str]:
    """Create deterministic, case-insensitively unique 10-byte DBF names."""
    used = {"ID"}
    result: dict[str, str] = {}
    for name in names:
        base = name[:10] or "field"
        candidate = base
        number = 2
        while candidate.upper() in used:
            suffix = f"_{number}"
            candidate = base[:10 - len(suffix)] + suffix
            number += 1
        used.add(candidate.upper())
        result[name] = candidate
    return result


def write_shapefile(
    polygons: Iterable[np.ndarray | PolygonFeature],
    path: Path | str,
    *,
    crs_wkt: str | None = None,
) -> Path:
    """Write polygons with a stable integer ID using the lightweight SHP format."""
    shapefile = _require_pyshp()

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    items = list(polygons)
    property_names = sorted({name for item in items if isinstance(item, PolygonFeature)
                             for name in item.properties})
    field_names = _dbf_field_names(property_names)

    def field_type(name):
        values = [item.properties.get(name) for item in items
                  if isinstance(item, PolygonFeature) and item.properties.get(name) is not None]
        if values and all(isinstance(value, (bool, np.bool_)) for value in values):
            return ("L", 1, 0)
        if values and all(isinstance(value, (int, np.integer)) for value in values):
            return ("N", 18, 0)
        if values and all(isinstance(value, (int, float, np.number)) for value in values):
            return ("F", 20, 8)
        return ("C", 254, 0)

    with shapefile.Writer(str(output), shapeType=shapefile.POLYGON) as writer:
        writer.field("id", "N", size=18, decimal=0)
        for name in property_names:
            kind, size, decimals = field_type(name)
            writer.field(field_names[name], kind, size=size, decimal=decimals)
        for feature_id, item in enumerate(items, 1):
            vertices = item.vertices if isinstance(item, PolygonFeature) else np.asarray(item, float)
            rings = [_shapefile_ring(vertices, clockwise=True)]
            if isinstance(item, PolygonFeature):
                rings.extend(_shapefile_ring(hole, clockwise=False) for hole in item.holes)
            writer.poly([ring.tolist() for ring in rings])
            properties = item.properties if isinstance(item, PolygonFeature) else {}
            writer.record(feature_id, *(properties.get(name) for name in property_names))
    if crs_wkt:
        output.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")
    return output
