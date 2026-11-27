from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .candidates import GridCandidate
from .detection import DetectedVehicle
from .holes import RasterHole
from .raster import RasterGrid
from .vector_io import _require_pyshp


def _round_or_none(value: float, ndigits: int = 3) -> float | None:
    """Round for JSON output; NaN/inf become None (JSON has no NaN token)."""
    v = float(value)
    return round(v, ndigits) if np.isfinite(v) else None


def _shp_ring(pts: np.ndarray) -> list[list[float]]:
    """Closed clockwise ring as required for shapefile outer rings."""
    ring = np.asarray(pts, dtype=np.float64)
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack((ring, ring[0]))
    # Signed area > 0 means counter-clockwise -> reverse to clockwise.
    x, y = ring[:-1, 0], ring[:-1, 1]
    signed_area = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed_area > 0:
        ring = ring[::-1]
    return ring.tolist()


def _write_prj(shp_path: Path, crs_wkt: str | None) -> None:
    if crs_wkt:
        shp_path.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")


def save_raster_points(raster: RasterGrid, path: str | Path, *, stat: str = "mean") -> None:
    from .io import write_las

    write_las(path, raster.cell_centers(stat))


def save_holes_shapefile(
    holes: list[RasterHole], path: str | Path, *, crs_wkt: str | None = None
) -> None:
    shapefile = _require_pyshp()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(out_path), shapeType=shapefile.POLYGON) as shp:
        shp.field("ID", "N")
        shp.field("CELLS", "N")
        shp.field("AREA", "F", decimal=3)
        for hole in holes:
            if len(hole.hull_world) < 3:
                continue
            shp.record(hole.id, hole.area_cells, hole.hull_area)
            shp.poly([_shp_ring(hole.hull_world)])
    _write_prj(out_path, crs_wkt)


def _write_eov_points(
    records: Iterable[tuple[tuple[float, float], dict]],
    path: str | Path,
    *,
    crs_name: str | None,
    name: str | None,
    label: str = "gepjarmu",
) -> Path:
    from pyproj import Transformer

    if label not in {
        "alagut", "fa", "cserje", "gepjarmu", "kerites", "letra",
        "emberek", "hulladek", "osveny", "tisztas", "egyeb",
    }:
        raise ValueError(f"Unknown project label: {label!r}")
    transformer = None
    if crs_name:
        source = "EPSG:23700" if crs_name.upper() == "EOV" else crs_name
        transformer = Transformer.from_crs(source, "EPSG:23700", always_xy=True)
    out_path = Path(path)
    payload = {
        "type": "FeatureCollection",
        "name": name or out_path.stem,
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::23700"}},
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    features = []
    coordinates = []
    for center, extra in records:
        if transformer is None:
            raise ValueError("GeoJSON export requires a source CRS; supply crs_name or --crs-name")
        x, y = transformer.transform(*center, errcheck=True)
        if not np.isfinite([x, y]).all():
            raise ValueError("Vehicle center must have finite EOV coordinates")
        point = [round(x, 2), round(y, 2)]
        coordinates.append(point)
        features.append({
            "type": "Feature",
            "properties": {
                "id": str(uuid.uuid4()),
                "timestamp": timestamp,
                "source": "LiDAR",
                "label": label,
                "extra": extra,
            },
            "geometry": {"type": "Point", "coordinates": point},
        })
    if coordinates:
        xs, ys = zip(*coordinates, strict=True)
        payload["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
    payload["features"] = features
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content + "\n", encoding="utf-8")
    return out_path


def save_candidates_geojson(
    candidates: Iterable[GridCandidate | DetectedVehicle],
    path: str | Path,
    *,
    label: str = "gepjarmu",
    crs_name: str | None = None,
    name: str | None = None,
) -> None:
    """Export one EOV center per candidate. crs_name describes the input coordinates."""
    records = []
    for candidate in candidates:
        extra = {
            "component_id": candidate.id,
            "pointSource": "detected",
            "detected": True,
            "method": candidate.source,
            "point_count": candidate.point_count,
            "length": _round_or_none(candidate.length),
            "width": _round_or_none(candidate.width),
            "height": _round_or_none(candidate.height),
            "rotation_deg": _round_or_none(candidate.rotation_deg),
        }
        if isinstance(candidate, DetectedVehicle):
            center = candidate.center_xy
            extra.update(
                vehicleCategory=candidate.vehicle_class, score=_round_or_none(candidate.class_score)
            )
        else:
            extra.update(
                cell_count=candidate.cell_count,
                value_min=_round_or_none(candidate.value_min),
                value_max=_round_or_none(candidate.value_max),
                value_mean=_round_or_none(candidate.value_mean),
            )
            if candidate.obb_2d is not None:
                center = candidate.obb_2d.center
            else:
                bounds = candidate.world_bounds
                dimensions = len(bounds) // 2
                center = (
                    (bounds[0] + bounds[dimensions]) / 2,
                    (bounds[1] + bounds[dimensions + 1]) / 2,
                )
        records.append((center, extra))
    _write_eov_points(records, path, crs_name=crs_name, name=name, label=label)


def save_candidates_shapefile(
    candidates: list[GridCandidate], path: str | Path, *, crs_wkt: str | None = None
) -> None:
    shapefile = _require_pyshp()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with shapefile.Writer(str(out_path), shapeType=shapefile.POLYGON) as shp:
        shp.field("ID", "N")
        shp.field("SRC", "C")
        shp.field("LEN", "F", decimal=3)
        shp.field("WID", "F", decimal=3)
        shp.field("HGT", "F", decimal=3)
        shp.field("PTS", "N")
        for candidate in candidates:
            if candidate.obb_2d is None:
                continue
            shp.record(
                candidate.id,
                candidate.source,
                candidate.length,
                candidate.width,
                # NaN height (raster-hole candidates) -> NULL field, not "nan"
                candidate.height if np.isfinite(candidate.height) else None,
                candidate.point_count,
            )
            shp.poly([_shp_ring(candidate.obb_2d.vertices)])
    _write_prj(out_path, crs_wkt)


def save_hulls_dxf(
    hulls: list[tuple[int, np.ndarray]],
    path: str | Path,
    *,
    layer_name: str = "HULLS",
) -> None:
    if not layer_name or any(char in layer_name for char in "\r\n"):
        raise ValueError("layer_name must be non-empty and contain no newlines")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="ascii", newline="\n") as f:
        f.write("0\nSECTION\n2\nHEADER\n")
        f.write("9\n$ACADVER\n1\nAC1009\n")
        f.write("0\nENDSEC\n")
        f.write("0\nSECTION\n2\nENTITIES\n")
        for _, hull in hulls:
            pts = np.asarray(hull, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) < 2:
                continue
            if not np.isfinite(pts[:, :2]).all():
                raise ValueError("hull coordinates must be finite")
            f.write("0\nPOLYLINE\n")
            f.write(f"8\n{layer_name}\n")
            f.write("66\n1\n70\n1\n62\n3\n")
            f.write("10\n0.0\n20\n0.0\n30\n0.0\n")
            for x, y in pts[:, :2]:
                f.write("0\nVERTEX\n")
                f.write(f"8\n{layer_name}\n")
                f.write(f"10\n{x:.6f}\n20\n{y:.6f}\n30\n0.0\n")
            f.write("0\nSEQEND\n")
            f.write(f"8\n{layer_name}\n")
        f.write("0\nENDSEC\n0\nEOF\n")


def save_candidates_dxf(
    candidates: list[GridCandidate],
    path: str | Path,
    *,
    layer_name: str = "CANDIDATES",
) -> None:
    hulls = [
        (candidate.id, candidate.obb_2d.vertices)
        for candidate in candidates
        if candidate.obb_2d is not None
    ]
    save_hulls_dxf(hulls, path, layer_name=layer_name)
