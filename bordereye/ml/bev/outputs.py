"""Geospatial output writers for BEV detections and segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bordereye import ERMapper, RasterGrid
from bordereye.export import _round_or_none, _write_eov_points


def _finite(value: Any, default: float = 0.0) -> float:
    number = float(value)
    return number if np.isfinite(number) else default


def _closed_ccw_ring(vertices: np.ndarray) -> list[list[float]]:
    ring = np.asarray(vertices, dtype=np.float64)
    if ring.shape != (4, 2) or not np.isfinite(ring).all():
        raise ValueError("detection vertices must be a finite (4, 2) array")
    signed_area = 0.5 * float(
        np.dot(ring[:, 0], np.roll(ring[:, 1], -1))
        - np.dot(ring[:, 1], np.roll(ring[:, 0], -1))
    )
    if signed_area < 0:
        ring = ring[::-1]
    ring = np.vstack((ring, ring[0]))
    return np.round(ring, 3).tolist()


def write_detections_geojson(
    detections: list[dict[str, Any]],
    path: str | Path,
    *,
    crs_name: str | None = None,
    name: str | None = None,
) -> Path:
    """Export detected vehicle centers from their source CRS to project EOV GeoJSON."""
    records = []
    for object_id, detection in enumerate(detections, 1):
        extra = {"object_id": object_id, "pointSource": "detected", "detected": True}
        for source, key, digits in (
            ("L", "length", 3), ("W", "width", 3), ("height", "height", 3),
            ("z", "center_z", 3), ("score", "score", 6), ("hm_score", "heatmap_score", 6),
            ("seg_center_score", "seg_center_score", 6), ("seg_obb_support", "seg_obb_support", 6),
        ):
            value = detection.get(source)
            extra[key] = None if value is None else _round_or_none(value, digits)
        angle = detection.get("th")
        extra["rotation_deg"] = None if angle is None else _round_or_none(np.degrees(angle))
        records.append(((detection["cx"], detection["cy"]), extra))
    return _write_eov_points(records, path, crs_name=crs_name, name=name)


def _shapefile_paths(path: str | Path) -> tuple[Path, Path]:
    obb_path = Path(path)
    if obb_path.suffix.lower() != ".shp":
        obb_path = obb_path.with_suffix(".shp")
    center_path = obb_path.with_name(f"{obb_path.stem}_centers.shp")
    return obb_path, center_path


def write_detection_shapefiles(
    detections: list[dict[str, Any]],
    path: str | Path,
    *,
    crs_wkt: str | None = None,
) -> tuple[Path, Path]:
    """Write OBB polygons and centre points as two linked shapefiles."""
    from bordereye.vector_io import _require_pyshp

    shapefile = _require_pyshp()
    obb_path, center_path = _shapefile_paths(path)
    obb_path.parent.mkdir(parents=True, exist_ok=True)

    def fields(writer) -> None:
        writer.field("OBJECT_ID", "N", size=10)
        writer.field("CLASS", "C", size=16)
        writer.field("SCORE", "F", size=12, decimal=6)
        writer.field("CENTER_X", "F", size=18, decimal=3)
        writer.field("CENTER_Y", "F", size=18, decimal=3)
        writer.field("LENGTH_M", "F", size=12, decimal=3)
        writer.field("WIDTH_M", "F", size=12, decimal=3)
        writer.field("ROT_DEG", "F", size=12, decimal=3)
        writer.field("CENTER_Z", "F", size=12, decimal=3)
        writer.field("HEIGHT_M", "F", size=12, decimal=3)

    def record(writer, object_id: int, detection: dict[str, Any]) -> None:
        writer.record(
            object_id,
            "vehicle",
            _finite(detection.get("score")),
            _finite(detection.get("cx")),
            _finite(detection.get("cy")),
            _finite(detection.get("L")),
            _finite(detection.get("W")),
            np.degrees(_finite(detection.get("th"))),
            _finite(detection.get("z")),
            _finite(detection.get("height")),
        )

    with shapefile.Writer(str(obb_path), shapeType=shapefile.POLYGON) as polygons:
        fields(polygons)
        for object_id, detection in enumerate(detections, 1):
            ring = _closed_ccw_ring(detection["verts"])[::-1]
            polygons.poly([ring])
            record(polygons, object_id, detection)

    with shapefile.Writer(str(center_path), shapeType=shapefile.POINT) as centers:
        fields(centers)
        for object_id, detection in enumerate(detections, 1):
            centers.point(_finite(detection.get("cx")), _finite(detection.get("cy")))
            record(centers, object_id, detection)

    if crs_wkt:
        for output in (obb_path, center_path):
            output.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")
    return obb_path, center_path


def write_segmentation_mask_ers(
    probability: np.ndarray,
    reference: RasterGrid,
    path: str | Path,
    *,
    threshold: float = 0.5,
    datum: str = "HD72",
    projection: str = "EOV",
) -> Path:
    """Write a georeferenced uint8 mask (background=0, vehicle=1)."""
    values = np.asarray(probability)
    if values.shape != reference.shape:
        raise ValueError(
            f"segmentation shape {values.shape} does not match raster {reference.shape}"
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mask threshold must be in [0, 1]")
    mask = (np.nan_to_num(values, nan=0.0) >= threshold).astype(np.float64)
    raster = RasterGrid.from_dense(
        mask,
        cell_size=reference.cell_size,
        origin=reference.origin,
        lazy=True,
    )
    output = Path(path)
    ERMapper().write(
        output,
        raster,
        celltype="Unsigned8BitInteger",
        nodata=255,
        datum=datum,
        projection=projection,
    )
    return output if output.suffix.lower() == ".ers" else Path(str(output) + ".ers")
