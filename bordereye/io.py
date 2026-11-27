from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_laspy():
    try:
        import laspy  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LAS/LAZ support requires the I/O extra: pip install 'bordereye[io]'"
        ) from exc
    return laspy


def _xyz_array(points) -> np.ndarray:
    return np.column_stack((
        np.asarray(points.x, dtype=np.float64),
        np.asarray(points.y, dtype=np.float64),
        np.asarray(points.z, dtype=np.float64),
    ))


def _las_fields(points) -> dict[str, np.ndarray]:
    """Detach scaled XYZ and every non-coordinate LAS dimension."""
    dimensions = tuple(points.point_format.dimension_names)
    result: dict[str, np.ndarray] = {"xyz": _xyz_array(points)}
    for name in dimensions:
        if name.lower() not in {"x", "y", "z"}:
            result[name] = np.asarray(points[name]).copy()
    # Stable analysis-friendly dtypes for the most commonly used fields.
    result["classification"] = (
        np.asarray(points.classification, dtype=np.int32)
        if "classification" in dimensions
        else np.zeros(len(points), dtype=np.int32)
    )
    if "intensity" in dimensions:
        result["intensity"] = np.asarray(points.intensity, dtype=np.float32)
    return result


def _integer_field(
    values: np.ndarray | None,
    *,
    name: str,
    point_count: int,
    dtype: type,
    maximum: int,
) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.shape != (point_count,):
        raise ValueError(f"{name} must be a 1D array with one value per point")
    try:
        valid_integers = np.isfinite(arr).all() and np.equal(arr, np.rint(arr)).all()
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not valid_integers:
        raise ValueError(f"{name} must contain finite integer values")
    if len(arr) and (arr.min() < 0 or arr.max() > maximum):
        raise ValueError(f"{name} values must be in [0, {maximum}]")
    return arr.astype(dtype)


def read_las(path: str | Path, *, with_intensity: bool = False):
    laspy = _require_laspy()
    las = laspy.read(str(path))
    xyz = _xyz_array(las)
    if with_intensity:
        return xyz, np.asarray(las.intensity, dtype=np.float32)
    return xyz


def read_las_data(path: str | Path):
    """Read and return the raw laspy LasData object."""
    laspy = _require_laspy()
    return laspy.read(str(path))


def write_las(
    path: str | Path,
    xyz: np.ndarray,
    *,
    classification: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    point_source_id: np.ndarray | None = None,
) -> None:
    """Write an Nx3+ array as a LAS 1.2 file (point format 0, mm scale).

    Optional per-point attributes (each a length-N array):
        classification: point class codes (uint8, 0-31 in LAS 1.2).
        intensity: return intensity (uint16).
        point_source_id: uint16 id — handy for storing cluster/instance ids.

    Extra *columns* of ``xyz`` beyond the first three are ignored.
    """
    laspy = _require_laspy()
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("xyz must be an Nx3+ array")
    if len(pts) == 0:
        raise ValueError("xyz must not be empty")
    if not np.isfinite(pts[:, :3]).all():
        raise ValueError("xyz coordinates must be finite")

    field_options = {"point_count": len(pts)}
    cls = _integer_field(
        classification, name="classification", dtype=np.uint8, maximum=31, **field_options
    )
    ity = _integer_field(
        intensity, name="intensity", dtype=np.uint16, maximum=65_535, **field_options
    )
    psid = _integer_field(
        point_source_id, name="point_source_id", dtype=np.uint16, maximum=65_535, **field_options
    )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = pts[:, :3].min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    if np.any(np.ptp(pts[:, :3], axis=0) / header.scales > np.iinfo(np.int32).max):
        raise ValueError("xyz extent is too large for LAS int32 coordinates at 0.001 scale")
    las = laspy.LasData(header)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    if cls is not None:
        las.classification = cls
    if ity is not None:
        las.intensity = ity
    if psid is not None:
        las.point_source_id = psid
    las.write(str(out_path))



def read_las_full(path: str | Path) -> dict:
    """Read a LAS/LAZ file and return all standard fields as a dict.

    Always returns:
        xyz            (N, 3) float64  — world coordinates
        classification (N,)   int32    — point class codes

    All other standard and extra-byte point dimensions are returned under
    their LAS names. Intensity is normalised to float32; other optional fields
    retain their stored dtype.

    Args:
        path: Path to a .las or .laz file. LAZ requires the lazrs or laszip
              backend (``pip install lazrs``); a clear error is raised if absent.

    Returns:
        Detached arrays that do not retain the complete packed LAS record.
    """
    laspy = _require_laspy()
    las = laspy.read(str(path))
    return _las_fields(las)


def iter_las_chunks(
    path: str | Path,
    *,
    chunk_size: int = 2_000_000,
    attributes: bool = False,
):
    """Yield LAS/LAZ points in bounded-memory chunks.

    By default each chunk is an ``Nx3`` float64 XYZ array. With
    ``attributes=True`` each chunk follows :func:`read_las_full`.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    laspy = _require_laspy()
    with laspy.open(str(path)) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            yield _las_fields(chunk) if attributes else _xyz_array(chunk)


DALES_DTYPE: np.dtype = np.dtype([
    ("x",         "<f4"),
    ("y",         "<f4"),
    ("z",         "<f4"),
    ("intensity", "<i4"),
    ("sem_class", "<i4"),
    ("ins_class", "<i4"),
])
"""Structured dtype of a DALES(Objects) PLY point record (24 bytes)."""

_DALES_PROPERTIES = (
    ("float", "x"), ("float", "y"), ("float", "z"),
    ("int", "intensity"), ("int", "sem_class"), ("int", "ins_class"),
)


def _read_dales_header(stream, path: Path) -> int:
    vertex_count = None
    file_format = None
    current_element = None
    properties: list[tuple[str, str]] = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError(f"Unexpected EOF before end_header in {path}")
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{path.name}: PLY header must be ASCII") from exc
        parts = text.split()
        if parts[:1] == ["format"] and len(parts) >= 2:
            file_format = parts[1]
        elif parts[:1] == ["element"] and len(parts) >= 3:
            current_element = parts[1]
            if current_element in {"vertex", "testing"}:
                vertex_count = int(parts[2])
        elif (
            parts[:1] == ["property"]
            and len(parts) >= 3
            and current_element in {"vertex", "testing"}
        ):
            properties.append(
                (parts[1].replace("float32", "float").replace("int32", "int"), parts[2])
            )
        if text == "end_header":
            break
    if file_format != "binary_little_endian":
        raise ValueError(f"{path.name}: expected binary_little_endian PLY, got {file_format!r}")
    if vertex_count is None:
        raise ValueError(f"Point count not found in PLY header: {path}")
    if vertex_count < 0:
        raise ValueError(f"{path.name}: vertex count must be non-negative")
    if tuple(properties) != _DALES_PROPERTIES:
        raise ValueError(
            f"{path.name}: PLY property layout {properties} does not match the "
            f"DALES schema {list(_DALES_PROPERTIES)}"
        )
    return vertex_count


def read_dales_ply(path: str | Path) -> np.ndarray:
    """Read a DALESObjects binary-little-endian PLY file.

    Fixed fields: x, y, z (float32); intensity, sem_class, ins_class (int32).
    The header is validated against this schema so a file with a different
    property layout fails loudly instead of yielding garbage.

    Args:
        path: Path to the .ply file.

    Returns:
        Structured array (:data:`DALES_DTYPE`) with fields
        x, y, z, intensity, sem_class, ins_class.
    """
    path = Path(path)
    with path.open("rb") as f:
        vertex_count = _read_dales_header(f, path)
        data = np.fromfile(f, dtype=DALES_DTYPE, count=vertex_count)
        if len(data) != vertex_count:
            raise ValueError(
                f"{path.name}: expected {vertex_count} points, read {len(data)}"
            )
        return data
