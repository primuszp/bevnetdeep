from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .raster import RasterGrid

_CELLTYPE_TO_DTYPE = {
    "ieee4bytereal": np.float32,
    "ieee8bytereal": np.float64,
    "signed8bitinteger": np.int8,
    "signed16bitinteger": np.int16,
    "signed32bitinteger": np.int32,
    "unsigned8bitinteger": np.uint8,
    "unsigned16bitinteger": np.uint16,
    "unsigned32bitinteger": np.uint32,
}


@lru_cache(maxsize=32)
def _cached_header(path: str, mtime_ns: int, size: int) -> tuple[tuple[str, str], ...]:
    """Parse an unchanged ERS header once; mtime and size invalidate safely."""
    del mtime_ns, size  # values deliberately participate in the cache key
    header: dict[str, str] = {}
    with Path(path).open("rt", encoding="utf-8") as stream:
        for line in stream:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                header[key.strip().lower()] = value.strip().strip('"')
    return tuple(header.items())


class ERMapper:
    """Reader/writer for the ER Mapper .ers header + flat binary raster format.

    Georeferencing follows the GDAL/ER Mapper convention: the
    ``RegistrationCoord`` is the world coordinate of the **top-left corner**
    of the registration cell (``RegistrationCellX/Y``, default cell 0,0) —
    not the cell centre. Files written by this class open pixel-exact in
    GDAL-based tools (QGIS, rasterio, gdal_translate).
    """

    def read(self, path: str | Path) -> RasterGrid:
        base = Path(path)
        header = self._read_header(base)
        data = self._read_raster(base, header)

        rows = int(header["nroflines"])
        cols = int(header["nrofcellsperline"])
        cell_x = float(header["xdimension"])
        cell_y = abs(float(header["ydimension"]))
        if rows <= 0 or cols <= 0:
            raise ValueError("ERS raster dimensions must be > 0")
        if not np.isfinite((cell_x, cell_y)).all() or cell_x <= 0 or cell_y <= 0:
            raise ValueError("ERS cell dimensions must be finite and > 0")
        x_value = header.get("eastings", header.get("longitude"))
        y_value = header.get("northings", header.get("latitude"))
        if x_value is None or y_value is None:
            raise ValueError("ERS RegistrationCoord must contain Eastings/Northings")
        reg_x = float(x_value)
        reg_y = float(y_value)
        if not np.isfinite((reg_x, reg_y)).all():
            raise ValueError("ERS registration coordinates must be finite")
        # Registration coordinate = top-left corner of the registration cell.
        # A non-zero RegistrationCellX/Y shifts the grid accordingly (same
        # adjustment GDAL's ERS driver applies).
        reg_cell_x = float(header.get("registrationcellx", "0"))
        reg_cell_y = float(header.get("registrationcelly", "0"))
        min_x = reg_x - reg_cell_x * cell_x
        max_y = reg_y + reg_cell_y * cell_y
        min_y = max_y - rows * cell_y
        if data.shape != (rows, cols):
            raise ValueError("ERS raster size does not match header")
        return RasterGrid.from_dense(
            data,
            cell_size=(cell_x, cell_y),
            origin=(min_x, min_y),
            lazy=True,
        )

    def write(
        self,
        path: str | Path,
        raster: RasterGrid,
        *,
        celltype: str = "IEEE4ByteReal",
        nodata: float = -99999.0,
        datum: str = "HD72",
        projection: str = "EOV",
    ) -> None:
        """Write a raster as an ERS header + flat binary pair.

        NaN cells are stored as *nodata* (default -99999, NOT 0: zero is a
        legitimate value for height-above-ground rasters).
        """
        base = Path(path)
        if any(char in value for value in (datum, projection) for char in ('"', "\r", "\n")):
            raise ValueError("datum and projection must not contain quotes or newlines")
        data = raster.to_dense("mean")
        rows, cols = data.shape
        cell_x, cell_y = raster.cell_size
        min_x, _, _, max_y = raster.bounds
        # Corner registration (GDAL convention): RegistrationCoord is the
        # top-left corner of cell (0,0), i.e. (min_x, max_y).
        reg_x = min_x
        reg_y = max_y
        header = {
            "celltype": celltype,
            "xdimension": str(cell_x),
            "ydimension": str(cell_y),
            "eastings": str(reg_x),
            "northings": str(reg_y),
            "nroflines": str(rows),
            "nrofcellsperline": str(cols),
            "nullcellvalue": repr(float(nodata)),
            "datum": datum,
            "projection": projection,
        }
        self._validate_output(data, header)
        self._write_header(base, header)
        self._write_raster(base, data, header)

    def _validate_output(self, data: np.ndarray, header: dict[str, str]) -> None:
        """Reject values that the selected ERS cell type cannot round-trip."""
        dtype = self._dtype(header)
        nodata = float(header["nullcellvalue"])
        valid = ~np.isnan(data)
        values = data[valid]
        if not np.isfinite(values).all():
            raise ValueError("ERS raster values must be finite or NaN")
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if not np.isfinite(nodata) or nodata != np.rint(nodata) or not info.min <= nodata <= info.max:
                raise ValueError(f"nodata must be an integer in [{info.min}, {info.max}] for {dtype.name}")
            if len(values) and (
                values.min() < info.min
                or values.max() > info.max
                or not np.equal(values, np.rint(values)).all()
            ):
                raise ValueError(f"raster values must be integers in [{info.min}, {info.max}] for {dtype.name}")
        else:
            limit = np.finfo(dtype).max
            if len(values) and np.abs(values).max() > limit:
                raise ValueError(f"raster values exceed the range of {dtype.name}")
        if np.isfinite(nodata) and np.equal(values, nodata).any():
            raise ValueError("nodata collides with a valid raster value")

    def _header_path(self, path: Path) -> Path:
        return path if path.suffix.lower() == ".ers" else Path(str(path) + ".ers")

    def _bin_path(self, path: Path) -> Path:
        return path.with_suffix("") if path.suffix.lower() == ".ers" else path

    def _read_header(self, path: Path) -> dict[str, str]:
        header_path = self._header_path(path)
        stat = header_path.stat()
        return dict(_cached_header(str(header_path), stat.st_mtime_ns, stat.st_size))

    def _write_header(self, path: Path, header: dict[str, str]) -> None:
        header_path = self._header_path(path)
        header_path.parent.mkdir(parents=True, exist_ok=True)
        with header_path.open("wt", encoding="utf-8", newline="\n") as f:
            f.write("DatasetHeader Begin\n")
            f.write('\tVersion = "6.4"\n')
            f.write("\tDatasetType = ERStorage\n")
            f.write("\tDataType = Raster\n")
            f.write("\tByteOrder = LSBFirst\n")
            f.write("\tCoordinateSpace Begin\n")
            f.write(f'\t\tDatum = "{header.get("datum", "HD72")}"\n')
            f.write(f'\t\tProjection = "{header.get("projection", "EOV")}"\n')
            f.write("\t\tCoordinateType = EN\n")
            f.write('\t\tUnits = "METERS"\n')
            f.write("\tCoordinateSpace End\n")
            f.write("\tRasterInfo Begin\n")
            f.write(f"\t\tCellType = {header['celltype']}\n")
            f.write(f"\t\tNullCellValue = {header['nullcellvalue']}\n")
            f.write("\t\tRegistrationCellX = 0\n")
            f.write("\t\tRegistrationCellY = 0\n")
            f.write("\t\tCellInfo Begin\n")
            f.write(f"\t\t\tXDimension = {header['xdimension']}\n")
            f.write(f"\t\t\tYDimension = {header['ydimension']}\n")
            f.write("\t\tCellInfo End\n")
            f.write(f"\t\tNrOfLines = {header['nroflines']}\n")
            f.write(f"\t\tNrOfCellsPerLine = {header['nrofcellsperline']}\n")
            f.write("\t\tRegistrationCoord Begin\n")
            f.write(f"\t\t\tEastings = {header['eastings']}\n")
            f.write(f"\t\t\tNorthings = {header['northings']}\n")
            f.write("\t\tRegistrationCoord End\n")
            f.write("\t\tNrOfBands = 1\n")
            f.write("\t\tBandID Begin\n")
            f.write('\t\t\tValue = "Band1"\n')
            f.write("\t\tBandID End\n")
            f.write("\tRasterInfo End\n")
            f.write("DatasetHeader End\n")
        _cached_header.cache_clear()

    def _dtype(self, header: dict[str, str]) -> np.dtype:
        celltype = header.get("celltype", "")
        dtype = _CELLTYPE_TO_DTYPE.get(celltype.lower())
        if dtype is None:
            raise ValueError(
                f"Unsupported ERS CellType {celltype!r}. "
                f"Supported: {', '.join(sorted(_CELLTYPE_TO_DTYPE))}"
            )
        return np.dtype(dtype)

    def _read_raster(self, path: Path, header: dict[str, str]) -> np.ndarray:
        n_bands = int(header.get("nrofbands", "1"))
        if n_bands != 1:
            raise ValueError(
                f"Only single-band ERS rasters are supported (file has {n_bands} bands)"
            )
        dtype = self._dtype(header)
        byte_order = header.get("byteorder", "lsbfirst").lower()
        if byte_order not in {"lsbfirst", "msbfirst"}:
            raise ValueError("ERS ByteOrder must be LSBFirst or MSBFirst")
        file_dtype = dtype.newbyteorder("<" if byte_order == "lsbfirst" else ">")
        rows = int(header["nroflines"])
        cols = int(header["nrofcellsperline"])
        bin_path = self._bin_path(path)
        expected_bytes = rows * cols * file_dtype.itemsize
        actual_bytes = bin_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"ERS binary size mismatch: expected {expected_bytes} bytes, got {actual_bytes}"
            )
        # fromfile avoids retaining a second immutable bytes copy for large rasters.
        raw = np.fromfile(bin_path, dtype=file_dtype, count=rows * cols)
        data = raw.reshape((rows, cols))
        # Only mask nodata when the header declares it; a missing NullCellValue
        # must NOT silently turn legitimate zeros into NaN.
        if "nullcellvalue" in header:
            nodata = float(header["nullcellvalue"])
            data = np.where(np.isnan(data) if np.isnan(nodata) else data == nodata, np.nan, data)
        elif not np.issubdtype(dtype, np.floating):
            data = data.astype(np.float64)
        return data

    def _write_raster(self, path: Path, data: np.ndarray, header: dict[str, str]) -> None:
        dtype = self._dtype(header)
        nodata = float(header.get("nullcellvalue", "0"))
        out = np.where(np.isnan(data), nodata, data).astype(dtype)
        bin_path = self._bin_path(path)
        bin_path.parent.mkdir(parents=True, exist_ok=True)
        out.tofile(bin_path)
