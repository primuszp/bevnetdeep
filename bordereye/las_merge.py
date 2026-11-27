"""Streaming LAS/LAZ merge with metadata and extra-dimension preservation."""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from .io import _require_laspy


@dataclass(frozen=True)
class MergeReport:
    output_file: Path
    input_files: int
    total_points: int
    extra_dim_conflicts: dict[str, int]


def _extra_signature(dimension) -> tuple:
    """The fields that determine whether two LAS extra dims are compatible."""
    scales = None if dimension.scales is None else tuple(map(float, dimension.scales))
    offsets = None if dimension.offsets is None else tuple(map(float, dimension.offsets))
    return (np.dtype(dimension.dtype).str, dimension.num_elements, scales, offsets,
            dimension.description or "")


def _crs_signature(header) -> tuple[tuple[str, int, bytes], ...]:
    """Raw projection VLRs; avoids requiring pyproj just to compare metadata."""
    projection_records = [
        vlr for vlr in (*header.vlrs, *(header.evlrs or ()))
        if vlr.user_id == "LASF_Projection"
    ]
    return tuple(sorted(
        (vlr.user_id, int(vlr.record_id), bytes(vlr.record_data_bytes()))
        for vlr in projection_records
    ))


def _safe_name(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")[:32] or "extra_dim"
    candidate, number = base, 2
    while candidate in used:
        suffix = f"_{number}"
        candidate = base[:32 - len(suffix)] + suffix
        number += 1
    used.add(candidate)
    return candidate


def merge_las(
    inputs: Iterable[Path | str],
    output: Path | str,
    *,
    chunk_size: int = 2_000_000,
    union_extra_dimensions: bool = True,
    overwrite: bool = False,
) -> MergeReport:
    """Merge LAS/LAZ files in bounded memory while preserving their schema.

    Core point format, LAS version, scales and offsets must match.  Extra
    dimensions are unioned; same-name type conflicts receive deterministic
    suffixes such as ``feature_2`` instead of being silently discarded.
    """
    laspy = _require_laspy()

    paths = [Path(path) for path in inputs]
    destination = Path(output)
    if not paths:
        raise ValueError("inputs must contain at least one LAS/LAZ file")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    resolved_destination = destination.resolve()
    if any(path.resolve() == resolved_destination for path in paths):
        raise ValueError("output must not be one of the input files")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(paths[0]) as reader:
        header = reader.header.copy()
        core = (
            header.version,
            header.point_format.id,
            tuple(header.scales),
            tuple(header.offsets),
            _crs_signature(header),
        )

    # Map each (source name, signature) pair to one output field name.
    definitions: dict[tuple[str, tuple], object] = {}
    per_file: dict[Path, dict[str, tuple]] = {}
    by_name: dict[str, set[tuple]] = {}
    for path in paths:
        with laspy.open(path) as reader:
            current = reader.header
            current_core = (
                current.version,
                current.point_format.id,
                tuple(current.scales),
                tuple(current.offsets),
                _crs_signature(current),
            )
            if current_core != core:
                raise ValueError(f"Incompatible core LAS header: {path}")
            file_dimensions: dict[str, tuple] = {}
            for dimension in current.point_format.extra_dimensions:
                signature = _extra_signature(dimension)
                file_dimensions[dimension.name] = signature
                definitions[(dimension.name, signature)] = dimension
                by_name.setdefault(dimension.name, set()).add(signature)
            per_file[path] = file_dimensions

    first_extra = per_file[paths[0]]
    if not union_extra_dimensions and any(value != first_extra for value in per_file.values()):
        raise ValueError("Extra dimensions differ; enable union_extra_dimensions")

    used = set(header.point_format.dimension_names)
    mapping: dict[tuple[str, tuple], str] = {}
    conflicts = {name: len(signatures) for name, signatures in by_name.items() if len(signatures) > 1}
    existing = {dimension.name: _extra_signature(dimension)
                for dimension in header.point_format.extra_dimensions}
    for name, signatures in by_name.items():
        for signature in sorted(signatures, key=repr):
            key = (name, signature)
            if existing.get(name) == signature:
                mapping[key] = name
                continue
            output_name = _safe_name(name if name not in used else f"{name}_2", used)
            source = definitions[key]
            dtype = np.dtype(source.dtype)
            if source.num_elements != 1:
                dtype = np.dtype((dtype, (source.num_elements,)))
            header.add_extra_dim(laspy.ExtraBytesParams(
                name=output_name,
                type=dtype,
                description=source.description or "",
                scales=source.scales,
                offsets=source.offsets,
            ))
            mapping[key] = output_name

    standard = list(header.point_format.standard_dimension_names)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    total = 0
    try:
        with laspy.open(temporary, mode="w", header=header,
                        do_compress=destination.suffix.lower() == ".laz") as writer:
            for path in paths:
                file_mapping = {name: mapping[(name, signature)]
                                for name, signature in per_file[path].items()}
                with laspy.open(path) as reader:
                    for chunk in reader.chunk_iterator(chunk_size):
                        output_chunk = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=header)
                        for name in standard:
                            output_chunk[name] = chunk[name]
                        for source_name, output_name in file_mapping.items():
                            output_chunk[output_name] = chunk[source_name]
                        writer.write_points(output_chunk)
                        total += len(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return MergeReport(destination, len(paths), total, conflicts)


def merge_pointclouds(
    input_dir: Path | str,
    output_file: Path | str,
    *,
    recursive: bool = False,
    chunk_size: int = 2_000_000,
    allow_extra_dim_union: bool = True,
) -> MergeReport:
    """Directory convenience wrapper retained from OmniEye."""
    from .tiles import list_las_files
    return merge_las(
        list_las_files(input_dir, recursive=recursive),
        output_file,
        chunk_size=chunk_size,
        union_extra_dimensions=allow_extra_dim_union,
    )
