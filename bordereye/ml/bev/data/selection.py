"""Lightweight parsing and filename helpers shared by BEV workflows."""

from __future__ import annotations

from pathlib import Path

SOURCE_PREFIXES = (
    "synthsparse",
    "synth",
    "dales",
    "ply",
    "eclair",
    "isprs",
    "yuto",
    "wisar",
)


def parse_csv_set(value: str) -> set[str] | None:
    """Parse a comma-separated selector; empty or ``all`` means no filter."""
    values = {item.strip() for item in str(value).split(",") if item.strip()}
    return None if not values or "all" in values else values


def source_from_filename(path: Path | str) -> str:
    """Infer a dataset source from its canonical tile filename prefix."""
    name = Path(path).name
    for prefix in SOURCE_PREFIXES:
        if name.startswith(prefix + "_"):
            return "dales" if prefix == "ply" else prefix
    return "unknown"
