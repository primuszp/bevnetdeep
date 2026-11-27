"""BEV-specific data roots, layered on the shared workspace layout."""

from __future__ import annotations

import os
from pathlib import Path

from bordereye.paths import derived_root, library_root, project_root
from bordereye.paths import output_root as workspace_output_root

__all__ = [
    "config_root",
    "dataset_root",
    "library_root",
    "output_root",
    "package_root",
    "project_root",
    "user_data_root",
]


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_root() -> Path:
    return package_root() / "configs"


def user_data_root() -> Path:
    """Writable per-user location for installed-package datasets and models."""
    configured = os.environ.get("BORDEREYE_DATA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]).expanduser().resolve() / "BorderEye"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base.expanduser().resolve() / "bordereye"


def _source_data(path: Path) -> Path | None:
    """Return a source-checkout asset only when it is actually present."""
    return path if path.exists() else None


def output_root() -> Path:
    configured = os.environ.get("BORDEREYE_BEV_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace_output_root() / "bev"


def dataset_root() -> Path:
    configured = os.environ.get("BORDEREYE_BEV_DATASET_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    # The workspace location wins; the in-checkout copy is the legacy layout.
    source = _source_data(derived_root() / "bev_dataset_v7") or _source_data(
        library_root() / "data" / "bev_dataset_v7"
    )
    return source or user_data_root() / "datasets" / "bev_dataset_v7"
