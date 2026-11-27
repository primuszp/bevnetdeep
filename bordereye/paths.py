"""Workspace layout discovery.

Scripts, examples and tests locate their inputs and outputs through this
module rather than by counting ``Path(__file__).parents[...]`` levels. That
keeps the workspace rearrangeable: moving ``bordereye_lib`` or regrouping
``data/`` into per-site subdirectories changes nothing here and nothing in the
callers.

Every location honours an environment override, so an installed package or a
machine that keeps its bulk data on another volume needs no source edits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_MARKER = ".bordereye-root"


def _env_dir(name: str) -> Path | None:
    configured = os.environ.get(name)
    if not configured:
        return None
    return Path(configured).expanduser().resolve()


def project_root() -> Path:
    """Return the BorderEye workspace root.

    ``BORDEREYE_PROJECT_ROOT`` is the portable, explicit override. Otherwise
    the search walks up from the caller's working directory and from this
    file, preferring the ``.bordereye-root`` marker and falling back to the
    library checkout's own layout.
    """
    configured = _env_dir("BORDEREYE_PROJECT_ROOT")
    if configured is not None:
        if not configured.is_dir():
            raise ValueError(f"BORDEREYE_PROJECT_ROOT is not a directory: {configured}")
        return configured

    candidates = (Path.cwd().resolve(), *Path.cwd().resolve().parents,
                  *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (candidate / ROOT_MARKER).is_file():
            return candidate
    for candidate in candidates:
        if candidate.name == "bordereye_lib" and (candidate / "pyproject.toml").is_file():
            return candidate.parent
    # Importing the installed package must not require a particular workspace.
    return Path.cwd().resolve()


def library_root() -> Path:
    """Return the ``bordereye_lib`` checkout containing this package."""
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """Measured inputs: the LAS/LAZ clouds, rasters and hand annotations."""
    return _env_dir("BORDEREYE_DATA_DIR") or project_root() / "data"


def work_root() -> Path:
    """Bulk regenerable material: datasets, derived tensors, models, output."""
    return _env_dir("BORDEREYE_WORK_DIR") or project_root() / "work"


def datasets_root() -> Path:
    """Third-party point-cloud datasets (DALES, ECLAIR, Sydney, ...)."""
    return _env_dir("BORDEREYE_DATASETS_DIR") or work_root() / "datasets"


def derived_root() -> Path:
    """Preprocessed tensors built from ``data_root`` and ``datasets_root``."""
    return _env_dir("BORDEREYE_DERIVED_DIR") or work_root() / "derived"


def models_root() -> Path:
    """Trained checkpoints."""
    return _env_dir("BORDEREYE_MODELS_DIR") or work_root() / "models"


_CHECKPOINT_ALIASES = {
    "v19": "bev_v19_final",
    "v19-final": "bev_v19_final",
    "v19-bootstrap": "bev_v19_bootstrap_v18",
    "v20": "bev_vehicle_12ch_v20_best",
    "v20-final": "bev_vehicle_12ch_v20_best",
    "v20-bootstrap": "bev_v20_bootstrap",
}


def retained_checkpoint_path(checkpoint_id: str) -> Path:
    """Resolve a retained checkpoint from the repository's Git LFS manifest."""
    resolved_id = _CHECKPOINT_ALIASES.get(checkpoint_id.lower(), checkpoint_id)
    manifest_path = project_root() / "models" / "checkpoints.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"retained checkpoint manifest not found: {manifest_path}"
        ) from exc
    artifacts = {item["id"]: item for item in manifest["artifacts"]}
    try:
        relative_path = artifacts[resolved_id]["local_path"]
    except KeyError as exc:
        available = ", ".join(sorted(artifacts))
        raise KeyError(
            f"unknown retained checkpoint {checkpoint_id!r}; available: {available}"
        ) from exc
    path = project_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(
            f"retained checkpoint is missing: {path}. Run 'git lfs pull'."
        )
    if path.stat().st_size < 1024 and path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/v1"
    ):
        raise FileNotFoundError(
            f"retained checkpoint is only a Git LFS pointer: {path}. "
            "Run 'git lfs pull'."
        )
    return path


def resolve_checkpoint_path(value: str | Path) -> Path:
    """Resolve an existing path or a retained-checkpoint ID/alias."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    candidate = str(value)
    if candidate.lower() in _CHECKPOINT_ALIASES:
        return retained_checkpoint_path(candidate)
    manifest_path = project_root() / "models" / "checkpoints.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(item["id"] == candidate for item in manifest["artifacts"]):
            return retained_checkpoint_path(candidate)
    return path


def output_root() -> Path:
    """Run artefacts: renders, metrics, evaluation reports."""
    return _env_dir("BORDEREYE_OUTPUT_DIR") or work_root() / "output"


def example_output(name: str = "") -> Path:
    """Return (and create) the output directory used by ``examples/``."""
    directory = output_root() / "examples"
    if name:
        directory = directory / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def data_file(name: str) -> Path:
    """Resolve a measured-input file by name, whatever the ``data/`` layout.

    Callers name the file (``"jeli_ground_mini.las"``) and stay agnostic about
    whether it sits flat in ``data/`` or in a per-site subdirectory. A path
    that already resolves is returned unchanged, so callers may still pass a
    full path or a relative one such as ``"rasters/jeli_mean_z_ground.ers"``.
    """
    root = data_root()
    direct = root / name
    if direct.exists():
        return direct
    matches = sorted(p for p in root.rglob(Path(name).name) if p.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        listing = ", ".join(str(p.relative_to(root)) for p in matches)
        raise FileNotFoundError(
            f"{name!r} is ambiguous under {root}; pass a relative path. Found: {listing}"
        )
    # Report the conventional location rather than the search failure.
    return direct
