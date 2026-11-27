"""Versioned output contract shared by BEV dataset builders."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

DATASET_VERSION = "bev_dataset_v7"
BEV_SCHEMA = "generic_v2"
MASK_POLICY = "fp_mask_v2"
PSEUDO_OBB_POLICY = "mask_length_prior_v1"


def save_args(config: dict, pseudo_policy: str = "") -> SimpleNamespace:
    """Adapt a builder configuration to the stable NPZ writer contract."""
    return SimpleNamespace(
        cell=config["cell"],
        features=config["features"],
        dataset_version=config.get("dataset_version", DATASET_VERSION),
        bev_schema=config.get("features", BEV_SCHEMA),
        mask_policy=config.get("mask_policy", MASK_POLICY),
        pseudo_obb_policy=pseudo_policy,
        npz_compress=bool(config.get("npz_compress", True)),
    )


def should_skip_existing(path: Path, config: dict) -> bool:
    return bool(config.get("skip_existing", False)) and Path(path).exists()
