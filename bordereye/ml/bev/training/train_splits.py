"""Training argument validation and dataset split helpers."""
from __future__ import annotations

from functools import cache
import json
from pathlib import Path

import numpy as np

from bordereye.ml.bev.core.data import source_weight_lookup
from bordereye.ml.bev.core.nn_contract import channel_names, output_stride_for_arch


def manifest_train_val_split(paths: list[Path], manifest: str | Path) -> tuple[list[Path], list[Path]]:
    """Use an exhaustive fixed split, rejecting missing, duplicate and test tiles."""
    data = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
    train, val = data["train_tiles"], data["val_tiles"]
    names = train + val
    if not train or not val or any(not isinstance(n, str) or Path(n).name != n for n in names):
        raise ValueError("Split manifest requires nonempty lists of tile basenames")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate or overlapping split manifest tiles")
    lookup = {p.name: p for p in paths}
    if len(lookup) != len(paths) or set(names) != set(lookup):
        raise ValueError("Split manifest must match all eligible dataset tiles exactly")
    if any(tile_is_test_split(lookup[n]) for n in names):
        raise ValueError("Split manifest cannot include test tiles")
    return [lookup[n] for n in train], [lookup[n] for n in val]


def validate_train_args(args) -> None:
    """Validate training options that must be coherent before IO starts."""
    if getattr(args, "epoch_resume_snapshot", False) and args.workers != 0:
        raise ValueError("Epoch resume snapshots require workers=0")
    if getattr(args, 'boundary_coordinate_augmentation', False) and (
        str(args.sources).lower() != 'wisar' or int(args.num_classes) != 22
    ):
        raise ValueError('Boundary-coordinate experiment is restricted to 22-class WiSAR')
    output_stride = output_stride_for_arch(args.arch)
    checks = [
        (args.crop > 0, "--crop must be positive"),
        (args.crop % output_stride == 0,
         f"--crop must be divisible by output stride={output_stride}"),
        (args.batch > 0, "--batch must be positive"),
        (args.epochs > 0, "--epochs must be positive"),
        (args.min_epochs >= 0, "--min-epochs must be non-negative"),
        (args.min_epochs <= args.epochs, "--min-epochs cannot exceed --epochs"),
        (args.lr > 0, "--lr must be positive"),
        (args.patience >= 0, "--patience must be non-negative"),
        (args.crops_per_epoch > 0, "--crops-per-epoch must be positive"),
        (0.0 < args.val_frac < 1.0, "--val-frac must be between 0 and 1"),
        (0.0 <= args.pos_frac <= 1.0, "--pos-frac must be between 0 and 1"),
        (args.val_crops >= 0, "--val-crops must be non-negative"),
        (args.workers >= 0, "--workers must be non-negative"),
        (args.cache_tiles > 0, "--cache-tiles must be positive"),
        (0.0 <= args.copy_paste_prob <= 1.0,
         "--copy-paste-prob must be between 0 and 1"),
        (args.copy_paste_max >= 0, "--copy-paste-max must be non-negative"),
        (args.reg_support_radius >= 0, "--reg-support-radius must be non-negative"),
        (args.reg_support_max_size_m >= 0,
         "--reg-support-max-size-m must be non-negative"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)


@cache
def _tile_meta(path_str: str) -> dict:
    """Read only lightweight schema/split metadata and memoise it.

    Accessing ``d['channels'].shape`` in an NPZ materialises and decompresses
    the complete raster. Prefer the short channel_names array; legacy unnamed
    tiles fall back to ch_scale before the expensive last-resort read.
    """
    p = Path(path_str)
    with np.load(p, allow_pickle=False) as d:
        names = channel_names(d)
        if names:
            n_ch = len(names)
        elif "ch_scale" in d:
            n_ch = len(d["ch_scale"])
        else:
            n_ch = int(d["channels"].shape[0])
        return {
            "source": str(d["source"]) if "source" in d else "unknown",
            "cell": round(float(d["cell"]) if "cell" in d else 0.25, 6),
            "n_ch": n_ch,
            "names": names,
            "source_split": str(d["source_split"]) if "source_split" in d else "",
        }


@cache
def _tile_label_meta(path_str: str) -> dict:
    """Read detection/mask supervision metadata only when it is requested."""
    p = Path(path_str)
    with np.load(p, allow_pickle=False) as d:
        boxes = d["boxes_valid"] if "boxes_valid" in d else d.get("boxes", np.zeros((0, 7)))
        pseudo_boxes = d["boxes_pseudo"] if "boxes_pseudo" in d else np.zeros((0, 7))
        fp_sum = float(d["fp_mask"].sum()) if "fp_mask" in d else 0.0
        seg_sum = float(d["seg_mask"].sum()) if "seg_mask" in d else 0.0
        return {
            "has_det": len(boxes) > 0,
            "has_pseudo_det": len(pseudo_boxes) > 0,
            "has_any_det": len(boxes) > 0 or len(pseudo_boxes) > 0,
            "fp_sum": fp_sum,
            "seg_sum": seg_sum,
        }


def tile_source(path: Path) -> str:
    """Return the dataset source recorded in a tile."""
    try:
        return _tile_meta(str(path))["source"]
    except Exception:
        return "unknown"


def tile_has_det_supervision(path: Path) -> bool:
    """Return true when a tile has valid or pseudo OBB supervision."""
    try:
        return _tile_label_meta(str(path))["has_any_det"]
    except Exception:
        return False


def tile_has_viz_signal(path: Path) -> bool:
    """Return true when a tile is useful for qualitative visualisation."""
    try:
        meta = _tile_label_meta(str(path))
        return bool(meta["has_any_det"] or meta["fp_sum"] > 0 or meta["seg_sum"] > 0)
    except Exception:
        return False


def tile_is_test_split(path: Path) -> bool:
    """Return true when a tile belongs to the test split.

    Checks directory name first (train/ or test/ parent), then falls back to
    the source_split key baked into the NPZ — same two-level logic as
    eval_bev.tile_split() so training and evaluation agree on the boundary.
    """
    parent = path.parent.name.lower()
    if parent in ("train", "test"):
        return parent == "test"
    try:
        return _tile_meta(str(path))["source_split"] == "test"
    except Exception:
        return False


def tile_cell(path: Path) -> float:
    """Return the BEV cell size recorded in one tile."""
    return float(_tile_meta(str(path))["cell"])


def tile_schema(path: Path) -> tuple[int, tuple[str, ...]]:
    """Return the channel count and channel-name schema for one tile."""
    meta = _tile_meta(str(path))
    return int(meta["n_ch"]), tuple(meta["names"])


def det_supervised_val_tiles(paths: list[Path], det_loss_weights: dict[str, float]) -> list[Path]:
    """Filter validation tiles to the ones that should affect detection metrics."""
    out = []
    for path in paths:
        src = tile_source(path)
        if tile_has_det_supervision(path) and source_weight_lookup(det_loss_weights, src, 1.0) > 0:
            out.append(path)
    return out


def stratified_train_val_split(paths: list[Path], val_frac: float,
                               seed: int) -> tuple[list[Path], list[Path]]:
    """Split tiles per source so every sufficiently large source is represented."""
    rng = np.random.default_rng(seed)
    by_source: dict[str, list[Path]] = {}
    for path in paths:
        by_source.setdefault(tile_source(path), []).append(path)

    solo = [src for src in sorted(by_source) if len(by_source[src]) == 1]
    if solo:
        print(f"WARNING: sources with a single tile go entirely to train and are "
              f"never validated: {solo}. Add more tiles for these sources to track "
              "their validation quality.")

    train: list[Path] = []
    val: list[Path] = []
    for src in sorted(by_source):
        group = list(by_source[src])
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        n_val = round(len(group) * val_frac)
        n_val = min(max(1, n_val), len(group) - 1)
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
