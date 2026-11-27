"""BEV channel normalization contract.

The model must see the same numeric channel contract during training,
evaluation, and field inference.  Keep sensor/domain adaptation here instead of
scattering ad-hoc percentile logic across scripts.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

NORM_MODES = ("legacy_tile_percentile", "physical", "train_stats")


def normalize_bev_channels(
    channels: np.ndarray,
    scale: np.ndarray | None = None,
    norm: dict | None = None,
) -> np.ndarray:
    """Normalize BEV channels according to an explicit contract.

    Modes:
      legacy_tile_percentile
        Historical behavior: divide by physical scale, then map each tile and
        channel from its own 1st-99th percentile to [0, 1].
      physical
        Divide by physical scale and clip to [0, 1].  This preserves absolute
        density/height meaning across scenes.
      train_stats
        Divide by physical scale, then clip/map using train-set statistics saved
        into the checkpoint.  This is the preferred cross-sensor mode.
    """
    ch = np.asarray(channels, dtype=np.float32)
    if ch.ndim != 3:
        raise ValueError(f"normalize_bev_channels expects (C,H,W), got {ch.shape}")
    sc = _prepare_scale(scale, ch.shape[0])
    x = ch / sc[:, None, None]
    cfg = dict(norm or {})
    mode = str(cfg.get("mode", "legacy_tile_percentile"))
    if mode not in NORM_MODES:
        raise ValueError(f"unknown BEV norm mode {mode!r}; expected one of {NORM_MODES}")
    if mode == "physical":
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    if mode == "train_stats":
        lo = np.asarray(cfg.get("clip_min"), dtype=np.float32)
        hi = np.asarray(cfg.get("clip_max"), dtype=np.float32)
        if lo.shape != (ch.shape[0],) or hi.shape != (ch.shape[0],):
            raise ValueError("train_stats normalization requires clip_min/clip_max per channel")
        return _clip_and_scale(x, lo[:, None, None], hi[:, None, None])
    p1, p99 = np.percentile(
        x, (1, 99), axis=(1, 2), keepdims=True).astype(np.float32)
    return _clip_and_scale(x, p1, p99)


def load_norm_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return validate_norm_config(cfg)


def validate_norm_config(cfg: dict | None) -> dict:
    cfg = dict(cfg or {})
    mode = str(cfg.get("mode", "legacy_tile_percentile"))
    if mode not in NORM_MODES:
        raise ValueError(f"unknown BEV norm mode {mode!r}")
    out = {"mode": mode}
    if mode == "train_stats":
        for key in ("clip_min", "clip_max"):
            if key not in cfg:
                raise ValueError(f"train_stats normalization missing {key}")
            out[key] = np.asarray(cfg[key], dtype=np.float32).tolist()
        if len(out["clip_min"]) != len(out["clip_max"]):
            raise ValueError("clip_min and clip_max lengths differ")
    return out


def compute_train_stats_norm(
    npz_paths: Iterable[str | Path],
    scale_fn,
    *,
    max_tiles: int = 0,
) -> dict:
    """Estimate robust train-set normalization limits from raw BEV tiles.

    The estimator stores a robust range per channel after physical scaling.  It
    intentionally aggregates tile-level percentiles instead of all pixels, so a
    few very large or empty tiles do not dominate the contract.
    """
    paths = [Path(p) for p in npz_paths]
    if max_tiles and max_tiles > 0:
        paths = paths[: int(max_tiles)]
    if not paths:
        raise ValueError("cannot compute normalization stats from an empty path list")
    p01, p99 = [], []
    names = None
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            ch = np.asarray(d["channels"], dtype=np.float32)
            sc = _prepare_scale(scale_fn(d), ch.shape[0])
            x = ch / sc[:, None, None]
            tile_percentiles = np.percentile(
                x, (1, 99), axis=(1, 2)).astype(np.float32)
            p01.append(tile_percentiles[0])
            p99.append(tile_percentiles[1])
            if names is None and "channel_names" in d:
                names = [str(v) for v in d["channel_names"].tolist()]
    p01a = np.stack(p01, axis=0)
    p99a = np.stack(p99, axis=0)
    lo = np.percentile(p01a, 5, axis=0).astype(np.float32)
    hi = np.percentile(p99a, 95, axis=0).astype(np.float32)
    hi = np.maximum(hi, lo + 1e-6)
    out = {
        "mode": "train_stats",
        "clip_min": lo.tolist(),
        "clip_max": hi.tolist(),
        "stat_tiles": len(paths),
    }
    if names:
        out["channel_names"] = names
    return out


def _prepare_scale(scale: np.ndarray | None, n_ch: int) -> np.ndarray:
    if scale is None:
        return np.ones(n_ch, np.float32)
    sc = np.asarray(scale, dtype=np.float32)
    if sc.shape[0] != n_ch:
        raise ValueError(f"scale length {sc.shape[0]} != channel count {n_ch}")
    return np.maximum(sc, 1e-6)


def _clip_and_scale(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    x = np.clip(x, lo, hi)
    return ((x - lo) / np.maximum(hi - lo, 1e-6)).astype(np.float32)
