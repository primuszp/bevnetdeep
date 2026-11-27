"""Canonical vehicle-footprint targets for incomplete airborne LiDAR labels.

The raw ``seg_mask`` records observed vehicle returns; it is not necessarily a
complete object silhouette.  This module defines the library-wide supervision
contract used by training and evaluation:

``observed``
    Trusted positive pixels measured by the sensor.
``footprint``
    Reconstructed full vehicle support (human mask, strong mask, or OBB prior).
``allowed``
    Region where extra prediction is plausible and must not count as a false
    positive when the footprint label is incomplete.
``weight``
    Per-pixel confidence. Zero means unknown, not background.

Geometry is expressed in output-grid coordinates and uses the image convention
(x right, y down). Callers convert world yaw to image yaw before rasterizing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootprintSupervision:
    """Dense target maps sharing one ``(H, W)`` shape."""

    target: np.ndarray
    allowed: np.ndarray
    weight: np.ndarray
    observed: np.ndarray


def oriented_rounded_rect_mask(
    shape: tuple[int, int],
    cx: float,
    cy: float,
    length: float,
    width: float,
    theta: float,
    *,
    corner_radius_fraction: float = 0.25,
    margin: float = 0.0,
) -> np.ndarray:
    """Rasterize a vehicle-like rounded rectangle.

    Pixel centres are tested against the exact signed-distance construction of
    a rounded rectangle. ``corner_radius_fraction=0`` gives a hard rectangle;
    ``0.5`` approaches a capsule along the long axis.
    """
    h, w = (int(shape[0]), int(shape[1]))
    if h <= 0 or w <= 0 or length <= 0 or width <= 0:
        return np.zeros((max(h, 0), max(w, 0)), bool)
    half_l = max(0.5 * float(length) + float(margin), 1e-6)
    half_w = max(0.5 * float(width) + float(margin), 1e-6)
    radius = np.clip(float(corner_radius_fraction), 0.0, 0.5) * 2.0 * min(half_l, half_w)
    radius = min(radius, half_l, half_w)
    extent = int(np.ceil(np.hypot(half_l, half_w))) + 2
    x0, x1 = max(0, int(np.floor(cx)) - extent), min(w, int(np.ceil(cx)) + extent + 1)
    y0, y1 = max(0, int(np.floor(cy)) - extent), min(h, int(np.ceil(cy)) + extent + 1)
    out = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) + 0.5 - float(cx)
    dy = yy.astype(np.float32) + 0.5 - float(cy)
    ca, sa = float(np.cos(theta)), float(np.sin(theta))
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    qx = np.abs(lx) - (half_l - radius)
    qy = np.abs(ly) - (half_w - radius)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside_offset = np.minimum(np.maximum(qx, qy), 0.0)
    out[y0:y1, x0:x1] = outside + inside_offset <= radius
    return out


def rasterize_box_rows(
    boxes: np.ndarray | None,
    origin: np.ndarray,
    cell: float,
    shape: tuple[int, int],
    *,
    corner_radius_fraction: float = 0.0,
) -> np.ndarray:
    """Rasterize world-coordinate ``[cx,cy,L,W,yaw_deg,...]`` box rows."""
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width), bool)
    if boxes is None:
        return out
    rows = np.asarray(boxes, np.float32)
    if rows.size == 0:
        return out
    rows = rows.reshape(-1, rows.shape[-1])
    if rows.shape[1] < 5:
        raise ValueError(f"box rows need at least 5 columns, got {rows.shape}")
    origin = np.asarray(origin, np.float64)
    for box in rows:
        out |= oriented_rounded_rect_mask(
            out.shape,
            cx=(float(box[0]) - origin[0]) / cell,
            cy=height - (float(box[1]) - origin[1]) / cell,
            length=float(box[2]) / cell,
            width=float(box[3]) / cell,
            theta=-float(np.deg2rad(box[4])),
            corner_radius_fraction=corner_radius_fraction,
        )
    return out


def build_incomplete_supervision(
    observed: np.ndarray,
    footprint: np.ndarray,
    allowed: np.ndarray | None = None,
    *,
    footprint_reliability: float = 1.0,
    unknown_weight: float = 0.0,
) -> FootprintSupervision:
    """Build target/weight maps without treating missing returns as background.

    Observed pixels are always trusted positives. Reconstructed footprint pixels
    are positive with ``footprint_reliability``. Pixels inside ``allowed`` but
    outside the reconstructed footprint receive ``unknown_weight``; normally
    zero. Definite background outside ``allowed`` retains weight one.
    """
    obs = np.asarray(observed, bool)
    fp = np.asarray(footprint, bool)
    if obs.shape != fp.shape:
        raise ValueError(f"observed/footprint shape mismatch: {obs.shape} vs {fp.shape}")
    allow = fp.copy() if allowed is None else np.asarray(allowed, bool)
    if allow.shape != obs.shape:
        raise ValueError(f"allowed shape mismatch: {allow.shape} vs {obs.shape}")
    allow |= fp | obs
    target = (fp | obs).astype(np.float32)
    weight = np.ones(obs.shape, np.float32)
    uncertain = allow & ~fp & ~obs
    weight[uncertain] = float(np.clip(unknown_weight, 0.0, 1.0))
    reconstructed_only = fp & ~obs
    weight[reconstructed_only] = float(np.clip(footprint_reliability, 0.0, 1.0))
    weight[obs] = 1.0
    return FootprintSupervision(target, allow.astype(np.uint8), weight, obs.astype(np.uint8))
