"""Greedy non-maximum suppression for rotated 2-D boxes.

Both the blob/hole detector and the rectangle search suppress overlapping
detections the same way, so the algorithm lives here once. The bounding-circle
prefilter matters: polygon clipping is comparatively expensive, and most pairs
in a tile are far enough apart that they cannot overlap at all.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from .geometry import polygon_iou

T = TypeVar("T")

# A corner can sit a few ULP outside its own bounding circle once the circle
# radius is derived from length/width rather than the corners themselves. Grow
# the prefilter radius so it can never reject a genuinely overlapping pair.
_RADIUS_SLACK = 1e-9


def greedy_rotated_nms(
    items: Sequence[T],
    iou_threshold: float,
    *,
    corners: Callable[[T], np.ndarray],
    circle: Callable[[T], tuple[float, float, float]],
    key: Callable[[T], Any],
) -> list[T]:
    """Keep the highest-ranked item of every overlapping group.

    Args:
        items: detections to suppress.
        iou_threshold: an item is dropped when it overlaps an already accepted
            item by more than this.
        corners: item -> (4, 2) world-XY corner array.
        circle: item -> (centre x, centre y, bounding-circle radius).
        key: sort key; items are processed in descending key order.

    Returns:
        The accepted items, best first.
    """
    kept: list[T] = []
    # Corners and bounding circles are computed once per accepted item and
    # reused for every later comparison against it.
    accepted: list[tuple[np.ndarray, float, float, float]] = []
    for item in sorted(items, key=key, reverse=True):
        cx, cy, radius = circle(item)
        item_corners = corners(item)
        if not any(
            np.hypot(cx - kx, cy - ky)
            <= (radius + k_radius) * (1.0 + _RADIUS_SLACK) + _RADIUS_SLACK
            and polygon_iou(item_corners, k_corners) > iou_threshold
            for k_corners, kx, ky, k_radius in accepted
        ):
            kept.append(item)
            accepted.append((item_corners, cx, cy, radius))
    return kept
