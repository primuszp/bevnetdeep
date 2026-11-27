"""Geometric feature extraction for detected object candidates.

These features describe the shape and density of a GridCandidate and are
suitable as input to rule-based vehicle classification or downstream ML.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import GridCandidate
from .geometry import convex_hull_2d, polygon_area


@dataclass(frozen=True)
class CandidateFeatures:
    """Dimensionless shape descriptors computed from a GridCandidate."""

    length: float
    """Longest horizontal extent of the oriented bounding box (metres)."""

    width: float
    """Shortest horizontal extent of the oriented bounding box (metres)."""

    height: float
    """Vertical extent: value_max - value_min for raster, voxel Z-span for voxel (metres)."""

    elongatedness: float
    """length / width. 1.0 = square, >3 = elongated (truck-like)."""

    rectangularity: float
    """OBB footprint area / convex-hull footprint area. 1.0 = perfect rectangle."""

    compactness: float
    """Fraction of the bounding box volume actually occupied by the cluster.
    = cell_count * cell_volume / (length * width * height). 1.0 = fully dense."""

    point_density: float
    """point_count / (length * width * height). Points per cubic metre."""

    aspect_hw: float
    """height / width. Tall, narrow objects have high values."""

    footprint_area: float
    """OBB length * width in square metres."""

    volume: float
    """OBB length * width * height in cubic metres."""


def extract_features(
    candidate: GridCandidate,
    *,
    cell_xy: np.ndarray | None = None,
    hull_area: float | None = None,
) -> CandidateFeatures:
    """Compute shape features for a single GridCandidate.

    Args:
        candidate: A candidate produced by extract_raster_candidates or
            extract_voxel_candidates (must have obb_2d set).
        cell_xy: World-space XY coordinates of the segment cells (N,2).
            Used for accurate convex hull computation. If None, hull is
            approximated from cell_count.
        hull_area: Pre-computed 2D convex hull area in m². Overrides
            cell_xy-based hull computation if provided.
    """
    L = candidate.length
    W = candidate.width
    H = max(candidate.height, 1e-6)
    L = max(L, 1e-6)
    W = max(W, 1e-6)

    footprint = L * W
    vol = footprint * H

    elongatedness = L / W
    aspect_hw = H / W
    point_density = candidate.point_count / max(vol, 1e-6)

    # Convex hull area — from actual cell centroids if available
    if hull_area is not None and hull_area > 0:
        ch_area = hull_area
        perimeter = None
    elif cell_xy is not None and len(cell_xy) >= 3:
        ch = convex_hull_2d(cell_xy)
        if len(ch) >= 3:
            # Keep core feature extraction dependency-free.  Convex hull
            # perimeter and area are trivial vectorised reductions here;
            # importing Shapely for every candidate was both slow and an
            # undeclared runtime dependency.
            ch_area = max(polygon_area(ch), 1e-9)
            perimeter = float(np.linalg.norm(ch - np.roll(ch, 1, axis=0), axis=1).sum())
        else:
            ch_area = max(candidate.cell_count * _cell_area(candidate), 1e-9)
            perimeter = None
    else:
        ch_area = max(candidate.cell_count * _cell_area(candidate), 1e-9)
        perimeter = None

    # Rectangularity: convex hull area / OBB area (Eum et al. 2017)
    # Higher = more rectangular. Always <= 1 since hull <= OBB.
    rectangularity = min(1.0, ch_area / max(footprint, 1e-9))

    # Compactness: perimeter^2 / hull_area (isoperimetric quotient, inverse)
    # Circle = 4*pi ~12.6; rectangle ~16; irregular/branchy objects >> 16
    if perimeter is not None:
        compactness = (perimeter ** 2) / ch_area
    else:
        compactness = min(1.0, candidate.cell_count / max(footprint / _cell_area(candidate), 1e-6))

    return CandidateFeatures(
        length=L,
        width=W,
        height=H,
        elongatedness=elongatedness,
        rectangularity=rectangularity,
        compactness=compactness,
        point_density=point_density,
        aspect_hw=aspect_hw,
        footprint_area=footprint,
        volume=vol,
    )


def _cell_area(candidate: GridCandidate) -> float:
    """Return the world-area of one grid cell for this candidate."""
    if candidate.source == "raster":
        # Approximate from footprint/cell_count (avoids importing the grid)
        return candidate.footprint_area / max(candidate.cell_count, 1)
    else:  # voxel
        dims = candidate.dimensions
        cell_vol = (dims[0] / max(candidate.cell_count, 1)) ** (1 / 3)
        return cell_vol ** 2
