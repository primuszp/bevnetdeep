"""Rule-based vehicle type classification.

Each VehicleClass defines a set of geometric constraints. classify_vehicle()
returns the best matching class or None if no class matches.

The rules are intentionally conservative (overlapping ranges are allowed) so
that a soft scoring approach can resolve ambiguous candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

from .features import CandidateFeatures


@dataclass(frozen=True)
class VehicleClass:
    """Geometric profile for one vehicle type."""

    name: str
    """Human-readable label, e.g. "car", "truck"."""

    min_length: float
    max_length: float
    min_width: float
    max_width: float
    min_height: float
    max_height: float
    max_elongatedness: float
    """Maximum length/width ratio (very elongated = truck/bus)."""
    min_rectangularity: float
    """Minimum OBB/hull area ratio (round objects score low)."""

    def score(self, f: CandidateFeatures) -> float:
        """Return a [0, 1] match score; 0 means any hard constraint violated.

        Height constraints are skipped when f.height is NaN (e.g. detections
        from ground-raster holes where no vertical info is available).
        """
        import math
        if not (self.min_length <= f.length <= self.max_length):
            return 0.0
        if not (self.min_width <= f.width <= self.max_width):
            return 0.0
        height_known = not math.isnan(f.height)
        if height_known and not (self.min_height <= f.height <= self.max_height):
            return 0.0
        if f.elongatedness > self.max_elongatedness:
            return 0.0
        if f.rectangularity < self.min_rectangularity:
            return 0.0

        def range_score(v: float, lo: float, hi: float) -> float:
            mid = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            return max(0.0, 1.0 - abs(v - mid) / half) if half > 0 else 1.0

        s = (
            range_score(f.length, self.min_length, self.max_length)
            * range_score(f.width, self.min_width, self.max_width)
        )
        if height_known:
            s *= range_score(f.height, self.min_height, self.max_height)
        return float(s)


# ─── Vehicle class registry ──────────────────────────────────────────────────
# Dimensions in metres; derived from typical European vehicle statistics.
# Ordered from most specific to least so that the first unambiguous match wins.

VEHICLE_CLASSES: list[VehicleClass] = [
    VehicleClass(
        name="motorcycle",
        min_length=1.2,  max_length=2.5,
        min_width=0.4,   max_width=1.0,
        min_height=0.8,  max_height=1.8,
        max_elongatedness=5.0,
        min_rectangularity=0.35,
    ),
    VehicleClass(
        name="car",
        min_length=3.5,  max_length=5.5,
        min_width=1.4,   max_width=2.3,
        min_height=1.1,  max_height=2.0,
        max_elongatedness=3.5,
        min_rectangularity=0.60,
    ),
    VehicleClass(
        name="van",
        min_length=4.5,  max_length=7.5,
        min_width=1.7,   max_width=2.6,
        min_height=1.8,  max_height=3.2,
        max_elongatedness=4.5,
        min_rectangularity=0.65,
    ),
    VehicleClass(
        name="truck",
        min_length=5.0,  max_length=16.5,
        min_width=1.9,   max_width=4.5,
        min_height=2.0,  max_height=4.5,
        max_elongatedness=9.0,
        min_rectangularity=0.60,
    ),
    VehicleClass(
        name="bus",
        min_length=9.0,  max_length=18.0,
        min_width=2.4,   max_width=4.0,
        min_height=2.5,  max_height=4.5,
        max_elongatedness=9.0,
        min_rectangularity=0.65,
    ),
]


def classify_vehicle(
    features: CandidateFeatures,
    *,
    classes: list[VehicleClass] | None = None,
) -> VehicleClass | None:
    """Return the best-matching vehicle class for the given shape features.

    Args:
        features: Shape descriptors from extract_features().
        classes: Override the default VEHICLE_CLASSES list.

    Returns:
        The VehicleClass with the highest non-zero score, or None if no class
        passes its hard geometric constraints.
    """
    registry = classes if classes is not None else VEHICLE_CLASSES
    best_class: VehicleClass | None = None
    best_score = 0.0

    for vc in registry:
        s = vc.score(features)
        if s > best_score:
            best_score = s
            best_class = vc

    return best_class


def classify_vehicle_name(
    features: CandidateFeatures,
    *,
    classes: list[VehicleClass] | None = None,
) -> str | None:
    """Convenience wrapper — returns just the class name or None."""
    result = classify_vehicle(features, classes=classes)
    return result.name if result is not None else None
