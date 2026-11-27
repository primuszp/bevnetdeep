"""Dataset-specific orientation conventions for the shared BEV detector."""
from __future__ import annotations

# WiSAR3D's official evaluator treats these categories as invariant under a
# 180-degree rotation.  The remaining oriented categories require a directed
# 0..2π yaw; the final five classes have no orientation metric at all.
WISAR_PI_SYMMETRIC_CLASS_IDS = frozenset({1, 3, 4, 6, 7, 11, 12, 19, 21})
WISAR_NO_ORIENTATION_CLASS_IDS = frozenset({2, 9, 13, 17, 18})


def use_doubled_yaw(source: str, class_id: int) -> bool:
    """Whether a target/prediction uses sin(2θ), cos(2θ).

    Generic BEV data describes unoriented long axes, so it retains the historic
    doubled-angle representation.  WiSAR3D follows its official evaluator:
    only explicitly π-symmetric classes use that representation.
    """
    return source.lower() != "wisar" or int(class_id) in WISAR_PI_SYMMETRIC_CLASS_IDS
