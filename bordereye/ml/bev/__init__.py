"""BEV vehicle segmentation, OBB prediction, training, and data tools."""

from __future__ import annotations

from ..._lazy import lazy_accessors

_LAZY = {
    "BevPrediction": "predict",
    "BevPredictor": "predict",
    "predict_bev": "predict",
    "train": "workflow",
    "evaluate": "workflow",
    "detect": "workflow",
    "run": "workflow",
}

__getattr__, __dir__ = lazy_accessors(__name__, _LAZY, globals())

__all__ = sorted(_LAZY)
