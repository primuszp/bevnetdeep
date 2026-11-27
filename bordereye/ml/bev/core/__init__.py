"""Stable, lazily loaded public API for the BEV neural-network core."""

from __future__ import annotations

from ...._lazy import lazy_accessors

_LAZY = {
    "BevNet": "nn_backbones",
    "BevNetDeep": "nn_backbones",
    "build_bev_model": "model_factory",
    "STRIDE": "nn_contract",
    "REG_DIM": "nn_contract",
    "world_to_pixel": "nn_contract",
    "channel_names": "nn_contract",
    "channel_scale": "nn_contract",
    "cue_indices": "nn_contract",
    "unpack_model_output": "nn_contract",
    "box_corners": "nn_contract",
    "focal_loss": "losses",
    "reg_l1": "losses",
    "angle_loss": "losses",
    "load_bevnet_checkpoint": "inference",
    "checkpoint_channel_names": "inference",
    "window_starts": "inference",
    "run_sliding_window": "inference",
    "decode_window": "inference",
    "nms": "inference",
    "match_detections_one_to_one": "inference",
}


__getattr__, __dir__ = lazy_accessors(__name__, _LAZY, globals())


__all__ = sorted(_LAZY)
