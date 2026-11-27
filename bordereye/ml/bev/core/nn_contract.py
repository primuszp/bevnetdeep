"""Stable BEV neural-network contract and channel metadata."""
from __future__ import annotations

import numpy as np
import torch

from bordereye.geometry import rect_corners_2d

STRIDE = 2
REG_DIM = 8
MODEL_ARCHES = ("v1", "v1_hr", "v1_hr_classreg", "v1_hr_boxdetail", "deep")


def output_stride_for_arch(arch: str) -> int:
    """Return the spatial output stride encoded by a model architecture."""
    return 1 if str(arch).lower() in ("v1_hr", "v1_hr_classreg", "v1_hr_boxdetail") else STRIDE


def select_class_regression(reg, class_ids):
    """Select the class-specific regression plane, preserving legacy tensors.

    Legacy models return ``(B,REG,H,W)``. Class-residual models return
    ``(B,C,REG,H,W)``; ``class_ids`` is ``(B,H,W)``.
    """
    if reg.ndim == 4:
        return reg
    if reg.ndim != 5:
        raise ValueError(f"regression tensor must be 4-D or 5-D, got {tuple(reg.shape)}")
    if class_ids.ndim != 3 or class_ids.shape != (reg.shape[0], reg.shape[3], reg.shape[4]):
        raise ValueError(
            f"class_ids shape {tuple(class_ids.shape)} incompatible with regression {tuple(reg.shape)}")
    index = class_ids.to(device=reg.device, dtype=torch.long)[:, None, None]
    index = index.expand(-1, 1, reg.shape[2], -1, -1)
    return reg.gather(1, index).squeeze(1)


def world_to_pixel(
    xy: np.ndarray,
    origin: np.ndarray,
    cell: float,
    height: int,
) -> np.ndarray:
    """Convert world XY coordinates to row-down BEV pixel coordinates."""
    points = np.asarray(xy)
    offset = np.asarray(origin)
    return np.column_stack(
        ((points[:, 0] - offset[0]) / cell, height - (points[:, 1] - offset[1]) / cell)
    )


LEGACY_CH_SCALE = np.array([1.0, 4.0, 10.0, 6.0, 2.0, 6.0], np.float32)
GENERIC_CH_SCALE = np.array([
    1.0, 1.0, 1.0, 3.0, 1.0,
    4.0, 2.0, 2.0, 4.0, 4.0,
    10.0, 4.0, 1.0, 1.0, 4.0, 6.0,
    8.0,
], np.float32)

CHANNEL_SCALE_BY_NAME = {
    "occ": 1.0,
    "log_count": 4.0,
    "hag_max": 10.0,
    "hag_mean": 6.0,
    "hag_std": 2.0,
    "height_step": 6.0,
    "occ_low": 1.0,
    "occ_all": 1.0,
    "occ_high": 1.0,
    "hag_max_low": 4.0,
    "hag_min_low": 4.0,
    "hag_range_low": 4.0,
    "hag_mean_low": 4.0,
    "hag_std_low": 2.0,
    "height_step_low": 4.0,
    "height_step_all": 8.0,
    "occ_0.25_0.5": 1.0,
    "occ_0.5_1.0": 1.0,
    "occ_1.0_1.5": 1.0,
    "occ_1.5_2.0": 1.0,
    "occ_2.0_3.0": 1.0,
    "occ_3.0_4.0": 1.0,
    "occ_4.0_plus": 1.0,
    "vertical_span_all": 30.0,
    "canopy_over_low": 1.0,
    "low_without_high": 1.0,
    "edt_low": 3.0,
    "blob_area_low": 4.0,
    "hole_mask": 1.0,
    "ground_occ": 1.0,
    "ground_shadow": 1.0,
    "dtm_slope": 1.0,
    "ground_edt": 3.0,
    "shadow_edt": 1.0,
    "veh_h_p05": 4.0,
    "veh_h_p50": 4.0,
    "veh_h_p95": 4.0,
    "veh_h_span": 4.0,
    "veh_density": 1.0,
    "veh_hmax": 4.0,
    "veh_dog": 2.0,
    "veh_laplacian": 2.0,
    "veh_tophat": 4.0,
    "veh_sobel": 4.0,
    "veh_h_p95_hillshade_multi": 1.0,
    "veh_h_p95_aargau_like": 1.0,
    "veh_width_hole": 1.0,
    "all_h_p95": 10.0,
    "all_density": 1.0,
    "all_hmax": 10.0,
    "all_dog": 4.0,
    "all_laplacian": 4.0,
    "all_tophat": 6.0,
    "all_sobel": 8.0,
}

CUE_CHANNEL_NAMES = (
    "ground_shadow", "shadow_edt", "veh_width_hole", "ground_edt",
    "veh_h_p95", "veh_h_span", "veh_density",
    "veh_dog", "veh_sobel", "veh_h_p95_hillshade_multi",
    "veh_h_p95_aargau_like", "all_sobel",
)


def cue_indices(names) -> tuple[int, ...]:
    """Return indices of vehicle-footprint cue channels in a schema."""
    names = tuple(str(x) for x in (names or ()))
    return tuple(i for i, n in enumerate(names) if n in CUE_CHANNEL_NAMES)


def unpack_model_output(output):
    """Return the stable ``(hm, reg, seg)`` output tuple."""
    if not isinstance(output, (tuple, list)) or len(output) < 3:
        raise ValueError("model output must contain at least hm, reg, and seg heads")
    return output[0], output[1], output[2]


def channel_names(d) -> tuple[str, ...]:
    """Return stable channel names from an NPZ/loaded dict, if present."""
    if "channel_names" not in d:
        return ()
    return tuple(str(x) for x in d["channel_names"].tolist())


def channel_scale(d) -> np.ndarray:
    """Return per-channel normalization scales for a BEV tile."""
    n_ch = int(d["channels"].shape[0])
    names = channel_names(d)
    if names:
        if len(names) != n_ch:
            raise ValueError(f"channel_names length {len(names)} != channels {n_ch}")
        missing = [name for name in names if name not in CHANNEL_SCALE_BY_NAME]
        if missing:
            raise ValueError(f"Unknown BEV channel name(s): {missing}")
        return np.asarray([CHANNEL_SCALE_BY_NAME[name] for name in names], np.float32)
    if "ch_scale" in d and len(d["ch_scale"]) == n_ch:
        return d["ch_scale"].astype(np.float32)
    if n_ch == len(LEGACY_CH_SCALE):
        return LEGACY_CH_SCALE
    if n_ch == len(GENERIC_CH_SCALE):
        return GENERIC_CH_SCALE
    return np.ones(n_ch, np.float32)


def box_corners(cx, cy, L, W, th_rad) -> np.ndarray:
    """Return ``(4, 2)`` float32 world-space OBB corners."""
    return rect_corners_2d(cx, cy, L, W, th_rad, dtype=np.float32)


def box_row_corners(box) -> np.ndarray:
    """Return corners for a dataset box row whose angle is stored in degrees."""
    b = np.asarray(box)
    return box_corners(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                       float(np.deg2rad(b[4])))
