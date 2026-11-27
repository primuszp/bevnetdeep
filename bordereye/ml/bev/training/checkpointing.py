"""Checkpoint selection policy and versioned supervision metadata."""
from __future__ import annotations

import math


def segmentation_contract(args) -> dict:
    """Serializable semantic contract stored in every new checkpoint."""
    return {
        "version": 2,
        "prediction": "complete_vehicle_footprint",
        "observed_mask_semantics": "trusted_positive_returns",
        "unknown_allowed_semantics": "ignore_not_background",
        "box_target_shape": "rounded_rectangle",
        "corner_radius_fraction": 0.25,
        "seg_target_policy": args.seg_target,
        "weak_hole_policy": args.weak_seg_hole_policy,
    }


def checkpoint_score(args, val_det: float, val_seg_lf: float,
                     footprint: dict[str, float], obb: dict[str, float],
                     *, has_detection_metric: bool) -> float:
    """Return the configured lower-is-better checkpoint score."""
    if not has_detection_metric:
        return float("nan")
    if args.checkpoint_metric == "det_plus_seg_lf" and math.isfinite(val_seg_lf):
        return val_det + max(0.0, float(args.checkpoint_seg_lf_weight)) * val_seg_lf
    if args.checkpoint_metric != "footprint_composite":
        return val_det
    return (
        val_det
        + max(0.0, args.checkpoint_center_weight) * obb.get("center_error_m", 0.0)
        + max(0.0, args.checkpoint_yaw_weight)
        * min(obb.get("yaw_abs_error_deg_macro", obb.get("yaw_abs_error_deg", 0.0)) / 45.0, 2.0)
        + max(0.0, args.checkpoint_bev_iou_weight) * (1.0 - obb.get("bev_iou", 0.0))
        + max(0.0, args.checkpoint_3d_iou_weight) * (1.0 - obb.get("iou3d", 0.0))
        + max(0.0, args.checkpoint_observed_recall_weight)
        * (1.0 - footprint.get("observed_recall", 0.0))
        + max(0.0, args.checkpoint_outside_weight)
        * footprint.get("outside_allowed_fraction", 0.0))
