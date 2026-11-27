"""Stateful, additive validation metrics for footprint segmentation and 3D OBBs."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bordereye.ml.bev.core.metrics import (
    incomplete_segmentation_counts,
    incomplete_segmentation_metrics,
    mean_obb_regression_metrics,
    obb_regression_metric_sums,
)
from bordereye.ml.bev.core.orientation import use_doubled_yaw

_SEG_KEYS = (
    "observed_tp", "observed_px", "predicted_px",
    "predicted_inside_allowed_px", "predicted_outside_allowed_px",
    "allowed_px", "trusted_intersection_px", "trusted_union_px",
    "trusted_target_px")
_OBB_KEYS = (
    "center_error_m_sum", "length_abs_error_m_sum", "width_abs_error_m_sum",
    "yaw_abs_error_deg_sum", "z_abs_error_m_sum", "height_abs_error_m_sum",
    "bev_iou_sum", "iou3d_sum", "yaw_count", "count")


@dataclass
class ValidationMetricAccumulator:
    """Accumulate unbiased pixel/object counts across validation batches."""

    seg_counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in _SEG_KEYS})
    obb_sums: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in _OBB_KEYS})
    yaw_class_sums: dict[int, float] = field(default_factory=dict)
    yaw_class_counts: dict[int, int] = field(default_factory=dict)

    def update(self, pred_seg_logits: torch.Tensor, seg_target: torch.Tensor,
               seg_weight: torch.Tensor, pred_reg: torch.Tensor,
               reg_target: torch.Tensor, reg_mask: torch.Tensor,
               *, cell: float, stride: int, threshold: float = 0.5,
               class_ids: torch.Tensor | None = None,
               orientation_source: str = "generic",
               no_orientation_ids=()) -> None:
        pred = (torch.sigmoid(pred_seg_logits) >= threshold).detach().cpu().numpy()
        target = (seg_target > 0.5).detach().cpu().numpy()
        weight = seg_weight.detach().cpu().numpy()
        for i in range(pred.shape[0]):
            observed = target[i, 0] & (weight[i, 0] >= 0.999)
            allowed = target[i, 0] | (weight[i, 0] <= 1e-6)
            counts = incomplete_segmentation_counts(
                pred[i, 0], observed, allowed, trusted_target=target[i, 0])
            for key, value in counts.items():
                self.seg_counts[key] += int(value)
        sums = obb_regression_metric_sums(
            pred_reg, reg_target, reg_mask, cell=cell, stride=stride,
            class_ids=class_ids, orientation_source=orientation_source,
            no_orientation_ids=no_orientation_ids)
        for key, value in sums.items():
            self.obb_sums[key] += float(value)
        if class_ids is not None:
            active = (reg_mask[:, 0] if reg_mask.ndim == 4 else reg_mask) > 0
            for class_id in no_orientation_ids:
                active &= class_ids != int(class_id)
            if bool(active.any()):
                ps, pc = pred_reg[:, 4][active], pred_reg[:, 5][active]
                ts, tc = reg_target[:, 4][active], reg_target[:, 5][active]
                pn = torch.sqrt(ps * ps + pc * pc).clamp_min(1e-6)
                tn = torch.sqrt(ts * ts + tc * tc).clamp_min(1e-6)
                encoded = torch.acos(
                    ((ps / pn) * (ts / tn) + (pc / pn) * (tc / tn)).clamp(-1.0, 1.0)
                ) * (180.0 / torch.pi)
                active_classes = class_ids[active]
                for class_id in torch.unique(active_classes).tolist():
                    selected = active_classes == int(class_id)
                    factor = 2.0 if use_doubled_yaw(orientation_source, class_id) else 1.0
                    self.yaw_class_sums[int(class_id)] = self.yaw_class_sums.get(
                        int(class_id), 0.0) + float((encoded[selected] / factor).sum().item())
                    self.yaw_class_counts[int(class_id)] = self.yaw_class_counts.get(
                        int(class_id), 0) + int(selected.sum().item())

    def compute(self) -> tuple[dict[str, float], dict[str, float]]:
        obb = mean_obb_regression_metrics(self.obb_sums)
        present = [
            self.yaw_class_sums[class_id] / count
            for class_id, count in self.yaw_class_counts.items() if count > 0
        ]
        if present:
            obb["yaw_abs_error_deg_macro"] = float(sum(present) / len(present))
            obb["yaw_macro_class_count"] = len(present)
        return incomplete_segmentation_metrics(self.seg_counts), obb
