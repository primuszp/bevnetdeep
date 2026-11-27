"""Pure ranked-detection and segmentation metrics for BEV evaluation."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from .metrics import incomplete_segmentation_counts, incomplete_segmentation_metrics
from .nn_contract import STRIDE

SEG_PARTIAL_DILATION_PX = 15


def average_precision_from_ranked(rows: list[tuple[float, bool]], gt_count: int) -> float:
    """Return all-point interpolated AP for score-ranked detections."""
    if gt_count <= 0 or not rows:
        return 0.0
    ranked = sorted(rows, key=lambda x: x[0], reverse=True)
    tp = np.cumsum([1.0 if is_tp else 0.0 for _, is_tp in ranked])
    fp = np.cumsum([0.0 if is_tp else 1.0 for _, is_tp in ranked])
    recall = tp / float(gt_count)
    precision = tp / np.maximum(tp + fp, 1.0)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1]) + 1
    return float(np.sum((mrec[changed] - mrec[changed - 1]) * mpre[changed]))


def segmentation_metrics(full_seg, seg_gt, thresh: float, eval_mask=None,
                         observed=None, allowed=None):
    """Return complete and partial-label segmentation counts and metrics."""
    coarse_pred = np.asarray(full_seg) >= thresh
    pred = np.repeat(np.repeat(coarse_pred, STRIDE, axis=0), STRIDE, axis=1)[
        :seg_gt.shape[0], :seg_gt.shape[1]
    ]
    if pred.shape != seg_gt.shape:
        raise ValueError(f"segmentation output too small: {pred.shape} for GT {seg_gt.shape}")
    partial_counts = incomplete_segmentation_counts(
        pred, seg_gt if observed is None else observed,
        seg_gt if allowed is None else allowed,
        trusted_target=seg_gt if observed is None else None)
    partial_metrics = incomplete_segmentation_metrics(partial_counts)
    if eval_mask is not None:
        em = np.asarray(eval_mask).astype(bool)
        pred, seg_gt = pred & em, seg_gt & em
    inter = int((pred & seg_gt).sum())
    union = int((pred | seg_gt).sum())
    pred_px, gt_px = int(pred.sum()), int(seg_gt.sum())
    if gt_px > 0:
        dil_gt = binary_dilation(seg_gt, iterations=SEG_PARTIAL_DILATION_PX)
        pred_in_dil = int((pred & dil_gt).sum())
        partial_iou = inter / max(pred_in_dil + gt_px - inter, 1)
    else:
        pred_in_dil, partial_iou = 0, 0.0
    return {
        "seg_inter": inter, "seg_union": union, "seg_pred_px": pred_px,
        "seg_gt_px": gt_px, "seg_pred_in_dil_px": pred_in_dil,
        "seg_iou": inter / max(union, 1),
        "seg_precision": inter / max(pred_px, 1),
        "seg_recall": inter / max(gt_px, 1), "seg_partial_iou": partial_iou,
        **{f"seg_{k}": v for k, v in partial_counts.items()},
        **{f"seg_{k}": v for k, v in partial_metrics.items()},
    }
