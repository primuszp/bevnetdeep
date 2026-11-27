"""Evaluation metrics that distinguish incomplete observations from footprint errors."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from bordereye.geometry import polygon_iou

from .nn_contract import box_corners
from .orientation import use_doubled_yaw

WISAR_MIN_RECALL = 0.1
WISAR_MIN_PRECISION = 0.1


def wisar_official_vehicle_ap(
    ground_truth: dict[str, np.ndarray],
    predictions: list[tuple[str, float, float, float]],
    distance_threshold: float,
) -> dict[str, float | int]:
    """Official WiSAR3D center-distance AP restricted to vehicles."""
    npos = sum(len(boxes) for boxes in ground_truth.values())
    taken: dict[str, set[int]] = defaultdict(set)
    tp, fp = [], []
    for frame, _score, cx, cy in predictions:
        gt = ground_truth.get(frame, np.empty((0, 2), np.float64))
        available = [index for index in range(len(gt)) if index not in taken[frame]]
        if available:
            distances = np.linalg.norm(gt[available] - np.asarray([cx, cy]), axis=1)
            best_local = int(np.argmin(distances))
            best_index = available[best_local]
            is_match = float(distances[best_local]) < distance_threshold
        else:
            best_index, is_match = -1, False
        if is_match:
            taken[frame].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp or sum(tp) == 0 or npos == 0:
        return {"threshold_m": distance_threshold, "ap": 0.0, "tp": 0,
                "fp": len(predictions), "gt": npos}
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / float(npos)
    recall_grid = np.linspace(0.0, 1.0, 101)
    precision_interp = np.interp(recall_grid, recall, precision, right=0.0)
    clipped = precision_interp[round(100 * WISAR_MIN_RECALL) + 1:] - WISAR_MIN_PRECISION
    clipped[clipped < 0.0] = 0.0
    return {
        "threshold_m": distance_threshold,
        "ap": float(np.mean(clipped) / (1.0 - WISAR_MIN_PRECISION)),
        "tp": int(tp_cum[-1]),
        "fp": int(fp_cum[-1]),
        "gt": npos,
    }


def incomplete_segmentation_counts(predicted, observed, allowed, trusted_target=None) -> dict[str, int]:
    """Return additive counts for partial-label-aware segmentation metrics.

    Predictions inside ``allowed`` are geometrically plausible. They contribute
    to footprint fill but never to outside false positives. ``trusted_target``
    may be a complete human/strong footprint; otherwise only observed recall and
    containment are claimed as ground-truth metrics.
    """
    pred = np.asarray(predicted, bool)
    obs = np.asarray(observed, bool)
    allow = np.asarray(allowed, bool)
    if pred.shape != obs.shape or pred.shape != allow.shape:
        raise ValueError("predicted, observed, and allowed masks must share a shape")
    allow |= obs
    trusted = obs if trusted_target is None else np.asarray(trusted_target, bool)
    if trusted.shape != pred.shape:
        raise ValueError("trusted_target must share the prediction shape")
    return {
        "observed_tp": int((pred & obs).sum()),
        "observed_px": int(obs.sum()),
        "predicted_px": int(pred.sum()),
        "predicted_inside_allowed_px": int((pred & allow).sum()),
        "predicted_outside_allowed_px": int((pred & ~allow).sum()),
        "allowed_px": int(allow.sum()),
        "trusted_intersection_px": int((pred & trusted).sum()),
        "trusted_union_px": int((pred | trusted).sum()),
        "trusted_target_px": int(trusted.sum()),
    }


def incomplete_segmentation_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Convert additive counts into interpretable partial-label metrics."""
    c = counts
    pred = max(int(c["predicted_px"]), 1)
    obs = max(int(c["observed_px"]), 1)
    allowed = max(int(c["allowed_px"]), 1)
    union = max(int(c["trusted_union_px"]), 1)
    return {
        "observed_recall": int(c["observed_tp"]) / obs,
        "obb_containment": int(c["predicted_inside_allowed_px"]) / pred,
        "outside_allowed_fraction": int(c["predicted_outside_allowed_px"]) / pred,
        "footprint_fill_ratio": int(c["predicted_inside_allowed_px"]) / allowed,
        "trusted_iou": int(c["trusted_intersection_px"]) / union,
    }


def obb_regression_metric_sums(
    pred, target, mask, *, cell: float, stride: int,
    class_ids=None, orientation_source: str = "generic",
    no_orientation_ids=(),
) -> dict[str, float]:
    """Decode supervised regression cells and return additive BEV/3D metrics.

    The function is validation-only and intentionally returns sums plus count so
    batches and distributed workers can be aggregated without averaging bias.
    Generic angles use the historic doubled-angle representation.  WiSAR
    classes are decoded according to their official directed/pi-symmetric
    contract. ``z`` is interpreted as box centre height.
    """
    p = pred.detach().float().cpu().numpy()
    t = target.detach().float().cpu().numpy()
    m = mask.detach().float().cpu().numpy()
    cls = (class_ids.detach().long().cpu().numpy() if class_ids is not None else None)
    out = {k: 0.0 for k in (
        "center_error_m_sum", "length_abs_error_m_sum", "width_abs_error_m_sum",
        "yaw_abs_error_deg_sum", "z_abs_error_m_sum", "height_abs_error_m_sum",
        "bev_iou_sum", "iou3d_sum", "yaw_count")}
    count = 0
    for bi, iy, ix in np.argwhere(m[:, 0] > 0):
        pv, tv = p[bi, :, iy, ix], t[bi, :, iy, ix]
        if not np.isfinite(pv).all() or not np.isfinite(tv).all():
            continue
        scale = float(cell) * int(stride)
        pc = np.array([(ix + pv[0]) * scale, (iy + pv[1]) * scale])
        tc = np.array([(ix + tv[0]) * scale, (iy + tv[1]) * scale])
        pL, pW = np.exp(np.clip(pv[2:4], np.log(0.1), np.log(100.0)))
        tL, tW = np.exp(np.clip(tv[2:4], np.log(0.1), np.log(100.0)))
        class_id = int(cls[bi, iy, ix]) if cls is not None else 0
        yaw_factor = 2.0 if use_doubled_yaw(orientation_source, class_id) else 1.0
        pth = np.arctan2(pv[4], pv[5]) / yaw_factor
        tth = np.arctan2(tv[4], tv[5]) / yaw_factor
        period = 180.0 if yaw_factor == 2.0 else 360.0
        yaw = abs(np.rad2deg(pth - tth)) % period
        yaw = min(yaw, period - yaw)
        pverts = box_corners(pc[0], pc[1], pL, pW, pth)
        tverts = box_corners(tc[0], tc[1], tL, tW, tth)
        bev_iou = float(polygon_iou(pverts, tverts))
        inter_area = bev_iou * (pL * pW + tL * tW) / max(1.0 + bev_iou, 1e-9)
        ph, th = max(float(pv[7]), 0.0), max(float(tv[7]), 0.0)
        pz0, pz1 = float(pv[6]) - ph / 2, float(pv[6]) + ph / 2
        tz0, tz1 = float(tv[6]) - th / 2, float(tv[6]) + th / 2
        inter_h = max(0.0, min(pz1, tz1) - max(pz0, tz0))
        inter_v = inter_area * inter_h
        union_v = pL * pW * ph + tL * tW * th - inter_v
        out["center_error_m_sum"] += float(np.linalg.norm(pc - tc))
        out["length_abs_error_m_sum"] += abs(float(pL - tL))
        out["width_abs_error_m_sum"] += abs(float(pW - tW))
        if class_id not in no_orientation_ids:
            out["yaw_abs_error_deg_sum"] += float(yaw)
            out["yaw_count"] += 1
        out["z_abs_error_m_sum"] += abs(float(pv[6] - tv[6]))
        out["height_abs_error_m_sum"] += abs(float(pv[7] - tv[7]))
        out["bev_iou_sum"] += bev_iou
        out["iou3d_sum"] += inter_v / max(union_v, 1e-9)
        count += 1
    out["count"] = count
    return out


def mean_obb_regression_metrics(sums: dict[str, float]) -> dict[str, float]:
    """Convert :func:`obb_regression_metric_sums` output to mean metrics."""
    n = max(int(sums.get("count", 0)), 1)
    result = {k.removesuffix("_sum"): float(v) / n
              for k, v in sums.items() if k.endswith("_sum")}
    result["yaw_abs_error_deg"] = float(sums.get("yaw_abs_error_deg_sum", 0.0)) / max(
        int(sums.get("yaw_count", sums.get("count", 0))), 1)
    result["count"] = int(sums.get("count", 0))
    result["yaw_count"] = int(sums.get("yaw_count", sums.get("count", 0)))
    return result
