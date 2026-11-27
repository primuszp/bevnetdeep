"""Conservative segmentation-component to OBB conversion for evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi

from bordereye.geometry import oriented_bounding_box_2d, polygon_iou


@dataclass(frozen=True)
class PseudoObbConfig:
    seg_threshold: float = 0.45
    min_area_m2: float = 1.0
    max_area_m2: float = 100.0
    min_model_iou: float = 0.25
    max_center_dist_m: float = 2.5
    min_score: float = 0.10


def _det_box(det):
    if isinstance(det, dict):
        return np.array([
            det.get("cx", 0.0), det.get("cy", 0.0), det.get("z", 0.0),
            det.get("length", det.get("L", 0.0)),
            det.get("width", det.get("W", 0.0)), det.get("height", 0.0),
            det.get("theta", det.get("th", 0.0)),
        ])
    arr = np.asarray(det, np.float64).reshape(-1)
    return arr[:7] if arr.size >= 7 else None


def pseudo_obbs_from_segmentation(segmentation, detections, origin, cell,
                                   *, cfg: PseudoObbConfig | None = None):
    """Fit OBBs to mask components and retain only detector-supported objects."""
    cfg = cfg or PseudoObbConfig()
    seg = np.asarray(segmentation, np.float32)
    if seg.ndim != 2:
        raise ValueError(f"segmentation must be 2D, got {seg.shape}")
    mask = ndi.binary_opening(seg >= cfg.seg_threshold, iterations=1)
    labels, count = ndi.label(mask)
    dets = [d for d in (_det_box(x) for x in detections) if d is not None]
    out = []
    H = seg.shape[0]
    for component_id in range(1, count + 1):
        rows, cols = np.nonzero(labels == component_id)
        area = len(rows) * cell * cell
        if not cfg.min_area_m2 <= area <= cfg.max_area_m2 or len(rows) < 3:
            continue
        xy = np.column_stack([
            origin[0] + (cols + 0.5) * cell,
            origin[1] + (H - rows - 0.5) * cell,
        ])
        obb = oriented_bounding_box_2d(xy)
        center = np.asarray(obb.center, np.float64)
        length, width = float(obb.length), float(obb.width)
        theta = float(np.deg2rad(obb.rotation_deg))
        supporting = []
        for source_det, det in zip(detections, dets, strict=True):
            center_ok = np.hypot(det[0] - center[0], det[1] - center[1]) <= cfg.max_center_dist_m
            has_verts = isinstance(source_det, dict) and "verts" in source_det
            model_iou = (polygon_iou(obb.vertices, np.asarray(source_det["verts"]))
                         if has_verts else 0.0)
            if center_ok and (model_iou >= cfg.min_model_iou or not has_verts):
                supporting.append(det)
        if not supporting:
            continue
        best = max(supporting, key=lambda d: float(d[3] * d[4]))
        score = float(next((x.get("score", 1.0) for x in detections
                            if isinstance(x, dict) and _det_box(x) is not None
                            and np.allclose(_det_box(x), best)), 1.0))
        if score < cfg.min_score:
            continue
        out.append({
            "cx": float(center[0]), "cy": float(center[1]),
            "z": float(best[2]), "length": float(length), "width": float(width),
            "height": float(best[5]), "theta": theta, "score": score,
            "area_m2": float(area),
        })
    return out
