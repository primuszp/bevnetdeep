"""Shared checkpoint loading, sliding inference, decoding, NMS, and matching."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bordereye._nms import greedy_rotated_nms
from bordereye.geometry import polygon_iou
from bordereye.paths import resolve_checkpoint_path

from .model_factory import build_bev_model
from .nn_contract import STRIDE, box_corners, cue_indices, unpack_model_output
from .orientation import use_doubled_yaw

MIN_BOX_M = 1.0
WISAR_MIN_BOX_M = 0.1
MAX_BOX_M = 12.0


def checkpoint_channel_names(ck: dict) -> tuple[str, ...]:
    return tuple(str(x) for x in ck.get("channel_names", ()))


def load_bevnet_checkpoint(
    ckpt_path: str | Path | Mapping[str, Any],
    dev: torch.device,
    *,
    expected_channel_names: tuple[str, ...] | list[str] | None = None,
) -> tuple[nn.Module, dict, np.ndarray, tuple[str, ...]]:
    """Load a BevNet checkpoint and validate channel compatibility.

    The BEV channel schema is part of the model contract. Silent schema drift is
    worse than a hard failure because all channels are geometry-like tensors and
    swapped channels still look numerically plausible.
    """
    ck = (ckpt_path if isinstance(ckpt_path, Mapping)
          else torch.load(
              resolve_checkpoint_path(ckpt_path),
              map_location=dev,
              weights_only=False,
          ))
    ck_names = checkpoint_channel_names(ck)
    expected = tuple(str(x) for x in (expected_channel_names or ()))
    if ck_names and expected and ck_names != expected:
        raise ValueError(
            "Checkpoint/data channel schema mismatch:\n"
            f"  checkpoint: {list(ck_names)}\n"
            f"  data:       {list(expected)}"
        )

    ch_scale = np.asarray(ck.get("ch_scale", []), np.float32)
    n_ch = int(ck.get("n_ch", len(ck_names) or len(ch_scale) or len(expected) or 6))
    if ck_names and len(ck_names) != n_ch:
        raise ValueError(f"Checkpoint n_ch={n_ch} but records {len(ck_names)} channel names")
    if len(ch_scale) not in (0, n_ch):
        raise ValueError(f"Checkpoint n_ch={n_ch} but ch_scale has length {len(ch_scale)}")
    names_for_model = ck_names or expected
    has_cue_branch = any(k.startswith("cue_branch") for k in ck["model"])
    # Prefer the explicit cue meta recorded by newer checkpoints; fall back to
    # deriving it from the channel schema for pre-meta checkpoints.
    if "cue_idx" in ck and ck["cue_idx"] is not None:
        cidx = tuple(int(i) for i in ck["cue_idx"]) or None
        if bool(cidx) != has_cue_branch:
            raise ValueError(
                f"Checkpoint cue meta (cue_idx={ck['cue_idx']}) disagrees with its "
                f"weights (cue_branch present={has_cue_branch}).")
    else:
        cidx = cue_indices(names_for_model) if has_cue_branch else None
        if has_cue_branch and not cidx:
            raise ValueError("Checkpoint has cue_branch weights but no cue channels in schema.")

    ck_args = ck.get("args", {}) if isinstance(ck.get("args", {}), dict) else {}
    num_classes = int(ck.get("num_classes", ck_args.get("num_classes", 1)))
    arch = ck.get("arch", "v1")
    recorded_base = int(ck.get("arch_base", ck_args.get(
        "arch_base", 64 if arch == "deep" else 48)))
    # Some early v1 checkpoints recorded the CLI default (64) even though the
    # compact model factory used its native base (48). The stem tensor is the
    # authoritative architecture contract and also supports small test models.
    stem_weight = ck["model"].get("stem.0.weight")
    model_base = int(stem_weight.shape[0]) if stem_weight is not None else recorded_base
    # "reg_gate" is saved as an explicit top-level key in new checkpoints.
    # Old checkpoints (v7/v8/v9) lack it → default False for backward compat.
    reg_gate = bool(ck.get("reg_gate", False))
    model = build_bev_model(
        arch,
        n_ch=n_ch,
        base=model_base,
        cue_idx=cidx,
        cbam_max=bool(ck_args.get("cbam_max", arch == "deep")),
        reg_gate=reg_gate,
        num_classes=num_classes,
    ).to(dev)

    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint weights do not match the BevNet architecture:\n"
            f"  missing: {missing}\n"
            f"  unexpected: {unexpected}"
        )
    model.eval()
    return model, ck, ch_scale, ck_names


# ── Sliding-window inference utilities ────────────────────────────────────────

def window_starts(size: int, win: int, step: int, *, stride: int = 1) -> list[int]:
    """Cover the far edge while keeping each window on the model's output grid."""
    if win <= 0:
        raise ValueError(f"window size must be positive, got {win}")
    if not 0 < step <= win:
        raise ValueError(f"window step must be between 1 and {win}; got {step}")
    if not 0 < stride <= win:
        raise ValueError(f"output stride must be between 1 and {win}; got {stride}")
    if size <= 0:
        raise ValueError(f"raster dimension must be positive, got {size}")
    if size <= win:
        return [0]
    aligned_step = max(stride, step // stride * stride)
    last = ((size - win + stride - 1) // stride) * stride
    return [*range(0, last, aligned_step), last]


def run_sliding_window(
    model: nn.Module,
    channels: np.ndarray,
    dev: torch.device,
    *,
    cell: float,
    win: int = 512,
    overlap: int = 64,
    conf: float = 0.01,
    score_mode: str = "hm",
    amp: bool = False,
    batch_size: int = 0,
    origin: tuple[float, float] | np.ndarray = (0.0, 0.0),
    output_stride: int | None = None,
    orientation_source: str = "generic",
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Run the shared tiled inference path used by evaluation and Jeli.

    ``batch_size`` groups windows into one model call and one host/device
    transfer. Zero selects a conservative device-aware value, one preserves
    the low-memory path, and larger values improve accelerator utilisation.
    """
    if channels.ndim != 3:
        raise ValueError(f"channels must have shape (C,H,W), got {channels.shape}")
    if not np.isfinite(cell) or cell <= 0:
        raise ValueError(f"cell must be a positive finite value, got {cell}")
    score_modes = {"hm", "hm_seg_center", "hm_seg_obb", "hm_seg_center_obb"}
    if score_mode not in score_modes:
        raise ValueError(f"unknown decode score mode {score_mode!r}; expected one of {sorted(score_modes)}")
    world_origin = np.asarray(origin, dtype=np.float64)
    if world_origin.shape != (2,) or not np.isfinite(world_origin).all():
        raise ValueError("origin must contain two finite coordinates")
    step = win - overlap
    if win <= 0 or not 0 <= overlap < win:
        raise ValueError(
            f"expected 0 <= overlap < window; got win={win}, overlap={overlap}"
        )
    batch_size = int(batch_size)
    if batch_size < 0:
        raise ValueError(f"inference batch size cannot be negative, got {batch_size}")
    if batch_size == 0:
        batch_size = 1
        if dev.type == "cuda":
            total_memory = torch.cuda.get_device_properties(dev).total_memory
            if total_memory >= 12 * 1024 ** 3:
                batch_size = 4
            elif total_memory >= 8 * 1024 ** 3:
                batch_size = 2

    stride = int(getattr(model, "output_stride", STRIDE) if output_stride is None else output_stride)
    if stride <= 0:
        raise ValueError(f"output_stride must be positive, got {stride}")
    n_channels, height, width = channels.shape
    out_h = (height + stride - 1) // stride
    out_w = (width + stride - 1) // stride
    full_hm = np.zeros((out_h, out_w), np.float32)
    full_seg = np.zeros_like(full_hm)
    detections: list[dict] = []
    use_amp = bool(amp) and dev.type == "cuda"
    row_starts = window_starts(height, win, step, stride=stride)
    col_starts = window_starts(width, win, step, stride=stride)
    windows = [
        (r0, c0, min(r0 + win, height), min(c0 + win, width))
        for r0 in row_starts
        for c0 in col_starts
    ]
    batch_size = min(batch_size, len(windows))
    tile_batch = np.zeros((batch_size, n_channels, win, win), np.float32)

    with torch.inference_mode():
        for batch_start in range(0, len(windows), batch_size):
            batch_windows = windows[batch_start:batch_start + batch_size]
            tiles = tile_batch[:len(batch_windows)]
            tiles.fill(0.0)
            for tile, (r0, c0, r1, c1) in zip(tiles, batch_windows, strict=True):
                tile[:, :r1 - r0, :c1 - c0] = channels[:, r0:r1, c0:c1]
            tile_tensor = torch.from_numpy(tiles).to(dev)
            with torch.amp.autocast("cuda", enabled=use_amp):
                phm, preg, pseg = unpack_model_output(model(tile_tensor))
            # Apply activations once for both decoding and stitched output, and
            # transfer whole batches instead of synchronising once per window.
            hm_batch = torch.sigmoid(phm.float()).cpu()
            reg_batch = preg.float().cpu()
            seg_batch = torch.sigmoid(pseg.float()).cpu()

            for index, (r0, c0, r1, c1) in enumerate(batch_windows):
                hm = hm_batch[index]
                reg = reg_batch[index]
                seg = seg_batch[index]
                detections.extend(decode_window(
                    hm, reg, r0, c0, world_origin, cell, height, conf,
                    seg_logits=seg, score_mode=score_mode,
                    valid_h=r1 - r0, valid_w=c1 - c0,
                    probabilities=True,
                    output_stride=stride,
                    orientation_source=orientation_source,
                ))

                rr, cc = r0 // stride, c0 // stride
                # Keep the stitched diagnostic heatmap class-agnostic while
                # detections retain their class_id. This preserves the legacy
                # return contract for visualisation and one-class consumers.
                hm_np = hm.amax(dim=0).numpy()
                seg_np = seg[0].numpy()
                hm_out = full_hm[rr:rr + hm_np.shape[0], cc:cc + hm_np.shape[1]]
                seg_out = full_seg[rr:rr + seg_np.shape[0], cc:cc + seg_np.shape[1]]
                np.maximum(hm_out, hm_np[:hm_out.shape[0], :hm_out.shape[1]], out=hm_out)
                np.maximum(seg_out, seg_np[:seg_out.shape[0], :seg_out.shape[1]], out=seg_out)
    return detections, full_hm, full_seg


def _obb_seg_support(seg: np.ndarray, cx: float, cy: float, Lg: float, Wg: float,
                     theta: float) -> float:
    """Mean segmentation probability inside a decoded OBB on the output grid."""
    H, W = seg.shape
    half_l = max(float(Lg) * 0.5, 1e-3)
    half_w = max(float(Wg) * 0.5, 1e-3)
    rad = int(np.ceil(np.hypot(half_l, half_w))) + 2
    x0, x1 = max(0, int(np.floor(cx)) - rad), min(W, int(np.ceil(cx)) + rad + 1)
    y0, y1 = max(0, int(np.floor(cy)) - rad), min(H, int(np.ceil(cy)) + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx.astype(np.float32) + 0.5 - float(cx)
    dy = yy.astype(np.float32) + 0.5 - float(cy)
    # World theta has y-up convention; image/grid rows grow downward.
    ca = float(np.cos(-theta))
    sa = float(np.sin(-theta))
    long_axis = dx * ca + dy * sa
    short_axis = -dx * sa + dy * ca
    mask = (np.abs(long_axis) <= half_l) & (np.abs(short_axis) <= half_w)
    if not mask.any():
        return 0.0
    return float(np.clip(seg[y0:y1, x0:x1][mask].mean(), 0.0, 1.0))


def _joint_detection_score(hm_score: float, seg_center: float, seg_obb: float,
                           mode: str) -> float:
    if mode == "hm":
        return hm_score
    if mode == "hm_seg_center":
        return hm_score * (0.5 + 0.5 * seg_center)
    if mode == "hm_seg_obb":
        return hm_score * (0.5 + 0.5 * seg_obb)
    if mode == "hm_seg_center_obb":
        return hm_score * (0.5 + 0.5 * seg_center) * (0.5 + 0.5 * seg_obb)
    raise ValueError(f"unknown decode score mode: {mode}")


def decode_window(
    hm_logits: torch.Tensor,
    reg: torch.Tensor,
    r0: int,
    c0: int,
    origin: tuple[float, float],
    cell: float,
    H: int,
    conf: float,
    seg_logits: torch.Tensor | None = None,
    score_mode: str = "hm",
    valid_h: int | None = None,
    valid_w: int | None = None,
    probabilities: bool = False,
    output_stride: int = STRIDE,
    orientation_source: str = "generic",
) -> list[dict]:
    """Decode one sliding-window output into world-space OBB detections."""
    hm = hm_logits if probabilities else torch.sigmoid(hm_logits)
    if hm.ndim == 2:
        hm = hm.unsqueeze(0)
    if hm.ndim != 3:
        raise ValueError(f"heatmap must have shape (classes,H,W), got {tuple(hm.shape)}")
    keep = (F.max_pool2d(hm[None], 3, 1, 1)[0] == hm) & (hm >= conf)
    class_ids, ys, xs = torch.nonzero(keep, as_tuple=True)
    if ys.numel() == 0:
        return []
    # One device->host transfer per window; per-peak .tolist()/float() on CUDA
    # tensors would force a GPU sync for every candidate.
    hm_np = hm.detach().cpu().numpy()
    reg_np = reg.detach().cpu().numpy()
    seg = (
        (seg_logits[0] if probabilities else torch.sigmoid(seg_logits[0]))
        .detach().cpu().numpy()
        if seg_logits is not None else None
    )
    out = []
    for class_id, iy, ix in zip(
        class_ids.tolist(), ys.tolist(), xs.tolist(), strict=True
    ):
        hm_score = float(hm_np[class_id, iy, ix])
        reg_at_peak = (reg_np[class_id, :, iy, ix] if reg_np.ndim == 4
                       else reg_np[:, iy, ix])
        dx, dy, logL, logW, s2, c2, z, h = (float(v) for v in reg_at_peak)
        local_col = (ix + dx) * output_stride
        local_row = (iy + dy) * output_stride
        if valid_w is not None and (local_col < 0 or local_col >= valid_w):
            continue
        if valid_h is not None and (local_row < 0 or local_row >= valid_h):
            continue
        col = c0 + local_col
        row = r0 + local_row
        x = origin[0] + col * cell
        y = origin[1] + (H - row) * cell
        if not np.isfinite([dx, dy, logL, logW, s2, c2, z, h]).all():
            continue
        min_box_m = WISAR_MIN_BOX_M if orientation_source.lower() == "wisar" else MIN_BOX_M
        logL = float(np.clip(logL, np.log(min_box_m), np.log(MAX_BOX_M)))
        logW = float(np.clip(logW, np.log(min_box_m), np.log(MAX_BOX_M)))
        L, W = float(np.exp(logL)), float(np.exp(logW))
        yaw_factor = 2.0 if use_doubled_yaw(orientation_source, class_id) else 1.0
        th = float(np.arctan2(s2, c2) / yaw_factor)
        verts = box_corners(x, y, L, W, th)
        seg_center = 1.0
        seg_obb = 1.0
        if seg is not None:
            cyg = float(iy + dy)
            cxg = float(ix + dx)
            cyi = int(np.clip(round(cyg), 0, seg.shape[0] - 1))
            cxi = int(np.clip(round(cxg), 0, seg.shape[1] - 1))
            y0, y1 = max(0, cyi - 1), min(seg.shape[0], cyi + 2)
            x0, x1 = max(0, cxi - 1), min(seg.shape[1], cxi + 2)
            seg_center = float(np.clip(seg[y0:y1, x0:x1].mean(), 0.0, 1.0))
            seg_obb = _obb_seg_support(seg, cxg, cyg, L / cell / output_stride,
                                       W / cell / output_stride, th)
        score = _joint_detection_score(hm_score, seg_center, seg_obb, score_mode)
        out.append({"cx": x, "cy": y, "L": L, "W": W, "th": th,
                    "class_id": int(class_id),
                    "score": score, "hm_score": hm_score,
                    "seg_center_score": seg_center, "seg_obb_support": seg_obb,
                    "z": z, "height": h, "verts": verts})
    return out


def _bounding_circle(verts: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the circumscribed circle of an OBB corner array."""
    v = np.asarray(verts, np.float64)
    c = v.mean(axis=0)
    r = float(np.sqrt(((v - c) ** 2).sum(axis=1).max()))
    return c, r


def nms(dets: list[dict], iou_thr: float) -> list[dict]:
    """Greedy score-descending rotated-box NMS over detection dicts.

    Rotated IoU is only evaluated for pairs whose bounding circles overlap;
    disjoint circles imply IoU = 0, so the result is exact.
    """
    def _circle(d: dict) -> tuple[float, float, float]:
        centre, radius = _bounding_circle(d["verts"])
        return float(centre[0]), float(centre[1]), radius

    return greedy_rotated_nms(
        dets,
        iou_thr,
        corners=lambda d: d["verts"],
        circle=_circle,
        key=lambda d: d["score"],
    )


def match_detections_one_to_one(
    dets: list[dict],
    gt_verts: list,
    ignore_verts: list,
    iou_thr: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy score-ordered one-to-one matching for evaluation.

    Each GT can be matched by at most one detection.  Ignore regions are
    applied only to detections left unmatched after GT assignment, so a true
    positive that also overlaps a broad ignore region is not discarded.
    """
    det_circles = [_bounding_circle(d["verts"]) for d in dets]
    ignore_circles = [_bounding_circle(v) for v in ignore_verts]
    gt_circles = [_bounding_circle(v) for v in gt_verts]

    best_ignore = np.zeros(len(dets), np.float32)
    for i, de in enumerate(dets):
        dc, dr = det_circles[i]
        for iv, (ic, ir) in zip(ignore_verts, ignore_circles, strict=True):
            if float(np.hypot(*(dc - ic))) >= dr + ir:
                continue
            best_ignore[i] = max(best_ignore[i], polygon_iou(de["verts"], iv))

    matched_gt = np.zeros(len(gt_verts), np.float32)
    matched_det = np.zeros(len(dets), np.float32)
    used_gt: set[int] = set()
    eval_indices = sorted(range(len(dets)), key=lambda i: dets[i]["score"], reverse=True)
    for di in eval_indices:
        dc, dr = det_circles[di]
        best_iou, best_gi = 0.0, -1
        for gi, gv in enumerate(gt_verts):
            if gi in used_gt:
                continue
            gc, gr = gt_circles[gi]
            if float(np.hypot(*(dc - gc))) >= dr + gr:
                continue
            iou = polygon_iou(dets[di]["verts"], gv)
            if iou > best_iou:
                best_iou, best_gi = iou, gi
        if best_gi >= 0 and best_iou >= iou_thr:
            used_gt.add(best_gi)
            matched_gt[best_gi] = best_iou
            matched_det[di] = best_iou
            # A GT match takes precedence over ignore overlap.  Downstream
            # accounting uses best_ignore only for unmatched detections.
            best_ignore[di] = 0.0
    return matched_gt, matched_det, best_ignore
