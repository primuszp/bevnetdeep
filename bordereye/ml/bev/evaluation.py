"""Evaluate BEV checkpoints on BEV NPZ tiles.

The evaluator runs sliding-window inference, decodes OBB detections, matches
them one-to-one against boxes_valid, and reports detection plus segmentation
metrics. It is intended for comparing training runs with the same dataset split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from bordereye.ml.bev.core.evaluation_metrics import (
    SEG_PARTIAL_DILATION_PX,
    average_precision_from_ranked,
    segmentation_metrics,
)
from bordereye.ml.bev.core.footprint_targets import rasterize_box_rows
from bordereye.ml.bev.core.inference import (
    load_bevnet_checkpoint,
    match_detections_one_to_one,
    nms,
    run_sliding_window,
)
from bordereye.ml.bev.core.metrics import incomplete_segmentation_metrics
from bordereye.ml.bev.core.nn_contract import (
    STRIDE,
    box_corners,
    channel_names,
    channel_scale,
    world_to_pixel,
)
from bordereye.ml.bev.core.normalization import (
    normalize_bev_channels,
    validate_norm_config,
)
from bordereye.ml.bev.core.pseudo_obb import (
    PseudoObbConfig,
    pseudo_obbs_from_segmentation,
)
from bordereye.ml.bev.data.paths import dataset_root, output_root
from bordereye.ml.models.checkpoints import resolve_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--weights", required=True,
        help="trained BEV weights file or checkpoint ID (e.g. paper-model)",
    )
    p.add_argument("--dataset-dir", default=str(dataset_root()))
    p.add_argument("--out-dir", default=str(output_root() / "eval"))
    p.add_argument(
        "--source", default="dales", help="label used in reports; also selects default pattern"
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="restrict evaluated tiles by dataset split. Use 'test' to avoid "
        "scoring on tiles the model trained on (leakage guard).",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run inference under CUDA autocast for speed (--no-amp to disable)",
    )
    p.add_argument(
        "--pattern",
        default="",
        help="glob pattern; default: dales=ply_*.npz, synth=synth_*.npz, all=*.npz",
    )
    p.add_argument(
        "--tiles",
        default="",
        help="comma-separated tile stems or filenames; overrides pattern when set",
    )
    p.add_argument("--max-tiles", type=int, default=0)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--win", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument(
        "--infer-batch-size",
        type=int,
        default=0,
        help="sliding windows per model call; 0 selects a conservative device-aware value",
    )
    p.add_argument("--conf-list", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    p.add_argument(
        "--conf",
        type=float,
        default=0.0,
        help="fixed operating confidence threshold for the headline metrics "
        "(unbiased). If 0, the threshold is selected on one half of the "
        "tiles and reported on the other half to avoid the selection bias "
        "of picking argmax-F1 on the same set that is reported.",
    )
    p.add_argument("--nms-iou", type=float, default=0.20)
    p.add_argument("--match-iou", type=float, default=0.30)
    p.add_argument(
        "--ap-iou-list", default="0.30,0.50", help="rotated-IoU thresholds for ranked all-point AP"
    )
    p.add_argument(
        "--ap-min-conf",
        type=float,
        default=0.01,
        help="minimum decoded confidence included in AP ranking",
    )
    p.add_argument("--seg-thresh", type=float, default=0.50)
    p.add_argument(
        "--decode-score-mode",
        choices=("hm", "hm_seg_center", "hm_seg_obb", "hm_seg_center_obb"),
        default="hm_seg_center_obb",
        help="final OBB confidence rule. hm keeps the legacy heatmap-only score; "
        "the seg modes down-weight detections unsupported by the segmentation head. "
        "Default hm_seg_center_obb: score = hm × (0.5+0.5×seg_center) × (0.5+0.5×seg_obb).",
    )
    p.add_argument(
        "--box-target",
        choices=("auto", "valid", "pseudo", "any"),
        default="auto",
        help="which OBB labels to score against. auto uses valid boxes when present, "
        "otherwise pseudo boxes. Use pseudo for YUTO/ECLAIR/ISPRS proxy eval.",
    )
    p.add_argument(
        "--mask-obb-report",
        action="store_true",
        help="also report conservative mask+det agreement OBB counts, using the "
        "retained pseudo-OBB filter as a postprocess sanity check",
    )
    p.add_argument(
        "--partial-label-eval",
        action="store_true",
        help="treat selected sources as partially labelled: unmatched detections are "
        "reported as novel candidates instead of false positives",
    )
    p.add_argument(
        "--partial-label-sources",
        default="yuto,eclair,isprs",
        help="comma-separated sources where labels are incomplete",
    )
    p.add_argument(
        "--viz-tiles",
        type=int,
        default=8,
        help="number of overlay PNGs to save; 0 = every evaluated tile",
    )
    p.add_argument(
        "--display-channel",
        default="veh_hmax",
        help="channel name or integer index for overlay background",
    )
    p.add_argument(
        "--export-detections",
        default="",
        help="optional CSV path for NMS detections in source coordinates; "
        "columns are frame,score,cx,cy,length,width,yaw,z,height",
    )
    return p.parse_args(argv)


def default_pattern(source: str) -> str:
    s = source.lower()
    # DALES tiles are dales_* in bev_dataset_v3 but keep their historical
    # ply_* stems in the older bev_dataset; try both (';' = alternatives).
    known = {
        "dales": "dales_*.npz;ply_*.npz",
        "synth": "synth_*.npz",
        "synthsparse": "synthsparse_*.npz",
        "wisar": "wisar_*.npz",
        "isprs": "isprs_*.npz",
        "eclair": "eclair_*.npz",
        "yuto": "yuto_*.npz",
    }
    return known.get(s, "*.npz")


def tile_split(path: Path) -> str:
    """Return 'train' or 'test' for a tile path.

    Priority:
      1. Parent directory name if it is 'train' or 'test' (new bev_dataset_v3 layout).
      2. source_split key inside the NPZ (older datasets).
      3. Falls back to 'train' (conservative: avoids treating unknown tiles as test).
    """
    parent = path.parent.name.lower()
    if parent in ("train", "test"):
        return parent
    try:
        with np.load(path) as d:
            if "source_split" in d:
                return str(d["source_split"])
    except Exception:
        pass
    return "train"


def filter_by_split(paths: list[Path], split: str) -> list[Path]:
    """Keep only tiles that belong to the requested split."""
    if split == "all":
        return paths
    return [p for p in paths if tile_split(p) == split]


def resolve_paths(dataset_dir: Path, pattern: str, tiles: str) -> list[Path]:
    if tiles.strip():
        out = []
        for raw in tiles.split(","):
            name = raw.strip()
            if not name:
                continue
            filename = name if Path(name).suffix.lower() == ".npz" else f"{name}.npz"
            candidates = [dataset_dir / filename]
            candidates.extend(dataset_dir / split / filename for split in ("train", "test"))
            existing = [p for p in candidates if p.exists()]
            if len(existing) > 1:
                raise FileExistsError(
                    f"Ambiguous tile {filename!r}; found: "
                    + ", ".join(str(p) for p in existing)
                )
            p = existing[0] if existing else candidates[0]
            out.append(p)
        missing = [str(p) for p in out if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing tile(s): " + ", ".join(missing))
        return out
    for pat in (p.strip() for p in pattern.split(";") if p.strip()):
        paths = sorted(dataset_dir.glob(pat))
        if not paths:
            paths = sorted(p for sub in ("train", "test") for p in (dataset_dir / sub).glob(pat))
        if paths:
            return paths
    return []


def load_tile(npz_path: Path, ch_scale_ck, ck_names, norm_config: dict, box_target: str):
    with np.load(npz_path, allow_pickle=False) as d:
        channels = d["channels"].astype(np.float32)
        names = channel_names(d)
        subset_idx = None
        if ck_names and names and tuple(ck_names) != names:
            if set(ck_names) <= set(names):
                # checkpoint trained on a named subset of this schema
                # (--use-channels): normalize the full stack, then select.
                subset_idx = [names.index(n) for n in ck_names]
            else:
                raise ValueError(
                    f"channel schema mismatch for {npz_path.name}: {ck_names} != {names}"
                )
        scale = (
            ch_scale_ck
            if len(ch_scale_ck) == channels.shape[0] and (not ck_names or tuple(ck_names) == names)
            else channel_scale(d)
        )
        channels = normalize_bev_channels(channels, scale, norm_config)
        if subset_idx is not None:
            channels = channels[subset_idx]
            names = tuple(ck_names)
        origin = d["origin"].astype(np.float64)
        cell = float(d["cell"])
        boxes_valid = (
            d["boxes_valid"] if "boxes_valid" in d else d.get("boxes", np.zeros((0, 7)))
        ).astype(np.float32)
        boxes_pseudo = (d["boxes_pseudo"] if "boxes_pseudo" in d else np.zeros((0, 7))).astype(
            np.float32
        )
        if box_target == "valid":
            boxes = boxes_valid
            box_target_used = "valid"
        elif box_target == "pseudo":
            boxes = boxes_pseudo
            box_target_used = "pseudo"
        elif box_target == "any":
            boxes = np.concatenate([boxes_valid, boxes_pseudo], axis=0)
            box_target_used = "any"
        else:
            boxes = boxes_valid if len(boxes_valid) else boxes_pseudo
            box_target_used = "valid" if len(boxes_valid) else "pseudo"
        ignore = (d["boxes_ignore"] if "boxes_ignore" in d else np.zeros((0, 7))).astype(np.float32)

        def _nonempty_mask(key: str):
            if key not in d:
                return None
            value = np.asarray(d[key], dtype=bool)
            return value if value.any() else None

        raw_seg = _nonempty_mask("seg_mask")
        box_seg = _nonempty_mask("seg_box_mask")
        if box_seg is not None:
            seg = box_seg
        else:
            footprint_seg = _nonempty_mask("fp_mask")
            seg = (
                footprint_seg
                if footprint_seg is not None
                else raw_seg
                if raw_seg is not None
                else np.zeros(channels.shape[1:], bool)
            )
        source = str(d["source"]) if "source" in d else npz_path.name.split("_")[0]
        split = str(d["source_split"]) if "source_split" in d else ""
    return (
        channels,
        names,
        origin,
        cell,
        boxes,
        ignore,
        seg,
        raw_seg,
        source,
        split,
        box_target_used,
        len(boxes_valid),
        len(boxes_pseudo),
    )


def run_model(ch: np.ndarray, model, dev, args, cell: float, decode_conf: float):
    return run_sliding_window(
        model,
        ch,
        dev,
        cell=cell,
        win=args.win,
        overlap=args.overlap,
        conf=decode_conf,
        score_mode=args.decode_score_mode,
        amp=bool(getattr(args, "amp", False)),
        batch_size=args.infer_batch_size,
    )


def local_box_verts(boxes: np.ndarray, origin: np.ndarray):
    return [
        box_corners(b[0] - origin[0], b[1] - origin[1], b[2], b[3], np.deg2rad(b[4])) for b in boxes
    ]


def evaluate_dets(
    dets_nms,
    gt_verts,
    ignore_verts,
    conf: float,
    match_iou: float,
    partial_label: bool = False,
    precomputed_matches: tuple[np.ndarray, np.ndarray] | None = None,
):
    """Score already-NMSed detections at one confidence threshold.

    NMS runs once per tile at the lowest threshold (see main); for greedy
    score-descending NMS, filtering the kept set by a higher threshold is
    exactly equivalent to filtering first and re-running NMS, because a
    suppressed box's suppressor always has a higher score and therefore also
    survives the filter.
    """
    keep = np.fromiter((d["score"] >= conf for d in dets_nms), dtype=bool, count=len(dets_nms))
    dets = [d for d, selected in zip(dets_nms, keep, strict=True) if selected]
    if precomputed_matches is None:
        _best_gt, best_det, best_ignore = match_detections_one_to_one(
            dets, gt_verts, ignore_verts, match_iou
        )
    else:
        all_best_det, all_best_ignore = precomputed_matches
        if len(all_best_det) != len(dets_nms) or len(all_best_ignore) != len(dets_nms):
            raise ValueError("precomputed match arrays must align with dets_nms")
        best_det = all_best_det[keep]
        best_ignore = all_best_ignore[keep]
    eval_mask = best_ignore < match_iou
    tp = int(((best_det >= match_iou) & eval_mask).sum())
    matched_ious = best_det[(best_det >= match_iou) & eval_mask]
    apparent_fp = int(eval_mask.sum() - tp)
    fp = 0 if partial_label else apparent_fp
    fn = len(gt_verts) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return (
        {
            "detections": len(dets),
            "eval_detections": int(eval_mask.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "apparent_fp": apparent_fp,
            "novel_candidates": apparent_fp if partial_label else 0,
            "known_supported": tp,
            "matched_iou_sum": float(matched_ious.sum()),
            "matched_iou_count": len(matched_ious),
            "partial_label": int(bool(partial_label)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        dets,
        best_det,
        best_ignore,
    )


def _summary_from_rows(rows, conf: float) -> dict:
    """Aggregate per-tile TP/FP/FN rows into one precision/recall/F1 summary."""
    tp = sum(int(r["tp"]) for r in rows)
    fp = sum(int(r["fp"]) for r in rows)
    fn = sum(int(r["fn"]) for r in rows)
    apparent_fp = sum(int(r.get("apparent_fp", r["fp"])) for r in rows)
    novel = sum(int(r.get("novel_candidates", 0)) for r in rows)
    det = sum(int(r["detections"]) for r in rows)
    edet = sum(int(r["eval_detections"]) for r in rows)
    iou_sum = sum(float(r.get("matched_iou_sum", 0.0)) for r in rows)
    iou_count = sum(int(r.get("matched_iou_count", 0)) for r in rows)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return {
        "conf": conf,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "apparent_fp": apparent_fp,
        "novel_candidates": novel,
        "detections": det,
        "eval_detections": edet,
        "precision": p,
        "recall": r,
        "f1": f1,
        "matched_iou_mean": iou_sum / max(iou_count, 1),
        "matched_iou_count": iou_count,
    }


def _seg_summary_from_rows(rows) -> dict:
    inter = sum(int(r.get("seg_inter", 0)) for r in rows)
    union = sum(int(r.get("seg_union", 0)) for r in rows)
    pred = sum(int(r.get("seg_pred_px", 0)) for r in rows)
    gt = sum(int(r.get("seg_gt_px", 0)) for r in rows)
    pred_in_dil = sum(int(r.get("seg_pred_in_dil_px", 0)) for r in rows)
    weak_holes = sum(int(r.get("seg_weak_hole_ignored_px", 0)) for r in rows)
    partial_union = pred_in_dil + gt - inter
    partial_count_keys = (
        "observed_tp",
        "observed_px",
        "predicted_px",
        "predicted_inside_allowed_px",
        "predicted_outside_allowed_px",
        "allowed_px",
        "trusted_intersection_px",
        "trusted_union_px",
        "trusted_target_px",
    )
    partial_counts = {k: sum(int(r.get(f"seg_{k}", 0)) for r in rows) for k in partial_count_keys}
    partial = incomplete_segmentation_metrics(partial_counts)
    return {
        "seg_inter": inter,
        "seg_union": union,
        "seg_pred_px": pred,
        "seg_gt_px": gt,
        "seg_iou": inter / max(union, 1),
        "seg_dice": 2 * inter / max(pred + gt, 1),
        "seg_precision": inter / max(pred, 1),
        "seg_recall": inter / max(gt, 1),
        "seg_partial_iou": inter / max(partial_union, 1),
        "seg_weak_hole_ignored_px": weak_holes,
        **{f"seg_{k}": v for k, v in partial.items()},
    }


def source_summaries(tile_rows, conf: float) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for r in tile_rows:
        by_source.setdefault(str(r.get("source", "unknown")), []).append(r)
    out = []
    for src, rows in sorted(by_source.items()):
        det = _summary_from_rows(rows, conf)
        seg = _seg_summary_from_rows(rows)
        is_partial = any(int(r.get("partial_label", 0)) for r in rows)
        out.append(
            {
                "source": src,
                "tiles": len(rows),
                "partial_label": int(is_partial),
                "box_target": ",".join(sorted(set(str(r.get("box_target", "")) for r in rows))),
                "gt_boxes": sum(int(r.get("gt_boxes", 0)) for r in rows),
                "valid_boxes": sum(int(r.get("valid_boxes", 0)) for r in rows),
                "pseudo_boxes": sum(int(r.get("pseudo_boxes", 0)) for r in rows),
                **det,
                **seg,
                "mask_obb_candidates": sum(int(r.get("mask_obb_candidates", 0)) for r in rows),
            }
        )
    return out


def select_headline_summary(tile_rows_by_thr, thresholds, paths, fixed_conf):
    """Pick the headline detection summary without the selection bias of taking
    argmax-F1 on the same tiles that are reported.

    * fixed_conf > 0: report at that operating threshold (fully unbiased).
    * otherwise: choose the threshold on even-indexed tiles, report on the
      odd-indexed held-out tiles.
    * single tile: fall back to the biased same-set argmax (clearly noted).
    """
    n_tiles = len(paths)
    if fixed_conf > 0:
        op = float(fixed_conf)
        rows = tile_rows_by_thr[op]
        return (
            _summary_from_rows(rows, op),
            rows,
            f"fixed operating threshold --conf {op:.3f} (unbiased)",
        )
    if n_tiles >= 2:
        sel_idx = list(range(0, n_tiles, 2))
        rep_idx = list(range(1, n_tiles, 2))
        best_thr = max(
            thresholds,
            key=lambda t: _summary_from_rows([tile_rows_by_thr[t][i] for i in sel_idx], t)["f1"],
        )
        rep_rows = [tile_rows_by_thr[best_thr][i] for i in rep_idx]
        return (
            _summary_from_rows(rep_rows, best_thr),
            rep_rows,
            (
                f"threshold {best_thr:.3f} selected on {len(sel_idx)} tiles, "
                f"reported on the held-out {len(rep_idx)} tiles"
            ),
        )
    biased = max(
        (_summary_from_rows(tile_rows_by_thr[t], t) for t in thresholds), key=lambda r: r["f1"]
    )
    return (
        biased,
        tile_rows_by_thr[float(biased["conf"])],
        "single tile: threshold selected and reported on the same tile (biased)",
    )


def conservative_mask_obb_count(full_seg, dets, cell: float) -> int:
    cfg = PseudoObbConfig(
        seg_threshold=0.45,
        min_model_iou=0.25,
        max_center_dist_m=2.5,
        min_score=0.10,
    )
    try:
        return len(
            pseudo_obbs_from_segmentation(
                full_seg, dets, np.zeros(2, np.float64), cell * STRIDE, cfg=cfg
            )
        )
    except Exception:
        return 0


def display_index(names, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return names.index(value) if value in names else 0


def save_overlay(
    path: Path,
    bg,
    origin,
    cell,
    boxes,
    ignore,
    dets,
    best_det,
    best_ignore,
    args,
    viz_conf: float | None = None,
):
    """Render one tile overlay. `bg` is the single display channel (H, W)."""
    H = bg.shape[0]
    vmax = np.percentile(bg[bg > 0], 99) if (bg > 0).any() else 1.0
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(bg, cmap="bone", vmax=vmax)
    for b in boxes:
        verts = box_corners(b[0], b[1], b[2], b[3], np.deg2rad(b[4]))
        ax.add_patch(
            MplPolygon(
                world_to_pixel(verts, origin, cell, H),
                closed=True,
                facecolor="none",
                edgecolor="#22c55e",
                linewidth=1.0,
            )
        )
    for b in ignore:
        verts = box_corners(b[0], b[1], b[2], b[3], np.deg2rad(b[4]))
        ax.add_patch(
            MplPolygon(
                world_to_pixel(verts, origin, cell, H),
                closed=True,
                facecolor="none",
                edgecolor="#f97316",
                linewidth=0.8,
            )
        )
    for i, de in enumerate(dets):
        verts = de["verts"] + origin
        ok = i < len(best_det) and best_det[i] >= args.match_iou and best_ignore[i] < args.match_iou
        ax.add_patch(
            MplPolygon(
                world_to_pixel(verts, origin, cell, H),
                closed=True,
                facecolor="none",
                edgecolor="#06b6d4" if ok else "#eab308",
                linewidth=1.0,
            )
        )
    conf_tag = f" @conf={viz_conf:.2f}" if viz_conf is not None else ""
    ax.set_title(f"{path.stem}: green GT, cyan TP, yellow FP/novel, orange ignore{conf_tag}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_plots(out_dir: Path, thresholds, summary_rows, tile_rows, source: str):
    xs = np.asarray(thresholds, np.float32)
    prec = np.asarray([r["precision"] for r in summary_rows], np.float32)
    rec = np.asarray([r["recall"] for r in summary_rows], np.float32)
    f1 = np.asarray([r["f1"] for r in summary_rows], np.float32)
    dets = np.asarray([r["eval_detections"] for r in summary_rows], np.float32)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(xs, prec, marker="o", label="precision")
    ax[0].plot(xs, rec, marker="o", label="recall")
    ax[0].plot(xs, f1, marker="o", label="F1")
    best_i = int(np.argmax(f1)) if len(f1) else 0
    ax[0].axvline(xs[best_i], color="#888", ls="--", lw=1)
    ax[0].set_title(f"{source} detection vs confidence")
    ax[0].set_xlabel("confidence threshold")
    ax[0].set_ylim(0, 1)
    ax[0].grid(alpha=0.25)
    ax[0].legend()
    ax[1].plot(xs, dets, marker="o", color="#7c3aed")
    ax[1].set_title("evaluated detections")
    ax[1].set_xlabel("confidence threshold")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "detection_threshold_curve.png", dpi=150)
    plt.close(fig)

    seg_iou = [float(r["seg_iou"]) for r in tile_rows]
    val = [float(r["f1"]) for r in tile_rows]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(seg_iou, bins=20, color="#0f766e", alpha=0.85)
    ax[0].set_title("per-tile segmentation IoU")
    ax[0].set_xlabel("IoU")
    ax[0].grid(alpha=0.2)
    ax[1].hist(val, bins=20, color="#2563eb", alpha=0.85)
    ax[1].set_title("per-tile detection F1 at selected threshold")
    ax[1].set_xlabel("F1")
    ax[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "per_tile_histograms.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    console_log = os.environ.get("EVAL_CONSOLE_LOG", "").strip()
    if console_log:
        stream = open(console_log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = stream
        sys.stderr = stream
    dev = resolve_device(args.device)
    model, ck, ch_scale_ck, ck_names = load_bevnet_checkpoint(args.weights, dev)
    norm_config = validate_norm_config(ck.get("norm_config", {"mode": "legacy_tile_percentile"}))
    thresholds = [float(x) for x in args.conf_list.split(",") if x.strip()]
    ap_ious = [float(x) for x in args.ap_iou_list.split(",") if x.strip()]
    if not ap_ious or any(x <= 0 or x > 1 for x in ap_ious):
        raise ValueError("--ap-iou-list values must be in (0, 1]")
    if args.ap_min_conf < 0 or args.ap_min_conf >= 1:
        raise ValueError("--ap-min-conf must be in [0, 1)")
    if args.conf > 0 and not any(abs(t - args.conf) < 1e-9 for t in thresholds):
        thresholds = sorted([*thresholds, float(args.conf)])
    pattern = args.pattern or default_pattern(args.source)
    paths = resolve_paths(Path(args.dataset_dir), pattern, args.tiles)
    if not args.tiles.strip():
        paths = filter_by_split(paths, args.split)
    if args.max_tiles > 0:
        paths = paths[: args.max_tiles]
    partial_sources = {
        s.strip().lower() for s in args.partial_label_sources.split(",") if s.strip()
    }
    if not paths:
        raise FileNotFoundError(
            f"No tiles match {Path(args.dataset_dir) / pattern} with split={args.split}"
        )

    print(
        f"device={dev} weights={args.weights} "
        f"amp={bool(getattr(args, 'amp', False)) and dev.type == 'cuda'}"
    )
    print(f"normalization={norm_config['mode']}")
    print(f"dataset={args.dataset_dir} pattern={pattern} split={args.split} tiles={len(paths)}")

    agg = {
        thr: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "apparent_fp": 0,
            "novel_candidates": 0,
            "eval_detections": 0,
            "detections": 0,
            "matched_iou_sum": 0.0,
            "matched_iou_count": 0,
        }
        for thr in thresholds
    }
    tile_cache = []
    tile_rows_by_thr = {thr: [] for thr in thresholds}
    # novel_rows_by_thr: collect at every threshold so we can emit at the headline
    # operating threshold (determined after all tiles are processed).
    novel_rows_by_thr: dict[float, list[dict]] = {thr: [] for thr in thresholds}
    ap_ranked: dict[float, list[tuple[float, bool]]] = {iou: [] for iou in ap_ious}
    ap_gt_count = 0
    exported_detections: list[dict] = []

    for i, path in enumerate(paths, 1):
        (
            ch,
            names,
            origin,
            cell,
            boxes,
            ignore,
            seg_gt,
            raw_seg_gt,
            src_name,
            split_name,
            box_target_used,
            valid_box_count,
            pseudo_box_count,
        ) = load_tile(path, ch_scale_ck, ck_names, norm_config, args.box_target)
        decode_conf = min(min(thresholds), float(args.ap_min_conf))
        dets_raw, _full_hm, full_seg = run_model(ch, model, dev, args, cell, decode_conf)
        gt_verts = local_box_verts(boxes, origin)
        ignore_verts = local_box_verts(ignore, origin)
        partial_label_tile = bool(args.partial_label_eval and src_name.lower() in partial_sources)
        allowed_seg = rasterize_box_rows(
            boxes, origin, cell, seg_gt.shape, corner_radius_fraction=0.25
        )
        if not allowed_seg.any():
            allowed_seg = seg_gt
        else:
            allowed_seg |= seg_gt
        seg_eval_mask = None
        weak_hole_px = 0
        if partial_label_tile and raw_seg_gt is not None:
            weak_holes = seg_gt & ~raw_seg_gt
            weak_hole_px = int(weak_holes.sum())
            if weak_hole_px > 0:
                seg_eval_mask = ~weak_holes
        segm = segmentation_metrics(
            full_seg,
            seg_gt,
            args.seg_thresh,
            eval_mask=seg_eval_mask,
            observed=raw_seg_gt if partial_label_tile and raw_seg_gt is not None else seg_gt,
            allowed=allowed_seg,
        )
        dets_nms = nms(dets_raw, args.nms_iou)  # once per tile; see evaluate_dets
        _matched_gt, matched_det, matched_ignore = match_detections_one_to_one(
            dets_nms, gt_verts, ignore_verts, args.match_iou
        )
        match_cache = (matched_det, matched_ignore)
        if args.export_detections:
            frame = path.stem
            if frame.startswith("wisar_"):
                frame = frame[len("wisar_") :].split("_", 1)[0]
            for det_idx, det in enumerate(dets_nms):
                if det["score"] < args.ap_min_conf:
                    continue
                gt_iou = float(matched_det[det_idx])
                ignore_iou = float(matched_ignore[det_idx])
                if ignore_iou >= args.match_iou:
                    match_status = "ignored"
                elif gt_iou >= args.match_iou:
                    match_status = "tp"
                else:
                    match_status = "novel" if partial_label_tile else "fp"
                exported_detections.append(
                    {
                        "frame": frame,
                        "source": src_name,
                        "match_status": match_status,
                        "matched_gt_iou": gt_iou,
                        "matched_ignore_iou": ignore_iou,
                        "score": float(det["score"]),
                        "cx": float(det["cx"] + origin[0]),
                        "cy": float(det["cy"] + origin[1]),
                        "length": float(det["L"]),
                        "width": float(det["W"]),
                        "yaw": float(det["th"]),
                        "z": float(det["z"]),
                        "height": float(det["height"]),
                    }
                )
        if not partial_label_tile:
            ap_gt_count += len(gt_verts)
            ap_keep = np.fromiter(
                (d["score"] >= args.ap_min_conf for d in dets_nms),
                dtype=bool,
                count=len(dets_nms),
            )
            ap_dets = [d for d, selected in zip(dets_nms, ap_keep, strict=True) if selected]
            for ap_iou in ap_ious:
                if abs(ap_iou - args.match_iou) < 1e-12:
                    adet, aignore = matched_det[ap_keep], matched_ignore[ap_keep]
                else:
                    _agt, adet, aignore = match_detections_one_to_one(
                        ap_dets, gt_verts, ignore_verts, ap_iou
                    )
                for di, det in enumerate(ap_dets):
                    if adet[di] < ap_iou and aignore[di] >= ap_iou:
                        continue
                    ap_ranked[ap_iou].append((float(det["score"]), bool(adet[di] >= ap_iou)))
        per_thr = {}
        for thr in thresholds:
            m, dets, best_det, best_ignore = evaluate_dets(
                dets_nms,
                gt_verts,
                ignore_verts,
                thr,
                args.match_iou,
                partial_label=partial_label_tile,
                precomputed_matches=match_cache,
            )
            if args.mask_obb_report:
                m["mask_obb_candidates"] = conservative_mask_obb_count(full_seg, dets, cell)
            else:
                m["mask_obb_candidates"] = 0
            per_thr[thr] = (m, dets, best_det, best_ignore)
            for k in agg[thr]:
                agg[thr][k] += float(m[k]) if k == "matched_iou_sum" else int(m[k])

        for thr in thresholds:
            sm, _tdets, _tbest_det, _tbest_ignore = per_thr[thr]
            tile_rows_by_thr[thr].append(
                {
                    "tile": path.stem,
                    "source": src_name,
                    "split": split_name,
                    "box_target": box_target_used,
                    "gt_boxes": len(boxes),
                    "valid_boxes": valid_box_count,
                    "pseudo_boxes": pseudo_box_count,
                    "ignore_boxes": len(ignore),
                    "partial_label": int(partial_label_tile),
                    "seg_weak_hole_ignored_px": weak_hole_px,
                    "raw_candidates": len(dets_raw),
                    "conf": thr,
                    **sm,
                    **{
                        k: segm[k]
                        for k in (
                            "seg_iou",
                            "seg_precision",
                            "seg_recall",
                            "seg_partial_iou",
                            "seg_inter",
                            "seg_union",
                            "seg_gt_px",
                            "seg_pred_px",
                            "seg_pred_in_dil_px",
                            "seg_observed_tp",
                            "seg_observed_px",
                            "seg_predicted_px",
                            "seg_predicted_inside_allowed_px",
                            "seg_predicted_outside_allowed_px",
                            "seg_allowed_px",
                            "seg_trusted_intersection_px",
                            "seg_trusted_union_px",
                            "seg_trusted_target_px",
                            "seg_observed_recall",
                            "seg_obb_containment",
                            "seg_outside_allowed_fraction",
                            "seg_footprint_fill_ratio",
                            "seg_trusted_iou",
                        )
                    },
                }
            )
        # Visual samples are re-scored after the loop at the headline operating
        # threshold, so the overlays show exactly what the reported metrics
        # count (a fixed mid-threshold here used to hide low-conf TPs and made
        # recall look worse on the pictures than in the numbers).
        if partial_label_tile:
            for thr in thresholds:
                _tm, tdets, tbest_det, tbest_ignore = per_thr[thr]
                for di, d in enumerate(tdets):
                    if tbest_ignore[di] >= args.match_iou or tbest_det[di] >= args.match_iou:
                        continue
                    novel_rows_by_thr[thr].append(
                        {
                            "tile": path.stem,
                            "source": src_name,
                            "split": split_name,
                            "conf": thr,
                            "score": float(d["score"]),
                            "hm_score": float(d.get("hm_score", d["score"])),
                            "seg_center_score": float(d.get("seg_center_score", 1.0)),
                            "seg_obb_support": float(d.get("seg_obb_support", 1.0)),
                            "cx": float(d["cx"]),
                            "cy": float(d["cy"]),
                            "L": float(d["L"]),
                            "W": float(d["W"]),
                            "aspect": float(max(d["L"], d["W"]) / max(min(d["L"], d["W"]), 1e-6)),
                            "theta_rad": float(d["th"]),
                            "height": float(d.get("height", 0.0)),
                            "z": float(d.get("z", 0.0)),
                        }
                    )
        if args.viz_tiles == 0 or len(tile_cache) < args.viz_tiles:
            # Keep only the single display channel; caching the full (C,H,W)
            # tensor for every viz tile held gigabytes on large tiles.
            bg_idx = min(max(display_index(names, args.display_channel), 0), ch.shape[0] - 1)
            tile_cache.append(
                (
                    path,
                    np.ascontiguousarray(ch[bg_idx]),
                    origin,
                    cell,
                    boxes,
                    ignore,
                    dets_nms,
                    partial_label_tile,
                    match_cache,
                )
            )
        print(
            f"[{i}/{len(paths)}] {path.name}: src={src_name} target={box_target_used} "
            f"gt={len(boxes)} cand={len(dets_raw)} seg_iou={segm['seg_iou']:.3f}",
            flush=True,
        )

    summary_rows = []
    for thr in thresholds:
        a = agg[thr]
        p = a["tp"] / max(a["tp"] + a["fp"], 1)
        r = a["tp"] / max(a["tp"] + a["fn"], 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        mean_iou = a["matched_iou_sum"] / max(a["matched_iou_count"], 1)
        summary_rows.append(
            {
                "conf": thr,
                "tp": a["tp"],
                "fp": a["fp"],
                "fn": a["fn"],
                "apparent_fp": a["apparent_fp"],
                "novel_candidates": a["novel_candidates"],
                "detections": a["detections"],
                "eval_detections": a["eval_detections"],
                "precision": p,
                "recall": r,
                "f1": f1,
                "matched_iou_mean": mean_iou,
                "matched_iou_count": a["matched_iou_count"],
            }
        )
    best_summary, tile_rows, select_note = select_headline_summary(
        tile_rows_by_thr, thresholds, paths, args.conf
    )
    op_thr = float(best_summary["conf"])
    # Use ALL tiles for per-source breakdown (not just the held-out subset used for
    # the headline number) — small-source datasets (ISPRS: 3 tiles) would otherwise
    # have 0-1 tiles in source_metrics, making the per-source CSV misleading.
    all_tile_rows_at_best_thr = tile_rows_by_thr[op_thr]
    source_rows = source_summaries(all_tile_rows_at_best_thr, op_thr)
    # novel candidates at the headline operating threshold for consistency with metrics
    novel_rows = novel_rows_by_thr[op_thr]
    print(f"[threshold] {select_note}")
    seg_summary = _seg_summary_from_rows(all_tile_rows_at_best_thr)
    ap_metrics = {
        f"ap_rIoU_{iou:.2f}": average_precision_from_ranked(ap_ranked[iou], ap_gt_count)
        for iou in ap_ious
    }
    final = {
        "weights": str(Path(args.weights).resolve()),
        "normalization": norm_config["mode"],
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "source": args.source,
        "pattern": pattern,
        "tiles": len(paths),
        "match_iou": args.match_iou,
        "nms_iou": args.nms_iou,
        "seg_thresh": args.seg_thresh,
        "decode_score_mode": args.decode_score_mode,
        "partial_label_eval": bool(args.partial_label_eval),
        "partial_label_sources": sorted(partial_sources),
        "novel_candidates_at_op_threshold": len(novel_rows),
        "best_detection": best_summary,
        "average_precision": {
            "definition": "all-point interpolated AP after rotated NMS and per-tile one-to-one matching",
            "min_confidence": args.ap_min_conf,
            "gt_boxes": ap_gt_count,
            **ap_metrics,
        },
        "threshold_selection": select_note,
        "segmentation": seg_summary,
        "source_metrics": source_rows,
    }

    with (out_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with (out_dir / "tile_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()))
        w.writeheader()
        w.writerows(tile_rows)
    if source_rows:
        with (out_dir / "source_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
            w.writeheader()
            w.writerows(source_rows)
        (out_dir / "source_summary.json").write_text(
            json.dumps(source_rows, indent=2), encoding="utf-8"
        )
    if novel_rows:
        with (out_dir / "novel_candidates.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(novel_rows[0].keys()))
            w.writeheader()
            w.writerows(novel_rows)
    (out_dir / "summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"checkpoint: {final['weights']}",
                f"dataset: {final['dataset_dir']}",
                f"source/pattern: {args.source} / {pattern}",
                f"tiles: {len(paths)}",
                f"decode score mode: {args.decode_score_mode}",
                f"threshold selection: {select_note}",
                f"reported conf: {best_summary['conf']:.3f}",
                f"det precision/recall/F1: {best_summary['precision']:.4f} / "
                f"{best_summary['recall']:.4f} / {best_summary['f1']:.4f}",
                f"det TP/FP/FN: {best_summary['tp']} / {best_summary['fp']} / {best_summary['fn']}",
                f"matched rotated IoU mean: {best_summary['matched_iou_mean']:.4f} "
                f"(n={best_summary['matched_iou_count']})",
                *[
                    f"AP@rotated-IoU {iou:.2f}: {ap_metrics[f'ap_rIoU_{iou:.2f}']:.4f}"
                    for iou in ap_ious
                ],
                f"partial-label eval: {bool(args.partial_label_eval)} sources={','.join(sorted(partial_sources))}",
                f"apparent FP / novel candidates: {best_summary.get('apparent_fp', 0)} / "
                f"{best_summary.get('novel_candidates', 0)}",
                f"novel candidate CSV rows at op threshold ({op_thr:.3f}): {len(novel_rows)}",
                f"seg IoU/P/R: {seg_summary['seg_iou']:.4f} / "
                f"{seg_summary['seg_precision']:.4f} / {seg_summary['seg_recall']:.4f}",
                f"seg Dice: {seg_summary['seg_dice']:.4f}",
                f"seg partial IoU (GT+{SEG_PARTIAL_DILATION_PX}px buffer): "
                f"{seg_summary['seg_partial_iou']:.4f}",
                f"seg observed recall / OBB containment: "
                f"{seg_summary['seg_observed_recall']:.4f} / "
                f"{seg_summary['seg_obb_containment']:.4f}",
                f"seg outside-allowed fraction / footprint fill: "
                f"{seg_summary['seg_outside_allowed_fraction']:.4f} / "
                f"{seg_summary['seg_footprint_fill_ratio']:.4f}",
                f"seg weak-hole ignored px: {seg_summary.get('seg_weak_hole_ignored_px', 0)}",
                "source metrics:",
                *[
                    f"  {r['source']}{'[partial]' if r.get('partial_label') else ''}: "
                    f"tiles={r['tiles']} target={r['box_target']} "
                    f"F1={r['f1']:.4f} recall={r['recall']:.4f} "
                    f"novel={r.get('novel_candidates', 0)} "
                    f"segIoU={r['seg_iou']:.4f} "
                    f"segPartialIoU={r['seg_partial_iou']:.4f} "
                    f"weakHoleIgnoredPx={r.get('seg_weak_hole_ignored_px', 0)} "
                    f"gt={r['gt_boxes']}"
                    for r in source_rows
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                f"# Model evaluation — {Path(args.weights).stem}",
                "",
                f"Evaluated on **{len(paths)}** `{args.source}` tiles from split **{args.split}**.",
                f"Matching uses one-to-one rotated-box IoU ≥ **{args.match_iou:.2f}**; "
                f"NMS IoU is **{args.nms_iou:.2f}**.",
                "",
                "## Headline result",
                "",
                "| Confidence | Precision | Recall | F1 | TP | FP | FN | Mean matched IoU |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                f"| {best_summary['conf']:.2f} | {best_summary['precision']:.1%} | "
                f"{best_summary['recall']:.1%} | {best_summary['f1']:.1%} | "
                f"{best_summary['tp']} | {best_summary['fp']} | {best_summary['fn']} | "
                f"{best_summary['matched_iou_mean']:.3f} |",
                "",
                f"Threshold policy: {select_note}.",
                "",
                "## Average precision",
                "",
                "| Metric | AP |",
                "|---|---:|",
                *[
                    f"| AP@rotated-IoU {iou:.2f} | {ap_metrics[f'ap_rIoU_{iou:.2f}']:.1%} |"
                    for iou in ap_ious
                ],
                "",
                "## Segmentation",
                "",
                "| IoU | Dice | Pixel precision | Pixel recall |",
                "|---:|---:|---:|---:|",
                f"| {seg_summary['seg_iou']:.1%} | {seg_summary['seg_dice']:.1%} | "
                f"{seg_summary['seg_precision']:.1%} | "
                f"{seg_summary['seg_recall']:.1%} |",
                "",
                "## How to read these metrics",
                "",
                "- **Precision:** of the reported vehicle detections, how many matched a labelled vehicle.",
                "- **Recall:** of the labelled vehicles, how many were detected.",
                "- **F1:** harmonic mean of precision and recall; useful as one balanced score.",
                "- **Mean matched IoU:** average geometric overlap of true-positive rotated boxes only.",
                "- **Segmentation IoU:** pixel overlap divided by pixel union for vehicle footprints.",
                "- TP/FP/FN are micro-aggregated counts across tiles, not averages of tile percentages.",
                "",
                "Raw details: `summary.json`, `threshold_metrics.csv`, `tile_metrics.csv`, and `source_metrics.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    save_plots(out_dir, thresholds, summary_rows, tile_rows, args.source)
    if args.export_detections:
        export_path = Path(args.export_detections)
        if not export_path.is_absolute():
            export_path = out_dir / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "frame", "source", "match_status", "matched_gt_iou", "matched_ignore_iou", "score",
            "cx", "cy", "length", "width", "yaw", "z", "height",
        ]
        with export_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(exported_detections)
        review_fields = [
            "frame", "source", "operating_confidence", "score", "matched_gt_iou",
            "matched_ignore_iou", "cx", "cy", "length", "width", "yaw", "z", "height",
            "audit_category", "audit_note", "reviewed",
        ]
        review_rows = [
            {
                **{key: row[key] for key in review_fields if key in row},
                "operating_confidence": op_thr,
                "audit_category": "",
                "audit_note": "",
                "reviewed": "",
            }
            for row in exported_detections
            if row["match_status"] == "fp" and float(row["score"]) >= op_thr
        ]
        with (out_dir / "fp_review_queue.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=review_fields)
            writer.writeheader()
            writer.writerows(review_rows)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    viz_conf = float(best_summary["conf"])
    for item in tile_cache:
        path, bg, origin, cell, boxes, ignore, dets_nms, partial, match_cache = item
        _m, dets, best_det, best_ignore = evaluate_dets(
            dets_nms,
            local_box_verts(boxes, origin),
            local_box_verts(ignore, origin),
            viz_conf,
            args.match_iou,
            partial_label=partial,
            precomputed_matches=match_cache,
        )
        save_overlay(
            viz_dir / f"{path.stem}_overlay.png",
            bg,
            origin,
            cell,
            boxes,
            ignore,
            dets,
            best_det,
            best_ignore,
            args,
            viz_conf,
        )

    print((out_dir / "summary.txt").read_text(encoding="utf-8"))
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
