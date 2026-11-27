"""Training logs, plots, and epoch visualisation."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

from bordereye import plotstyle
from bordereye.ml.bev.core.consistency import (
    decode_topk_boxes,
    decode_topk_teacher_angle,
    seg_obb_consistency_loss,
)
from bordereye.ml.bev.core.nn_contract import (
    STRIDE,
    box_corners,
    select_class_regression,
    unpack_model_output,
)
from bordereye.ml.bev.data.paths import output_root, project_root


def safe_run_name(value: str) -> str:
    stem = Path(value).stem or "run"
    keep = []
    for ch in stem:
        keep.append(ch if ch.isalnum() or ch in ("-", "_", ".") else "_")
    name = "".join(keep).strip("._")
    return name or "run"


def resolve_output_dir(args) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
        return out if out.is_absolute() else project_root() / out
    group = "_tests" if args.smoke else "runs"
    return output_root() / group / safe_run_name(args.out_name)


def _read_log_rows(log_path: Path, run_id: str) -> list[dict]:
    rows = []
    with log_path.open("r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("run_id") != run_id:
                continue
            try:
                row = {"run_id": r.get("run_id", ""), "epoch": float(r.get("epoch", 0))}
                numeric = (
                    "train_objective", "train_baseline_loss", "train_loss",
                    "train_hm", "train_reg", "train_seg",
                    "train_cons", "train_orient", "train_moment", "train_zheight",
                    "train_shadow", "train_seg_explain", "train_seg_center", "train_greg",
                    "val_loss", "val_det", "val_det_ema", "val_seg_labelfree",
                    "val_hm", "val_reg", "val_seg", "best_val",
                    "val_observed_recall", "val_obb_containment",
                    "val_outside_allowed_fraction", "val_footprint_fill_ratio",
                    "val_center_error_m", "val_length_abs_error_m",
                    "val_width_abs_error_m", "val_bev_iou", "val_iou3d",
                    "val_z_abs_error_m", "val_height_abs_error_m",
                    "train_yaw_mae_deg", "val_yaw_mae_deg", "train_orient_active_frac",
                    "train_orient_mean_aniso",
                    "consistency_weight", "orient_weight", "moment_weight",
                    "z_height_weight", "seg_explain_weight", "seg_center_weight",
                    "gaussian_reg_weight",
                    "zheight_active_frac", "seg_explain_active_frac", "seg_explain_mean_area",
                    "seg_explain_mean_reliability", "seg_explain_mean_aniso",
                    "seg_center_active_frac", "seg_center_components",
                    "seg_center_mean_reliability", "seg_center_mean_aniso",
                    "zheight_mean_height_teacher", "lr", "grad_norm", "epoch_seconds",
                    "checkpoint_saved", "bad_epochs",
                )
                for k in numeric:
                    row[k] = float(r.get(k, 0.0) or 0.0)
                if row["train_objective"] == 0.0:
                    row["train_objective"] = row["train_loss"]
                if row["train_baseline_loss"] == 0.0:
                    row["train_baseline_loss"] = (
                        row["train_hm"] + row["train_reg"] + 0.5 * row["train_seg"])
                rows.append(row)
            except (KeyError, ValueError):
                continue
    return rows


def append_experiment_log(log_path: Path, row: dict) -> None:
    """Append a row while preserving a union CSV schema across experiment versions."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    rows = []
    fieldnames = list(row.keys())
    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            fieldnames = list(reader.fieldnames)
        if reader.fieldnames and all(k in fieldnames for k in row):
            with log_path.open("a", newline="", encoding="utf-8") as fa:
                csv.DictWriter(fa, fieldnames=fieldnames).writerow(
                    {k: row.get(k, "") for k in fieldnames})
            return
        for old in reader:
            rows.append(old)
    for key in row:
        if key not in fieldnames:
            fieldnames.append(key)
    rows.append({k: row.get(k, "") for k in fieldnames})
    tmp = log_path.with_suffix(log_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for old in rows:
            writer.writerow({k: old.get(k, "") for k in fieldnames})
    try:
        tmp.replace(log_path)
    except PermissionError:
        print(f"[log warning] Could not rewrite {log_path.name}; appending with existing CSV schema.")
        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(
                {k: row.get(k, "") for k in fieldnames})


# Plot theme lives in bordereye.plotstyle so a training dashboard, a research
# audit and a paper figure come out looking like one project.
PLOT_STYLE = plotstyle.rc_params()


def _panel_grid(fig, n_panels: int, n_cols: int = 2):
    """Axes for exactly *n_panels* charts, laid out in *n_cols* columns.

    Panels are built on demand rather than from a fixed 3x2 grid: a short or
    ablated run used to render empty framed boxes for metrics it never logged,
    which read as "this collapsed to zero" rather than "not measured".
    """
    n_rows = -(-n_panels // n_cols)
    axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()
    for extra in axes[n_panels:]:
        extra.set_visible(False)
    return axes[:n_panels]


def make_curves_plot(log_path: Path, plot_path: Path, run_id: str) -> None:
    """Standard ML training dashboard for one consistency run."""
    rows = _read_log_rows(log_path, run_id)
    if not rows:
        return
    x = np.asarray([r["epoch"] for r in rows], dtype=float) + 1.0   # 1-based epochs

    def col(name):
        return np.asarray([r[name] for r in rows], dtype=float)

    def has(name):
        y = col(name)
        return bool(np.isfinite(y).any() and np.any(y))

    plt.rcParams.update(PLOT_STYLE)

    # Best validation epoch (the saved model) is tracked on the EMA-smoothed
    # det-specific metric. Older logs without val_det_ema fall back to val_det,
    # then to total validation loss.
    vloss = col("val_loss")
    vdet = col("val_det")
    if not np.any(vdet):
        vdet = vloss
    vdet_ema = col("val_det_ema")
    save_metric = vdet_ema if np.isfinite(vdet_ema).any() and np.any(vdet_ema) else vdet
    best_i = int(np.nanargmin(save_metric)) if len(save_metric) else 0

    SELF_SUP = (("train_cons", "seg-OBB consistency"), ("train_orient", "orientation"),
                ("train_moment", "moment"), ("train_zheight", "z / height"),
                ("train_shadow", "shadow-OBB"), ("train_greg", "gaussian reg"),
                ("train_seg_explain", "mask-anchor explain"),
                ("train_seg_center", "mask-anchor centre"))
    ANCHORS = (("seg_center_active_frac", "centre anchors active"),
               ("seg_explain_active_frac", "explain anchors active"),
               ("seg_center_mean_reliability", "anchor reliability"))
    TEACHER = (("train_orient_active_frac", "teacher active fraction"),
               ("train_orient_mean_aniso", "teacher anisotropy"))

    # Every measure gets its own panel with its own scale. Twinned y-axes were
    # removed here: pairing learning rate with gradient norm (or a loss with a
    # [0,1] diagnostic) on one frame implies a correspondence between two
    # arbitrary scales that the data does not contain.
    panels = ["objective", "optimizer", "supervised"]
    if any(has(n) for n, _ in SELF_SUP):
        panels.append("selfsup")
    if any(has(n) for n, _ in ANCHORS):
        panels.append("anchors")
    if has("val_yaw_mae_deg") or has("train_yaw_mae_deg"):
        panels.append("yaw")
    if any(has(n) for n, _ in TEACHER):
        panels.append("teacher")
    panels.append("lr")
    if has("grad_norm"):
        panels.append("grad")

    n_rows = -(-len(panels) // 2)
    fig = plt.figure(figsize=(13.5, 3.5 * n_rows + 1.0))
    axes = _panel_grid(fig, len(panels))
    at = dict(zip(panels, axes, strict=True))

    S = plotstyle.SERIES

    # Headline: train vs validation; the det-specific metric drives the best mark.
    ax0 = at["objective"]
    ax0.plot(x, col("train_baseline_loss"), label="train", color=S[0], lw=2.0)
    ax0.plot(x, vloss, label="validation (total)", color=S[1], lw=1.4, alpha=0.85)
    ax0.plot(x, vdet, label="validation det raw", color=S[2], lw=1.4, alpha=0.85)
    ax0.plot(x, save_metric, label="validation det EMA (saved)", color=S[3], lw=2.2)
    ax0.plot(x, col("best_val"), label="best-so-far", color=plotstyle.INK_MUTED, lw=1.0)
    if has("val_seg_labelfree"):
        ax0.plot(x, col("val_seg_labelfree"), label="validation seg label-free",
                 color=S[4], lw=1.4)
    saved_idx = np.where(col("checkpoint_saved") > 0.5)[0]
    if len(saved_idx):
        ax0.scatter(x[saved_idx], save_metric[saved_idx], s=22, zorder=4,
                    color=S[3], edgecolor="white", linewidth=0.7, label="checkpoint saved")
    if len(save_metric):
        # The saved model is the one fact a reader looks for: label it on the
        # point instead of spending a legend row on it.
        ax0.scatter([x[best_i]], [save_metric[best_i]], s=70, zorder=5,
                    color=plotstyle.STATUS["good"], edgecolor="white", linewidth=1.2)
        plotstyle.annotate_point(
            ax0, x[best_i], save_metric[best_i],
            f"best saved · ep {int(x[best_i])} · {save_metric[best_i]:.3f}",
            color=plotstyle.STATUS["good"])
    ax0.set_title("Detection objective — train vs validation")
    ax0.set_ylabel("detection objective loss")

    # Optimizer objective vs its supervised part.
    ax1 = at["optimizer"]
    ax1.plot(x, col("train_objective"), label="optimizer objective", color=S[0], lw=2.0)
    ax1.plot(x, col("train_baseline_loss"), label="supervised part", color=S[1], lw=1.6)
    ax1.set_yscale("symlog", linthresh=1.0)
    ax1.set_title("Optimizer objective vs supervised part")
    ax1.set_ylabel("loss (symlog)")

    # Supervised components: train solid, validation dashed, same hue per head.
    ax2 = at["supervised"]
    for (name, label, c), vname in ((("train_hm", "heatmap", S[0]), "val_hm"),
                                    (("train_reg", "regression", S[1]), "val_reg"),
                                    (("train_seg", "segmentation", S[2]), "val_seg")):
        if has(name):
            ax2.plot(x, col(name), label=label, lw=1.8, color=c)
        if has(vname):
            ax2.plot(x, col(vname), lw=1.3, ls="--", color=c, alpha=0.9)
    ax2.plot([], [], color=plotstyle.INK_MUTED, lw=1.6, label="train (solid)")
    ax2.plot([], [], color=plotstyle.INK_MUTED, lw=1.3, ls="--", label="validation (dashed)")
    ax2.set_title("Supervised loss components")
    ax2.set_ylabel("loss")

    if "selfsup" in at:
        ax3 = at["selfsup"]
        for i, (name, label) in enumerate(n for n in SELF_SUP if has(n[0])):
            ax3.plot(x, col(name), label=label, color=S[i % len(S)], lw=1.6)
        ax3.set_yscale("symlog", linthresh=1.0)
        ax3.set_title("Self-supervised / mask-anchor losses")
        ax3.set_ylabel("loss (symlog)")

    if "anchors" in at:
        ax = at["anchors"]
        for i, (name, label) in enumerate(n for n in ANCHORS if has(n[0])):
            ax.plot(x, col(name), label=label, color=S[i % len(S)], lw=1.6)
        ax.set_ylim(0, 1)
        ax.set_title("Mask-anchor diagnostics")
        ax.set_ylabel("fraction / reliability [0,1]")

    if "yaw" in at:
        ax = at["yaw"]
        if has("train_yaw_mae_deg"):
            ax.plot(x, col("train_yaw_mae_deg"), label="train", color=S[0], lw=1.8)
        if has("val_yaw_mae_deg"):
            ax.plot(x, col("val_yaw_mae_deg"), label="validation", color=S[1], lw=2.0)
        ax.set_title("OBB orientation quality")
        ax.set_ylabel("yaw MAE (degrees)")

    if "teacher" in at:
        ax = at["teacher"]
        for i, (name, label) in enumerate(n for n in TEACHER if has(n[0])):
            ax.plot(x, col(name), label=label, color=S[i % len(S)], lw=1.6)
        ax.set_title("Orientation teacher diagnostics")
        ax.set_ylabel("fraction / anisotropy")

    ax = at["lr"]
    ax.plot(x, col("lr"), label="learning rate", color=S[0], lw=1.8)
    ax.set_yscale("log")
    ax.set_title("Learning rate")
    ax.set_ylabel("learning rate (log)")

    if "grad" in at:
        ax = at["grad"]
        ax.plot(x, col("grad_norm"), label="gradient norm", color=S[1], lw=1.6)
        ax.set_title("Gradient norm")
        ax.set_ylabel("gradient norm")

    for name, ax in at.items():
        ax.set_xlabel("epoch")
        plotstyle.epoch_axis(ax)
        # Legends sit above the frame; inside the axes they cover the very
        # region the reader is inspecting (the old panel 1 clipped the
        # validation peak behind its legend box).
        if name != "objective":
            plotstyle.legend_above(ax, n_cols=3)
        else:
            plotstyle.legend_above(ax, n_cols=3)

    fig.suptitle(
        f"BEV consistency training — run {run_id}\n"
        f"epoch {int(x[-1])} · saved metric {save_metric[-1]:.3f} · "
        f"best {save_metric[best_i]:.3f} (ep {int(x[best_i])})",
        fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.subplots_adjust(hspace=0.62, wspace=0.22)
    plotstyle.save(fig, plot_path, dpi=plotstyle.DPI_SCREEN)


def _target_boxes_from_maps(reg_np, mask_np, cell: float, stride: int = STRIDE,
                            min_weight=1e-6):
    ys, xs = np.nonzero(mask_np > min_weight)
    out = []
    for iy, ix in zip(ys.tolist(), xs.tolist(), strict=True):
        dx, dy, logL, logW, s2, c2 = reg_np[:6, iy, ix]
        Lg = np.exp(np.clip(logL, np.log(0.5), np.log(20.0)))
        Wg = np.exp(np.clip(logW, np.log(0.5), np.log(20.0)))
        out.append({
            "px": float(ix), "py": float(iy),
            "cx": float(ix + dx), "cy": float(iy + dy),
            "Lg": float(Lg / cell / stride),
            "Wg": float(Wg / cell / stride),
            "theta": float(0.5 * np.arctan2(s2, c2)),
            "weight": float(mask_np[iy, ix]),
        })
    return out


def _mask_iou(a, b, thr=0.5):
    aa = np.asarray(a) >= thr
    bb = np.asarray(b) >= thr
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return inter / max(union, 1)


def _center_points_from_heatmap(hm_np, thresh=0.05):
    if hm_np is None or hm_np.size == 0:
        return np.zeros((0, 2), np.int64)
    pts = []
    H, W = hm_np.shape
    for y in range(H):
        for x in range(W):
            v = hm_np[y, x]
            if v <= thresh:
                continue
            y0, y1 = max(0, y - 1), min(H, y + 2)
            x0, x1 = max(0, x - 1), min(W, x + 2)
            if v >= hm_np[y0:y1, x0:x1].max():
                pts.append((y, x))
    return np.asarray(pts, np.int64).reshape(-1, 2)


@torch.no_grad()
def _debug_heatmap(logits):
    """Class-agnostic diagnostic peaks; not the benchmark postprocessor."""
    merged = logits.amax(dim=1, keepdim=True)
    local_max = torch.nn.functional.max_pool2d(merged, 3, stride=1, padding=1)
    return merged, merged.masked_fill(merged < local_max, -80.0)


def save_epoch_viz(model, sample, dev, out_path: Path, args, occ_idx: int,
                   ep: int, cell: float) -> None:
    """Six-panel per-epoch debug dashboard for segmentation, centres, and OBBs."""
    model.eval()
    stride = int(getattr(model, "output_stride", STRIDE))
    if isinstance(sample, (tuple, list)):
        crop_t, hm_t, reg_t, mask_t, _igm_t, seg_t = sample[:6]
        crop = crop_t.unsqueeze(0).to(dev)
        hm_target = hm_t.amax(dim=0).cpu().numpy()
        reg_target = reg_t.cpu().numpy()
        mask_target = mask_t[0].cpu().numpy()
        seg_target = seg_t[0].cpu().numpy()
    else:
        crop = sample.to(dev)
        hm_target = reg_target = mask_target = seg_target = None
    phm, preg_raw, pseg = unpack_model_output(model(crop))
    preg = select_class_regression(preg_raw, phm.argmax(dim=1))
    nonfinite_out = (
        int((~torch.isfinite(phm)).sum().item())
        + int((~torch.isfinite(preg_raw)).sum().item())
        + int((~torch.isfinite(pseg)).sum().item())
    )
    phm_viz = torch.nan_to_num(phm, nan=-80.0, posinf=80.0, neginf=-80.0)
    # Legacy diagnostic decoders are class-agnostic. For a multi-class model,
    # visualize the strongest class at each spatial cell; never flatten class
    # and spatial dimensions into indices consumed by the shared reg head.
    phm_merged, phm_decode = _debug_heatmap(phm_viz)
    preg_viz = torch.nan_to_num(preg, nan=0.0, posinf=0.0, neginf=0.0)
    pseg_viz = torch.nan_to_num(pseg, nan=-80.0, posinf=80.0, neginf=-80.0)
    seg_prob = torch.sigmoid(pseg_viz)[0, 0].cpu().numpy()
    hm_prob = torch.sigmoid(phm_merged)[0, 0].cpu().numpy()
    viz_thresh = min(float(args.viz_hm_thresh), float(args.consistency_hm_thresh))
    _, union = seg_obb_consistency_loss(
        phm_decode, preg_viz, pseg_viz, cell=cell, stride=stride,
        topk=args.consistency_topk, hm_thresh=viz_thresh,
        sharp=args.consistency_sharp)
    union = union[0].cpu().numpy()
    cx, cy, Lg, Wg, th, score = decode_topk_boxes(
        phm_decode, preg_viz, cell, stride, topk=args.consistency_topk,
        hm_thresh=viz_thresh)
    cx, cy, Lg, Wg, th, score = (t[0].cpu().numpy() for t in (cx, cy, Lg, Wg, th, score))
    # Exactly the decoder's tensor/order, including tied scores.
    flat_order = torch.sigmoid(phm_decode).flatten(1).topk(len(score), dim=1).indices[0].cpu().numpy()
    peak_y = (flat_order // hm_prob.shape[1]).astype(np.float32)
    peak_x = (flat_order % hm_prob.shape[1]).astype(np.float32)
    keep = score > viz_thresh
    # teacher orientation from the local seg axis (what the orientation loss targets)
    tcx, tcy, tth, taniso, tmass = decode_topk_teacher_angle(
        phm_decode, pseg_viz, topk=args.consistency_topk, win_radius=args.orient_win_radius)
    tcx, tcy, tth, taniso, tmass = (t[0].cpu().numpy() for t in (tcx, tcy, tth, taniso, tmass))

    occ = crop[0, occ_idx].cpu().numpy()
    OS = seg_prob.shape[0]
    tgt_boxes = (_target_boxes_from_maps(reg_target, mask_target, cell, stride)
                 if reg_target is not None and mask_target is not None else [])
    target_centers = _center_points_from_heatmap(hm_target, 0.05)
    box_iou_pred = _mask_iou(seg_prob, union, 0.5)
    box_iou_target = _mask_iou(seg_target, union, 0.5) if seg_target is not None else 0.0
    pred_area = int((seg_prob >= 0.5).sum())
    target_area = int((seg_target >= 0.5).sum()) if seg_target is not None else 0
    peak_count = int(keep.sum())

    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    ax = axes.ravel()

    # (1) input channel + target footprint contour
    im0 = ax[0].imshow(occ, cmap="cividis", origin="upper")
    if seg_target is not None and seg_target.max() > 0:
        seg_on_input = np.repeat(
            np.repeat(seg_target, stride, axis=0), stride, axis=1,
        )
        seg_on_input = seg_on_input[:occ.shape[0], :occ.shape[1]]
        ax[0].contour(seg_on_input, levels=[0.5], colors=["#ffffff"], linewidths=0.7, alpha=0.85)
    ax[0].set_title("Input cue + target footprint")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # (2) segmentation target
    if seg_target is not None:
        im_t = ax[1].imshow(seg_target, cmap="Greens", origin="upper", vmin=0, vmax=1)
        ax[1].set_title(f"Segmentation target (area={target_area})")
        fig.colorbar(im_t, ax=ax[1], fraction=0.046, pad=0.04)
    else:
        ax[1].text(0.5, 0.5, "no segmentation target", transform=ax[1].transAxes,
                   ha="center", va="center", fontsize=12)
        ax[1].set_title("Segmentation target")

    # (3) predicted segmentation
    im1 = ax[2].imshow(seg_prob, cmap="magma", origin="upper", vmin=0, vmax=1)
    ax[2].contour(seg_prob, levels=[0.5], colors=["#ffffff"], linewidths=0.7, alpha=0.8)
    ax[2].set_title(f"Predicted segmentation (max={seg_prob.max():.3f}, area={pred_area})")
    fig.colorbar(im1, ax=ax[2], fraction=0.046, pad=0.04).set_label("P(object footprint)")

    # (4) centre heatmap with target/predicted centres
    im_h = ax[3].imshow(hm_prob, cmap="inferno", origin="upper", vmin=0, vmax=max(0.25, hm_prob.max()))
    for i in range(len(cx)):
        if not keep[i]:
            continue
        px = peak_x[i] if i < len(peak_x) else cx[i]
        py = peak_y[i] if i < len(peak_y) else cy[i]
        ax[3].plot(px, py, marker="+", color="#ff1744", markersize=8, mew=1.5)
        ax[3].plot(cx[i], cy[i], marker="o", color="#00e5ff", markersize=3.5)
        ax[3].plot([px, cx[i]], [py, cy[i]], color="#00e5ff", lw=0.7, alpha=0.75)
        ax[3].text(cx[i] + 1.5, cy[i], f"{score[i]:.2f}", color="white", fontsize=6)
    for ty, tx in target_centers:
        ax[3].plot(tx, ty, marker="o", markerfacecolor="none", markeredgecolor="#ffffff",
                   markersize=6, mew=1.1)
    ax[3].set_title(f"Center heatmap (max={hm_prob.max():.3f}, peaks>={viz_thresh:.2f}: {peak_count})")
    ax[3].plot([], [], marker="+", color="#ff1744", ls="", label="pred center peak")
    ax[3].plot([], [], marker="o", color="#00e5ff", ls="", label="offset OBB center")
    ax[3].plot([], [], marker="o", markerfacecolor="none", markeredgecolor="#ffffff",
               ls="", label="target center")
    ax[3].legend(loc="upper right", fontsize=7.5)
    fig.colorbar(im_h, ax=ax[3], fraction=0.046, pad=0.04)

    # (5) predicted segmentation + decoded OBBs + centres + target boxes
    ax[4].imshow(seg_prob, cmap="magma", origin="upper", vmin=0, vmax=1)
    n_box = 0
    for i in range(len(cx)):
        if not keep[i]:
            continue
        # decoded theta is WORLD-frame; negate to draw on the IMAGE grid (row down)
        corners = box_corners(cx[i], cy[i], Lg[i], Wg[i], -th[i])
        alpha = 0.4 + 0.6 * min(1.0, max(0.0, float(score[i])))
        ax[4].add_patch(Polygon(corners, closed=True, fill=False,
                                edgecolor="#00e5ff", lw=1.4, alpha=alpha))
        ax[4].plot(cx[i], cy[i], marker="o", color="#00e5ff", markersize=3.5)
        px = peak_x[i] if i < len(peak_x) else cx[i]
        py = peak_y[i] if i < len(peak_y) else cy[i]
        ax[4].plot([px, cx[i]], [py, cy[i]], color="#ff1744", lw=0.7, alpha=0.75)
        ax[4].plot(px, py, marker="+", color="#ff1744", markersize=7, mew=1.2)
        ax[4].text(cx[i], cy[i], f"{score[i]:.2f}", color="white",
                   fontsize=6, ha="center", va="center")
        if taniso[i] > args.orient_min_aniso and tmass[i] > args.orient_min_mass:
            hl = max(Lg[i] * 0.5, 2.0)
            dxx, dyy = np.cos(tth[i]) * hl, np.sin(tth[i]) * hl
            ax[4].plot([tcx[i] - dxx, tcx[i] + dxx], [tcy[i] - dyy, tcy[i] + dyy],
                       color="#ffe600", lw=1.4, alpha=0.9)
        n_box += 1
    for tb in tgt_boxes:
        corners = box_corners(tb["cx"], tb["cy"], tb["Lg"], tb["Wg"], -tb["theta"])
        ax[4].add_patch(Polygon(corners, closed=True, fill=False,
                                edgecolor="#39ff14", lw=1.1, alpha=0.85))
        ax[4].plot(tb["cx"], tb["cy"], marker="o", markerfacecolor="none",
                   markeredgecolor="#ffffff", markersize=5, mew=1.0)
    ax[4].set_xlim(0, OS)
    ax[4].set_ylim(OS, 0)
    ax[4].set_title(f"Diagnostic OBBs (kept={n_box}, target supports={len(tgt_boxes)})")
    ax[4].plot([], [], color="#00e5ff", lw=1.4, label="pred OBB")
    ax[4].plot([], [], color="#39ff14", lw=1.1, label="target OBB")
    ax[4].plot([], [], color="#ffe600", lw=1.4, label="seg axis")
    ax[4].legend(loc="upper right", fontsize=7.2, facecolor="black",
                 labelcolor="white", framealpha=0.5)

    # (6) consistency: target/pred seg versus soft box-union
    base_seg = seg_target if seg_target is not None and seg_target.max() > 0 else seg_prob
    ax[5].imshow(base_seg, cmap="Greens", origin="upper", vmin=0, vmax=1, alpha=0.9)
    ax[5].imshow(np.ma.masked_less(union, 0.05), cmap="Blues",
                 origin="upper", vmin=0, vmax=1, alpha=0.55)
    ax[5].set_title(f"Seg vs soft OBB union (IoU pred={box_iou_pred:.3f}, target={box_iou_target:.3f})")
    ax[5].plot([], [], color="#2ca25f", lw=6, label="target seg" if seg_target is not None else "pred seg")
    ax[5].plot([], [], color="#3182bd", lw=6, label="box union")
    ax[5].legend(loc="upper right", fontsize=7.5)

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"BEV debug dashboard — epoch {ep + 1} (diagnostic, not benchmark NMS)   "
        f"max P(hm)={hm_prob.max():.3f} · max P(seg)={seg_prob.max():.3f} · "
        f"decoded={n_box} · mask/box IoU={box_iou_pred:.3f}"
        + (f" · NONFINITE_OUT={nonfinite_out}" if nonfinite_out else ""),
        fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    model.train()
