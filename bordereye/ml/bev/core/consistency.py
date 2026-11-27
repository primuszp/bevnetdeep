"""
bev_consistency.py — self-supervised segmentation-to-OBB geometry losses.

Idea
----
The segmentation head is supervised on every source with semantic vehicle labels,
while the OBB regression head has few confident instance boxes. If a model
segments vehicles well across domains and the box is geometrically tied to that
segmentation, segmentation-only datasets can still improve centre, size, yaw,
z, and height prediction.

This module makes that tie explicit and differentiable:

  1. Take the predicted heatmap, pick its top-k centre cells.
  2. Decode the OBB the regression head predicts at each of those cells.
  3. Render each OBB as a *soft* (differentiable) rectangle mask on the stride grid.
  4. Compare the soft-union of those rectangles against the predicted vehicle
     segmentation (detached) via a soft-Dice loss.

On a tile with NO box GT, this still produces a gradient on the regression head:
it is pulled towards boxes that cover the (well-supervised) segmentation. The seg
head is the teacher; the box is the student. That is the self-correcting loop.

Anti-collapse notes
-------------------
- `seg` is detached when used as the target, so the seg head is not dragged toward
  a degenerate box (otherwise both heads collapse onto one blob).
- Soft-Dice penalises both under-coverage (box too small -> low recall) and
  over-coverage (box too big -> the box area inflates the denominator), so the box
  cannot trivially win by growing huge or shrinking to a point.
- Only cells whose predicted heatmap probability exceeds `hm_thresh` contribute,
  so background top-k cells do not inject spurious boxes.

The regression layout matches bev_model / train_bev exactly:
    reg = [dx, dy, logL, logW, sin2theta, cos2theta, z, height]
with the centre offset in stride-grid cells and L/W in metres.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from .mask_obb_teacher import DEFAULT_PRIOR, VehiclePrior, split_component_by_length

_ZH_WARNED: set[str] = set()


_CHANNEL_SCALE_BY_NAME = {
    "hag_max": 10.0,
    "hag_mean": 6.0,
    "hag_std": 2.0,
    "hag_max_low": 4.0,
    "hag_min_low": 4.0,
    "hag_range_low": 4.0,
    "hag_mean_low": 4.0,
    "hag_std_low": 2.0,
    "vertical_span_all": 30.0,
    "hag_p05": 4.0,
    "hag_p95": 4.0,
    "veh_hmax": 4.0,
    "all_hmax": 10.0,
}


def _warn_once(key: str, msg: str) -> None:
    if key not in _ZH_WARNED:
        print(msg, flush=True)
        _ZH_WARNED.add(key)


def decode_obb_at_cells(
    reg_sel: torch.Tensor, iy: torch.Tensor, ix: torch.Tensor, cell: float, stride: int
):
    """Decode selected regression vectors into grid-space OBB parameters.

    Args:
        reg_sel: (B, REG_DIM, K) regression vectors gathered at K cells.
        iy, ix:  (B, K) integer row/col indices of those cells.
        cell:    BEV cell size in metres (e.g. 0.25).
        stride:  output stride (e.g. STRIDE=2).

    Returns (each (B, K)):
        cx, cy:  box centre in stride-grid cell-centre units.
        Lg, Wg:  box length / width in stride-grid cells.
        theta:   long-axis angle in radians (from the sin2/cos2 encoding).
    """
    dx, dy = reg_sel[:, 0], reg_sel[:, 1]
    log_min = math.log(0.5)
    log_max = math.log(20.0)
    logL = reg_sel[:, 2].clamp(min=log_min, max=log_max)
    logW = reg_sel[:, 3].clamp(min=log_min, max=log_max)
    sin2, cos2 = reg_sel[:, 4], reg_sel[:, 5]

    # cell-centre convention matches _draw_seg_box: grid cell i has centre i+0.5,
    # and the box centre ox = ix + dx is already in /stride pixel units.
    cx = ix.float() + dx
    cy = iy.float() + dy
    Lg = torch.exp(logL) / cell / stride
    Wg = torch.exp(logW) / cell / stride
    # Avoid undefined atan2 gradients at the exact zero vector. Random network
    # outputs rarely hit this, but zero-initialised smoke tests and some saturated
    # AMP states can; default to 0 rad until the reg head emits a real direction.
    zero_dir = (sin2.abs() + cos2.abs()) < 1e-6
    sin2 = torch.where(zero_dir, torch.zeros_like(sin2), sin2)
    cos2 = torch.where(zero_dir, torch.ones_like(cos2), cos2)
    theta = 0.5 * torch.atan2(sin2, cos2)
    return cx, cy, Lg, Wg, theta


def decode_topk_boxes(
    phm: torch.Tensor,
    preg: torch.Tensor,
    cell: float,
    stride: int,
    topk: int = 32,
    hm_thresh: float = 0.3,
):
    """Decode the top-k heatmap peaks into grid-space OBBs (for viz / inspection).

    Returns each (B, K): cx, cy, Lg, Wg, theta(rad), score(heatmap prob).
    No gradient is intended here — call under torch.no_grad().
    """
    B, _, H, W = phm.shape
    K = min(topk, H * W)
    hm_prob = torch.sigmoid(phm).view(B, -1)
    topv, topi = hm_prob.topk(K, dim=1)
    iy, ix = topi // W, topi % W
    reg_sel = _gather_reg(preg, topi)
    cx, cy, Lg, Wg, theta = decode_obb_at_cells(reg_sel, iy, ix, cell, stride)
    return cx, cy, Lg, Wg, theta, topv


def _topk_peaks(phm: torch.Tensor, topk: int):
    """Select top-k heatmap peaks (no grad). Returns topv,topi,iy,ix all (B,K)."""
    B, _, H, W = phm.shape
    K = min(topk, H * W)
    hm_prob = torch.sigmoid(phm).detach().view(B, -1)
    topv, topi = hm_prob.topk(K, dim=1)
    return topv, topi, topi // W, topi % W


def _gather_reg(preg: torch.Tensor, topi: torch.Tensor) -> torch.Tensor:
    """Gather regression vectors at the flat peak indices. (B,REG,K), keeps grad."""
    reg_flat = preg.view(preg.shape[0], preg.shape[1], -1)
    idx = topi.unsqueeze(1).expand(-1, preg.shape[1], -1)
    return torch.gather(reg_flat, 2, idx)


def _teacher_2theta(
    seg: torch.Tensor,
    iy: torch.Tensor,
    ix: torch.Tensor,
    win_radius: float,
    seg_thresh: float = 0.4,
):
    """Segmentation orientation teacher via blob image moments (no grad).

    For each (iy,ix) peak, take the *connected segmentation blob* that contains
    the peak (seg > seg_thresh), optionally clipped to a circular window of
    radius `win_radius` to bound merged parking rows, and compute the seg-weighted
    2nd-moment (covariance). The principal axis of that covariance is the vehicle
    orientation — the standard blob-orientation method
    (alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob).

    Restricting to the connected blob (rather than a Gaussian window around the
    peak) is what matters: a circular window centred on a car in a parking row
    pulls in the neighbouring cars and rotates the axis along the row. The blob
    is separated from its neighbours by background, so its moments give the
    per-car axis. The teacher is detached, so non-differentiable labelling is fine.

    Returns (each (B,K)):
        sin2t, cos2t: unit 2theta vector of the principal axis.
        aniso:        elongation in [0,1]; ~0 round (ambiguous), ~1 thin line.
        mass:         total seg weight in the blob (gate out empty windows).

    The frame matches render_obb_soft: x = column, y = row, so the returned
    2theta is directly comparable to the box's atan2(sin2,cos2).
    """
    B, H, W = seg.shape
    K = iy.shape[1]
    dev, dt = seg.device, seg.dtype
    R = float(win_radius)

    seg_np = seg.detach().to(torch.float32).cpu().numpy()
    iy_np = iy.detach().cpu().numpy().astype(np.int64)
    ix_np = ix.detach().cpu().numpy().astype(np.int64)
    # Per-peak work is restricted to the window's bounding box rather than the
    # full H×W grid; pixels outside the circular window contribute 0, so the
    # moments are numerically identical to the full-grid computation.
    rad = int(np.ceil(R))

    sin2t = np.zeros((B, K), np.float32)
    cos2t = np.ones((B, K), np.float32)
    aniso = np.zeros((B, K), np.float32)
    mass = np.zeros((B, K), np.float32)

    for b in range(B):
        s = seg_np[b]
        labels, n_lab = ndi.label(s > seg_thresh)
        if n_lab == 0:
            continue
        for k in range(K):
            cy = min(max(int(iy_np[b, k]), 0), H - 1)
            cx = min(max(int(ix_np[b, k]), 0), W - 1)
            comp = labels[cy, cx]
            if comp == 0:
                continue
            y0, y1 = max(cy - rad, 0), min(cy + rad + 1, H)
            x0, x1 = max(cx - rad, 0), min(cx + rad + 1, W)
            sub_yy, sub_xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            in_win = ((sub_xx - cx) ** 2 + (sub_yy - cy) ** 2) <= R * R
            m = (labels[y0:y1, x0:x1] == comp) & in_win
            w = s[y0:y1, x0:x1] * m
            Wsum = float(w.sum())
            if Wsum < 1e-6:
                continue
            dx = (sub_xx + 0.5) - (cx + 0.5)
            dy = (sub_yy + 0.5) - (cy + 0.5)
            mx = (w * dx).sum() / Wsum
            my = (w * dy).sum() / Wsum
            ddx = dx - mx
            ddy = dy - my
            Cxx = (w * ddx * ddx).sum() / Wsum
            Cyy = (w * ddy * ddy).sum() / Wsum
            Cxy = (w * ddx * ddy).sum() / Wsum
            a = Cxx - Cyy
            bb = 2.0 * Cxy
            mag = float(np.hypot(a, bb)) + 1e-9
            cos2t[b, k] = a / mag
            sin2t[b, k] = bb / mag
            aniso[b, k] = float(np.clip(mag / (Cxx + Cyy + 1e-9), 0.0, 1.0))
            mass[b, k] = Wsum

    def to(arr):
        return torch.from_numpy(arr).to(device=dev, dtype=dt)

    return to(sin2t), to(cos2t), to(aniso), to(mass)


def seg_orientation_loss(
    phm: torch.Tensor,
    preg: torch.Tensor,
    pseg: torch.Tensor,
    topk: int = 32,
    hm_thresh: float = 0.3,
    win_radius: float = 3.0,
    min_mass: float = 1.0,
    min_aniso: float = 0.05,
):
    """Self-supervised box-orientation loss driven by the segmentation.

    The soft-Dice consistency term constrains a box's *coverage* but barely its
    *angle*: a fat / mis-rotated box covers an elongated blob about as well as a
    tight aligned one, so on sources with no OBB GT the orientation drifts. This
    term injects the missing angle gradient: it reads the local vehicle axis
    straight from the (well-supervised, detached) segmentation and pulls the box's
    predicted sin2theta/cos2theta onto it.

    Only the regression head receives gradient (the teacher angle is detached).
    Round / ambiguous blobs are down-weighted by `aniso`, and empty windows are
    gated out by `min_mass`, so the term is quiet where orientation is undefined.

    Returns (loss, info) with info = dict of per-batch diagnostics for logging:
        active_frac: mean fraction of peaks that contributed,
        mean_aniso:  mean elongation of contributing windows.
    """
    sin2 = preg[:, 4]  # (B,H,W) keeps grad
    cos2 = preg[:, 5]
    topv, topi, iy, ix = _topk_peaks(phm, topk)

    def flat(t):
        return t.reshape(t.shape[0], -1)

    s_sel = torch.gather(flat(sin2), 1, topi)  # (B,K)
    c_sel = torch.gather(flat(cos2), 1, topi)
    r = torch.sqrt(s_sel * s_sel + c_sel * c_sel) + 1e-6
    # reg sin2θ/cos2θ is WORLD-frame (y up); the seg teacher is IMAGE-frame (row
    # down). Convert the box angle to image frame (sin2θ -> -sin2θ) so the two are
    # comparable — without this the teacher fights the box with a mirrored target.
    s_pred, c_pred = -s_sel / r, c_sel / r

    seg = torch.sigmoid(pseg).detach().squeeze(1)
    sin2t, cos2t, aniso, mass = _teacher_2theta(seg, iy, ix, win_radius)

    gate = ((topv > hm_thresh) & (mass > min_mass) & (aniso > min_aniso)).to(s_pred.dtype)
    weight = gate * aniso  # confidence-weighted
    align = 1.0 - (s_pred * sin2t + c_pred * cos2t)  # 1 - cos(2θ_pred - 2θ_teacher)
    denom = weight.sum() + 1e-6
    loss = (weight * align).sum() / denom
    info = {
        "active_frac": float(gate.mean().item()),
        "mean_aniso": float((aniso * gate).sum().item() / (gate.sum().item() + 1e-6)),
    }
    return loss, info


@torch.no_grad()
def decode_topk_teacher_angle(
    phm: torch.Tensor, pseg: torch.Tensor, topk: int = 32, win_radius: float = 3.0
):
    """For viz: teacher orientation at each top-k peak.

    Returns each (B,K): cx, cy (peak cell centre, stride-grid units),
    theta_teacher (rad), aniso, mass.
    """
    _topv, _topi, iy, ix = _topk_peaks(phm, topk)
    seg = torch.sigmoid(pseg).squeeze(1)
    sin2t, cos2t, aniso, mass = _teacher_2theta(seg, iy, ix, win_radius)
    theta = 0.5 * torch.atan2(sin2t, cos2t)
    cx = ix.float() + 0.5
    cy = iy.float() + 0.5
    return cx, cy, theta, aniso, mass


def _render_obb_boxes(
    cx: torch.Tensor,
    cy: torch.Tensor,
    Lg: torch.Tensor,
    Wg: torch.Tensor,
    theta: torch.Tensor,
    weight: torch.Tensor,
    out_size: int,
    sharp: float = 2.0,
):
    """Soft-render K oriented boxes per batch item, WITHOUT the union reduction.

    Returns (B, K, out_size, out_size) in [0, 1]: each box as the product of two
    logistic ramps across its edge pairs in the box-local frame, scaled by
    `weight`. `sharp` controls edge crispness. Callers that want a single union
    mask take `.amax(dim=1)` (see render_obb_soft); per-box callers (e.g. the
    local shadow consistency loss) keep the K axis.
    """
    B, K = cx.shape
    dev, dt = cx.device, cx.dtype
    # cell-centre grid coordinates in /stride units (i + 0.5)
    coord = torch.arange(out_size, device=dev, dtype=dt) + 0.5
    gy = coord.view(1, 1, out_size, 1)  # row -> y
    gx = coord.view(1, 1, 1, out_size)  # col -> x

    cxe = cx.view(B, K, 1, 1)
    cye = cy.view(B, K, 1, 1)
    # box theta is WORLD-frame (y up); we render onto the IMAGE grid (row down),
    # so negate the angle to match the seg/shadow we compare against.
    theta = -theta
    ct = torch.cos(theta).view(B, K, 1, 1)
    st = torch.sin(theta).view(B, K, 1, 1)
    half_L = (Lg * 0.5).view(B, K, 1, 1)
    half_W = (Wg * 0.5).view(B, K, 1, 1)

    dx = gx - cxe  # (B,K,1,W) -> broadcasts to (B,K,H,W)
    dy = gy - cye  # (B,K,H,1)
    local_x = dx * ct + dy * st  # along length
    local_y = -dx * st + dy * ct  # along width

    inside = torch.sigmoid(sharp * (half_L - local_x.abs())) * torch.sigmoid(
        sharp * (half_W - local_y.abs())
    )
    return inside * weight.view(B, K, 1, 1)


def render_obb_soft(
    cx: torch.Tensor,
    cy: torch.Tensor,
    Lg: torch.Tensor,
    Wg: torch.Tensor,
    theta: torch.Tensor,
    weight: torch.Tensor,
    out_size: int,
    sharp: float = 2.0,
):
    """Soft-render K oriented boxes per batch item into a single union mask.

    All box params are (B, K). Returns (B, out_size, out_size) in [0, 1], the
    soft union (max) of the K rendered rectangles, each scaled by `weight`.
    """
    return _render_obb_boxes(cx, cy, Lg, Wg, theta, weight, out_size=out_size, sharp=sharp).amax(
        dim=1
    )


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample soft-Dice loss, averaged over the batch. Inputs (B,H,W) in [0,1].

    With eps as a smoothing constant, an empty pred against an empty target gives
    loss ~ 0 (no spurious gradient on vehicle-free crops)."""
    num = 2.0 * (pred * target).sum(dim=(1, 2))
    den = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (num + eps) / (den + eps)
    return (1.0 - dice).mean()


def seg_obb_consistency_loss(
    phm: torch.Tensor,
    preg: torch.Tensor,
    pseg: torch.Tensor,
    cell: float,
    stride: int,
    topk: int = 32,
    hm_thresh: float = 0.3,
    sharp: float = 2.0,
):
    """Self-correcting seg<->OBB consistency loss.

    Args:
        phm:  (B,1,H,W) heatmap logits.
        preg: (B,REG_DIM,H,W) regression maps.
        pseg: (B,1,H,W) segmentation logits.
        cell: BEV cell size in metres.
        stride: output stride.
        topk: number of heatmap peaks to instantiate boxes from.
        hm_thresh: minimum heatmap probability for a peak to contribute.
        sharp: rectangle edge sharpness for the soft renderer.

    Returns (loss, soft_union) where soft_union is (B,H,W) for optional logging/viz.
    """
    _B, _, H, _W = phm.shape
    topv, topi, iy, ix = _topk_peaks(phm, topk)  # peak selection: no grad
    reg_sel = _gather_reg(preg, topi)  # (B,REG,K) — keeps grad
    cx, cy, Lg, Wg, theta = decode_obb_at_cells(reg_sel, iy, ix, cell, stride)
    weight = (topv > hm_thresh).to(reg_sel.dtype)  # detached gate

    union = render_obb_soft(cx, cy, Lg, Wg, theta, weight, out_size=H, sharp=sharp)
    seg_prob = torch.sigmoid(pseg).detach().squeeze(1)  # (B,H,W) — teacher
    return soft_dice_loss(union, seg_prob), union


def _seg_component_anchors(
    seg: torch.Tensor,
    max_components: int,
    min_area: float,
    max_area: float,
    seg_thresh: float,
    split_cell_m: float = 0.0,
    prior: VehiclePrior = DEFAULT_PRIOR,
):
    """Find temporary segmentation anchors; does not export pseudo OBB labels.

    With ``split_cell_m > 0`` (the metric size of one output-grid cell,
    ``cell * stride``), merged components longer than ~1.6 vehicle lengths are
    sliced along their principal axis into vehicle-length sub-anchors instead
    of being dropped by the max_area filter. Parking rows then contribute one
    anchor per plausible vehicle rather than nothing.
    """
    B, H, W = seg.shape
    dev = seg.device
    seg_np = seg.detach().to(torch.float32).cpu().numpy()
    iy = np.zeros((B, max_components), np.int64)
    ix = np.zeros((B, max_components), np.int64)
    comp_masks = np.zeros((B, max_components, H, W), np.float32)
    comp_area = np.zeros((B, max_components), np.float32)

    for b in range(B):
        labels, n_lab = ndi.label(seg_np[b] > seg_thresh)
        comps = []
        for lab in range(1, n_lab + 1):
            m = labels == lab
            if split_cell_m > 0.0:
                pieces = [sub for sub, _, _ in split_component_by_length(m, split_cell_m, prior)]
            else:
                pieces = [m]
            for piece in pieces:
                area = float(piece.sum())
                if area < min_area or area > max_area:
                    continue
                ys, xs = np.nonzero(piece)
                weights = seg_np[b][piece]
                wsum = float(weights.sum())
                if wsum <= 1e-6:
                    cy, cx = float(ys.mean()), float(xs.mean())
                else:
                    cy = float((ys * weights).sum() / wsum)
                    cx = float((xs * weights).sum() / wsum)
                comps.append((area, cy, cx, piece))
        comps.sort(key=lambda t: t[0], reverse=True)
        for k, (area, cy, cx, m) in enumerate(comps[:max_components]):
            iy[b, k] = int(np.clip(round(cy), 0, H - 1))
            ix[b, k] = int(np.clip(round(cx), 0, W - 1))
            comp_masks[b, k] = m.astype(np.float32) * seg_np[b]
            comp_area[b, k] = area

    return (
        torch.from_numpy(iy).to(dev),
        torch.from_numpy(ix).to(dev),
        torch.from_numpy(comp_masks).to(dev, dtype=seg.dtype),
        torch.from_numpy(comp_area).to(dev, dtype=seg.dtype),
    )


def _seg_to_grid(seg: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """Map a mask teacher to the model output grid before component analysis."""
    if seg.ndim != 3:
        raise ValueError(f"_seg_to_grid expects (B,H,W), got {tuple(seg.shape)}")
    if seg.shape[-2:] == (out_h, out_w):
        return seg
    return (
        F.adaptive_max_pool2d(seg.unsqueeze(1).to(torch.float32), output_size=(out_h, out_w))
        .squeeze(1)
        .to(seg.dtype)
    )


def _component_reliability(
    comp: torch.Tensor, area: torch.Tensor, min_aniso: float = 0.05, power: float = 1.0
):
    """Shape reliability for temporary mask anchors.

    The returned weight is detached and in [0,1]. It down-weights round or merged
    components whose principal-axis geometry is not a reliable vehicle instance.
    """
    if comp.numel() == 0:
        return area * 0.0, area * 0.0
    B, K, H, W = comp.shape
    dev, dt = comp.device, comp.dtype
    yy = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1) + 0.5
    xx = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W) + 0.5
    mass = comp.sum(dim=(-1, -2)).clamp_min(1e-6)
    mx = (comp * xx).sum(dim=(-1, -2)) / mass
    my = (comp * yy).sum(dim=(-1, -2)) / mass
    dx = xx - mx.view(B, K, 1, 1)
    dy = yy - my.view(B, K, 1, 1)
    cxx = (comp * dx * dx).sum(dim=(-1, -2)) / mass
    cyy = (comp * dy * dy).sum(dim=(-1, -2)) / mass
    cxy = (comp * dx * dy).sum(dim=(-1, -2)) / mass
    trace = cxx + cyy + 1e-6
    aniso = (torch.sqrt((cxx - cyy) ** 2 + (2.0 * cxy) ** 2 + 1e-9) / trace).clamp(0.0, 1.0)
    rel = ((aniso - float(min_aniso)) / max(1.0 - float(min_aniso), 1e-6)).clamp(0.0, 1.0)
    if power != 1.0:
        rel = rel.pow(float(power))
    rel = rel * (area > 0).to(dt)
    return rel.detach(), aniso.detach()


def seg_component_heatmap_loss(
    phm: torch.Tensor,
    seg_target: torch.Tensor | None,
    pseg: torch.Tensor | None,
    *,
    max_components: int = 24,
    seg_thresh: float = 0.4,
    min_area: float = 4.0,
    max_area: float = 260.0,
    radius: float = 1.5,
    peak: float = 0.75,
    min_aniso: float = 0.05,
    reliability_power: float = 1.0,
    teacher: str = "target",
    split_cell_m: float = 0.0,
    prior: VehiclePrior = DEFAULT_PRIOR,
):
    """Teach centre heatmap peaks from segmentation components.

    This is the missing bridge for segmentation-only domains: OBB consistency can
    train regression at chosen cells, but inference first needs heatmap peaks.
    Component centroids become temporary soft centre targets for the current
    batch only; no pseudo labels are exported or persisted.

    ``split_cell_m > 0`` enables length-prior splitting of merged components
    (see _seg_component_anchors), so parking rows produce one centre anchor per
    plausible vehicle instead of being filtered out.
    """
    if teacher == "target" and seg_target is not None:
        seg = seg_target.detach().squeeze(1)
    elif pseg is not None:
        seg = torch.sigmoid(pseg).detach().squeeze(1)
    elif seg_target is not None:
        seg = seg_target.detach().squeeze(1)
    else:
        return phm.sum() * 0.0, {"active_frac": 0.0, "components": 0.0}

    B, _, H, W = phm.shape
    K = int(max_components)
    if K <= 0:
        return phm.sum() * 0.0, {"active_frac": 0.0, "components": 0.0}

    seg = _seg_to_grid(seg, H, W)
    iy, ix, comp, area = _seg_component_anchors(
        seg, K, min_area, max_area, seg_thresh, split_cell_m=split_cell_m, prior=prior
    )
    active = (area > 0).to(phm.dtype)
    if not bool(active.any()):
        return phm.sum() * 0.0, {"active_frac": 0.0, "components": 0.0, "mean_reliability": 0.0}
    reliability, aniso = _component_reliability(
        comp, area, min_aniso=min_aniso, power=reliability_power
    )

    dev, dt = phm.device, phm.dtype
    yy = torch.arange(H, device=dev, dtype=dt).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev, dtype=dt).view(1, 1, 1, W)
    cy = iy.to(dt).view(B, K, 1, 1)
    cx = ix.to(dt).view(B, K, 1, 1)
    sigma = max(float(radius), 1e-3)
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
    target = (g * reliability.view(B, K, 1, 1)).amax(dim=1).clamp(0.0, 1.0) * float(peak)

    pred = torch.clamp(torch.sigmoid(phm[:, 0]), 1e-4, 1.0 - 1e-4)
    pos = target > 0.05
    neg = ~pos
    neg_w = torch.pow(1.0 - target, 4)
    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2) * target * pos.to(dt)
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2) * neg_w * neg.to(dt)
    norm = target[pos].sum().clamp(min=1.0)
    loss = -(pos_loss.sum() + neg_loss.sum()) / norm
    info = {
        "active_frac": float(active.mean().item()),
        "components": float(active.sum().item() / max(B, 1)),
        "mean_reliability": float(
            (reliability * active).sum().item() / (active.sum().item() + 1e-6)
        ),
        "mean_aniso": float((aniso * active).sum().item() / (active.sum().item() + 1e-6)),
    }
    return loss, info


def seg_mask_obb_explanation_loss(
    preg: torch.Tensor,
    seg_target: torch.Tensor | None,
    pseg: torch.Tensor | None,
    *,
    cell: float,
    stride: int,
    max_components: int = 24,
    seg_thresh: float = 0.4,
    min_area: float = 4.0,
    max_area: float = 260.0,
    sharp: float = 2.0,
    min_box_m: float = 1.0,
    max_box_m: float = 12.0,
    aspect_min: float = 1.2,
    aspect_max: float = 6.0,
    min_aniso: float = 0.05,
    reliability_power: float = 1.0,
    teacher: str = "target",
    split_cell_m: float = 0.0,
    prior: VehiclePrior = DEFAULT_PRIOR,
    size_prior_weight: float = 0.0,
):
    """Train-time OBB explanation of segmentation components.

    This is not pseudo-OBB supervision: the segment mask only chooses temporary
    component anchors and a footprint target in the current batch. The OBB comes
    from the regression head and is judged by how well its soft render explains
    the component footprint.

    Two prior-driven extensions for weakly labelled domains:

    * ``split_cell_m > 0`` slices merged components into vehicle-length anchors
      (see _seg_component_anchors) so parking rows supervise per-car boxes.
    * ``size_prior_weight > 0`` adds a log-normal penalty pulling the decoded
      (sorted) box dimensions toward ``prior.length_m x prior.width_m``. Blobby
      or partial masks constrain size poorly, so the prior carries most of the
      size signal; the squared log-deviation is capped at 3 sigma per axis.
    """
    if teacher == "target" and seg_target is not None:
        seg = seg_target.detach().squeeze(1)
    elif pseg is not None:
        seg = torch.sigmoid(pseg).detach().squeeze(1)
    elif seg_target is not None:
        seg = seg_target.detach().squeeze(1)
    else:
        return preg.sum() * 0.0, {"active_frac": 0.0, "mean_area": 0.0}

    _B, _, H, W = preg.shape
    K = int(max_components)
    if K <= 0:
        return preg.sum() * 0.0, {"active_frac": 0.0, "mean_area": 0.0}

    seg = _seg_to_grid(seg, H, W)
    iy, ix, comp, area = _seg_component_anchors(
        seg, K, min_area, max_area, seg_thresh, split_cell_m=split_cell_m, prior=prior
    )
    reliability, aniso = _component_reliability(
        comp, area, min_aniso=min_aniso, power=reliability_power
    )
    flat_idx = (iy * W + ix).long()
    reg_sel = _gather_reg(preg, flat_idx)
    cx, cy, Lg, Wg, theta = decode_obb_at_cells(reg_sel, iy, ix, cell, stride)

    min_box_g = float(min_box_m) / cell / stride
    max_box_g = float(max_box_m) / cell / stride
    Lg = Lg.clamp(min=min_box_g, max=max_box_g)
    Wg = Wg.clamp(min=min_box_g, max=max_box_g)
    active = (area > 0).to(preg.dtype)
    boxes = _render_obb_boxes(cx, cy, Lg, Wg, theta, active, out_size=H, sharp=sharp)

    inter = (boxes * comp).sum(dim=(-1, -2))
    box_mass = boxes.sum(dim=(-1, -2))
    comp_mass = comp.sum(dim=(-1, -2))
    dice_loss = 1.0 - (2.0 * inter + 1.0) / (box_mass + comp_mass + 1.0)

    ratio = torch.maximum(Lg, Wg) / torch.minimum(Lg, Wg).clamp_min(1e-3)
    aspect_loss = (F.relu(float(aspect_min) - ratio) + F.relu(ratio - float(aspect_max))) * 0.05
    loss_per = dice_loss + aspect_loss
    if size_prior_weight > 0.0:
        # Sorted dims: an L<->W swap is theta+90 deg in the 2theta encoding, so
        # the prior must not double-penalise the equivalent representation.
        grid_m = float(cell) * float(stride)
        long_m = (torch.maximum(Lg, Wg) * grid_m).clamp_min(1e-3)
        short_m = (torch.minimum(Lg, Wg) * grid_m).clamp_min(1e-3)
        dev_l = (torch.log(long_m) - math.log(prior.length_m)) / max(prior.length_sigma, 1e-3)
        dev_w = (torch.log(short_m) - math.log(prior.width_m)) / max(prior.width_sigma, 1e-3)
        size_loss = 0.5 * (dev_l.pow(2).clamp(max=9.0) + dev_w.pow(2).clamp(max=9.0))
        loss_per = loss_per + float(size_prior_weight) * size_loss
    weight = active * reliability.to(preg.dtype)
    denom = weight.sum().clamp(min=1.0)
    loss = (loss_per * weight).sum() / denom
    info = {
        "active_frac": float(active.mean().item()),
        "mean_area": float((area * active).sum().item() / (active.sum().item() + 1e-6)),
        "mean_reliability": float(
            (reliability * active).sum().item() / (active.sum().item() + 1e-6)
        ),
        "mean_aniso": float((aniso * active).sum().item() / (active.sum().item() + 1e-6)),
    }
    return loss, info


def _shadow_to_stride(shadow: torch.Tensor, out_size: int) -> torch.Tensor:
    """Resample the ground-shadow channel to the (out_size,out_size) stride grid.

    Max-pooling is used (not bilinear) so the thin vehicle-footprint shadow is
    preserved when it is sub-stride wide. Returns (B, out_size, out_size) in [0,1].
    """
    if shadow.ndim == 3:
        shadow = shadow.unsqueeze(1)
    s = shadow.to(torch.float32)
    if s.shape[-1] != out_size or s.shape[-2] != out_size:
        s = F.adaptive_max_pool2d(s, output_size=(out_size, out_size))
    return s.squeeze(1).clamp(0.0, 1.0)


def shadow_box_consistency_loss(
    phm: torch.Tensor,
    preg: torch.Tensor,
    shadow: torch.Tensor,
    pseg: torch.Tensor | None = None,
    *,
    cell: float,
    stride: int,
    topk: int = 32,
    hm_thresh: float = 0.3,
    sharp: float = 2.0,
    win_radius: float = 8.0,
    min_mass: float = 1.0,
    seg_gate: bool = True,
    seg_thresh: float = 0.4,
    precision_only: bool = False,
):
    """Tie decoded OBB *footprints* to the ground-occlusion shadow channel.

    Rationale: a vehicle blocks the LiDAR from reaching the ground beneath it, so
    the ground DTM has a hole at the vehicle footprint (`ground_shadow`).

    Why the original *global soft-Dice* version was wrong (measured): on real data
    `ground_shadow` is NOT vehicle-specific — buildings, trees and slopes cast
    ground shadow too (in-box mean ~0.5 vs out-of-box ~0.28 on DALES). A global
    Dice between the box-union and the whole channel therefore asks the boxes to
    cover *every* shadow, which is unsatisfiable, so the loss sat at ~0.9 and
    never learned.

    This redesign fixes both failure modes:
      * seg-gated   — the shadow target is masked by the predicted vehicle
        segmentation (detached), removing building/tree shadow that no vehicle
        is segmented over.
      * local       — each top-k peak only sees the shadow inside a circular
        window of `win_radius` stride-cells around it, so distant shadow cannot
        leak in and a per-car footprint is the unit of comparison.
      * per-box dice / precision — the box is compared to its own local shadow
        blob. The box mass sits in the denominator, so a box that spills beyond
        the shadow is penalised (precision) and a box that under-covers it is
        penalised (recall). `precision_only=True` drops the recall side and only
        penalises box mass that has no shadow under it.

    Args:
        phm:    (B,1,H,W) heatmap logits.
        preg:   (B,REG_DIM,H,W) regression maps.
        shadow: (B,1,Hf,Wf) or (B,Hf,Wf) ground_shadow BEV channel in [0,1].
        pseg:   (B,1,H,W) segmentation logits used to gate the shadow target
                (required when seg_gate=True).
        win_radius:    local window radius in stride-grid cells.
        min_mass:      min (gated) shadow mass in the window for a peak to count.
        seg_gate:      multiply the shadow target by sigmoid(pseg).detach().
        precision_only: only penalise box pixels lacking shadow (no recall term).
    Returns (loss, union) where union is the soft box mask (B,H,W) for viz.
    """
    B, _, H, W = phm.shape
    topv, topi, iy, ix = _topk_peaks(phm, topk)
    K = topv.shape[1]
    reg_sel = _gather_reg(preg, topi)  # keeps grad
    cx, cy, Lg, Wg, theta = decode_obb_at_cells(reg_sel, iy, ix, cell, stride)
    weight = (topv > hm_thresh).to(reg_sel.dtype)  # detached gate

    boxes = _render_obb_boxes(cx, cy, Lg, Wg, theta, weight, out_size=H, sharp=sharp)
    target = _shadow_to_stride(shadow, H).detach()  # (B,H,W) in [0,1]
    if seg_gate:
        if pseg is None:
            raise ValueError("shadow_box_consistency_loss: seg_gate=True needs pseg")
        seg = torch.sigmoid(pseg).detach().squeeze(1)
        seg = (seg > seg_thresh).to(target.dtype) * seg  # soft, thresholded
        target = target * seg

    dev, dt = phm.device, phm.dtype
    coord = torch.arange(H, device=dev, dtype=dt) + 0.5
    gy = coord.view(1, 1, H, 1)
    gx = coord.view(1, 1, 1, W)
    cxe = (ix.float() + 0.5).view(B, K, 1, 1)
    cye = (iy.float() + 0.5).view(B, K, 1, 1)
    r2 = (gx - cxe) ** 2 + (gy - cye) ** 2
    win = (r2 <= float(win_radius) ** 2).to(dt)  # (B,K,H,W) hard window
    t_local = (target.unsqueeze(1) * win).detach()  # per-peak shadow target
    mass = t_local.sum(dim=(-1, -2))  # (B,K)

    box_win = boxes * win  # box mass in the window
    inter = (box_win * t_local).sum(dim=(-1, -2))
    if precision_only:
        # fraction of box mass that has no shadow under it
        box_mass = box_win.sum(dim=(-1, -2)).clamp(min=1e-6)
        per_peak = 1.0 - inter / box_mass
    else:
        denom = box_win.sum(dim=(-1, -2)) + mass + 1e-6
        per_peak = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)  # local soft-Dice

    gate = ((topv > hm_thresh) & (mass > min_mass)).to(dt)
    loss = (per_peak * gate).sum() / (gate.sum() + 1e-6)
    union = boxes.amax(dim=1)
    return loss, union


# ── tiny self-test ───────────────────────────────────────────────────────────────
def _topk_reg_decode(phm: torch.Tensor, preg: torch.Tensor, cell: float, stride: int, topk: int):
    topv, topi, iy, ix = _topk_peaks(phm, topk)
    reg_sel = _gather_reg(preg, topi)
    cx, cy, Lg, Wg, theta = decode_obb_at_cells(reg_sel, iy, ix, cell, stride)
    return topv, iy, ix, cx, cy, Lg, Wg, theta


def seg_moment_obb_loss(
    phm: torch.Tensor,
    preg: torch.Tensor,
    pseg: torch.Tensor,
    cell: float,
    stride: int,
    topk: int = 32,
    hm_thresh: float = 0.3,
    win_radius: float = 8.0,
    min_mass: float = 1.0,
    min_aniso: float = 0.03,
    min_box_m: float = 1.0,
    max_box_m: float = 12.0,
):
    """Match decoded OBB centre/shape moments to local segmentation moments."""
    B, _, H, W = phm.shape
    topv, iy, ix, cx, cy, Lg, Wg, theta = _topk_reg_decode(phm, preg, cell, stride, topk)
    min_box_g = min_box_m / cell / stride
    max_box_g = max_box_m / cell / stride
    Lg = Lg.clamp(min=min_box_g, max=max_box_g)
    Wg = Wg.clamp(min=min_box_g, max=max_box_g)
    seg = torch.sigmoid(pseg).detach().squeeze(1)
    dev, dt = seg.device, seg.dtype
    coord = torch.arange(max(H, W), device=dev, dtype=dt) + 0.5
    gy = coord[:H].view(1, 1, H, 1)
    gx = coord[:W].view(1, 1, 1, W)
    cxe = (ix.float() + 0.5).view(B, -1, 1, 1)
    cye = (iy.float() + 0.5).view(B, -1, 1, 1)
    dx0 = gx - cxe
    dy0 = gy - cye
    r2 = dx0 * dx0 + dy0 * dy0
    sigma = max(float(win_radius) * 0.5, 1e-3)
    win = torch.exp(-r2 / (2.0 * sigma * sigma)) * (r2 <= win_radius * win_radius).to(dt)
    w = (win * seg.unsqueeze(1)).detach()
    mass = w.sum(dim=(-1, -2)) + 1e-6
    tx = (w * gx).sum(dim=(-1, -2)) / mass
    ty = (w * gy).sum(dim=(-1, -2)) / mass
    ddx = gx - tx.view(B, -1, 1, 1)
    ddy = gy - ty.view(B, -1, 1, 1)
    Cxx = (w * ddx * ddx).sum(dim=(-1, -2)) / mass
    Cyy = (w * ddy * ddy).sum(dim=(-1, -2)) / mass
    Cxy = (w * ddx * ddy).sum(dim=(-1, -2)) / mass
    trace = Cxx + Cyy + 1e-6
    mag = torch.sqrt((Cxx - Cyy) ** 2 + (2.0 * Cxy) ** 2 + 1e-9)
    aniso = (mag / trace).clamp(0.0, 1.0)
    ct, st = torch.cos(-theta), torch.sin(-theta)  # WORLD->IMAGE frame (row down)
    vL = (Lg.clamp(min=1e-3) ** 2) / 12.0
    vW = (Wg.clamp(min=1e-3) ** 2) / 12.0
    Pxx = vL * ct * ct + vW * st * st
    Pyy = vL * st * st + vW * ct * ct
    Pxy = (vL - vW) * st * ct
    gate = ((topv > hm_thresh) & (mass > min_mass) & (aniso > min_aniso)).to(dt)
    weight = gate * aniso
    center_loss = ((cx - tx) ** 2 + (cy - ty) ** 2) / max(float(win_radius) ** 2, 1.0)
    cov_loss = ((Pxx - Cxx) ** 2 + (Pyy - Cyy) ** 2 + 2.0 * (Pxy - Cxy) ** 2) / (trace * trace)
    area_loss = (
        torch.log((Lg * Wg).clamp(min=1e-3)) - torch.log((mass / 0.7).clamp(min=1e-3))
    ).abs() * 0.05
    denom = weight.sum() + 1e-6
    loss = (weight * (center_loss + cov_loss + area_loss)).sum() / denom
    info = {
        "active_frac": float(gate.mean().item()),
        "mean_aniso": float((aniso * gate).sum().item() / (gate.sum().item() + 1e-6)),
        "mean_mass": float((mass * gate).sum().item() / (gate.sum().item() + 1e-6)),
    }
    return loss, info


def gaussian_obb_reg_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cell: float,
    stride: int,
    eps: float = 1e-6,
    min_box_m: float = 1.0,
    max_box_m: float = 12.0,
    class_ids: torch.Tensor | None = None,
    doubled_yaw_ids=None,
) -> torch.Tensor:
    """Gaussian covariance loss for encoded OBB maps at supervised cells."""
    m = mask[:, 0] if mask.ndim == 4 else mask
    active = m > 0
    if not bool(active.any()):
        return pred.sum() * 0.0
    p = pred.permute(0, 2, 3, 1)[active]
    t = target.permute(0, 2, 3, 1)[active]
    wt = m[active].to(pred.dtype)
    center = ((p[:, 0:2] - t[:, 0:2]) ** 2).sum(dim=1)
    log_min, log_max = math.log(min_box_m), math.log(max_box_m)
    pL = torch.exp(p[:, 2].clamp(min=log_min, max=log_max)).clamp(min=eps)
    pW = torch.exp(p[:, 3].clamp(min=log_min, max=log_max)).clamp(min=eps)
    tL = torch.exp(t[:, 2].clamp(min=log_min, max=log_max)).clamp(min=eps)
    tW = torch.exp(t[:, 3].clamp(min=log_min, max=log_max)).clamp(min=eps)
    scale = torch.log(pL / tL).abs() + torch.log(pW / tW).abs()
    if class_ids is None or doubled_yaw_ids is None:
        yaw_factor = torch.full_like(p[:, 4], 2.0)
    else:
        active_classes = class_ids[active]
        doubled = torch.zeros_like(active_classes, dtype=torch.bool)
        for class_id in doubled_yaw_ids:
            doubled |= active_classes == int(class_id)
        yaw_factor = torch.where(doubled, 2.0, 1.0).to(dtype=p.dtype)
    ptheta = torch.atan2(p[:, 4], p[:, 5]) / yaw_factor
    ttheta = torch.atan2(t[:, 4], t[:, 5]) / yaw_factor

    def cov_terms(L, W, th):
        Lg = L / cell / stride
        Wg = W / cell / stride
        vL = Lg * Lg / 12.0
        vW = Wg * Wg / 12.0
        c, s = torch.cos(th), torch.sin(th)
        return (vL * c * c + vW * s * s, vL * s * s + vW * c * c, (vL - vW) * s * c)

    pxx, pyy, pxy = cov_terms(pL, pW, ptheta)
    txx, tyy, txy = cov_terms(tL, tW, ttheta)
    tr = (txx + tyy).clamp(min=eps)
    cov = ((pxx - txx) ** 2 + (pyy - tyy) ** 2 + 2.0 * (pxy - txy) ** 2) / (tr * tr)
    loss = center + 0.25 * scale + cov
    return (loss * wt).sum() / wt.sum().clamp(min=1.0)


def _bev_channel(
    bev_crop: torch.Tensor, channel_names, name: str, out_hw: tuple[int, int]
) -> torch.Tensor | None:
    names = tuple(str(x) for x in (channel_names or ()))
    if name not in names:
        return None
    idx = names.index(name)
    if idx >= bev_crop.shape[1]:
        return None
    ch = bev_crop[:, idx : idx + 1]
    scale = float(_CHANNEL_SCALE_BY_NAME.get(name, 1.0))
    ch = ch * scale
    if ch.shape[-2:] != out_hw:
        ch = F.interpolate(ch, size=out_hw, mode="bilinear", align_corners=False)
    return ch.squeeze(1)


def _first_bev_channel(
    bev_crop: torch.Tensor, channel_names, names, out_hw: tuple[int, int]
) -> torch.Tensor | None:
    for name in names:
        ch = _bev_channel(bev_crop, channel_names, name, out_hw)
        if ch is not None:
            return ch
    return None


def seg_z_height_consistency_loss(
    phm: torch.Tensor,
    preg: torch.Tensor,
    pseg: torch.Tensor,
    bev_crop: torch.Tensor,
    channel_names,
    cell: float,
    stride: int,
    topk: int = 32,
    hm_thresh: float = 0.3,
    win_radius: float = 8.0,
    min_mass: float = 1.0,
    min_height: float = 0.4,
    max_height: float = 4.5,
    detach_teacher: bool = True,
):
    """Use segmentation-masked BEV HAG statistics to supervise z/height.

    reg[6] and reg[7] are direct metres in the existing target encoding. This
    loss leaves the segmentation teacher detached by default and only updates
    the gathered regression vectors at heatmap-selected cells.

    KNOWN LIMITATION: the metre reconstruction multiplies *normalized* BEV
    channels by the fixed physical scales in _CHANNEL_SCALE_BY_NAME, which is
    only exact under norm-mode "physical". Under "train_stats" or
    "legacy_tile_percentile" the recovered heights are affinely distorted, so
    keep z_height_weight=0 (current presets) or switch to physical
    normalization before enabling this teacher.
    """
    B, _, H, W = phm.shape
    topv, topi, iy, ix = _topk_peaks(phm, topk)

    # generic schema: veh_hmax / all_hmax / hag_*
    # v2_020 schema:  veh_h_p95 / all_h_p95 (percentile vehicle-band channels)
    top = _first_bev_channel(
        bev_crop,
        channel_names,
        (
            "veh_hmax",
            "all_hmax",
            "hag_p95",
            "hag_max_low",
            "hag_max",
            "hag_mean_low",
            "veh_h_p95",
            "all_h_p95",
        ),
        (H, W),
    )
    if top is None:
        _warn_once("zh_no_top", "[z-height teacher] no HAG top channel found; loss disabled.")
        zero = preg.sum() * 0.0
        return zero, {
            "active_frac": 0.0,
            "mean_height_teacher": 0.0,
            "mean_z_teacher": 0.0,
            "mean_mass": 0.0,
        }

    # v2_020: veh_h_p05 is the bottom height; veh_h_span is the vertical extent
    bottom = _first_bev_channel(
        bev_crop, channel_names, ("hag_p05", "hag_min_low", "veh_h_p05"), (H, W)
    )
    span = _first_bev_channel(
        bev_crop, channel_names, ("vertical_span_all", "hag_range_low", "veh_h_span"), (H, W)
    )
    degraded = False
    if bottom is None:
        degraded = True
        if span is None:
            bottom = torch.full_like(top, 0.25)
            _warn_once(
                "zh_bottom_fallback",
                "[z-height teacher] missing hag_min_low/hag_p05 and span channels; "
                "using bottom=0.25m fallback.",
            )
        else:
            bottom = top - span.clamp(min=0.0)
            _warn_once(
                "zh_span_fallback",
                "[z-height teacher] missing hag_min_low/hag_p05; "
                "using top - vertical span/range fallback.",
            )

    seg = torch.sigmoid(pseg).squeeze(1)
    if detach_teacher:
        seg = seg.detach()
        top = top.detach()
        bottom = bottom.detach()

    dev, dt = phm.device, phm.dtype
    coord = torch.arange(max(H, W), device=dev, dtype=dt) + 0.5
    gy = coord[:H].view(1, 1, H, 1)
    gx = coord[:W].view(1, 1, 1, W)
    cxe = (ix.float() + 0.5).view(B, -1, 1, 1)
    cye = (iy.float() + 0.5).view(B, -1, 1, 1)
    dx = gx - cxe
    dy = gy - cye
    r2 = dx * dx + dy * dy
    sigma = max(float(win_radius) * 0.5, 1e-3)
    win = torch.exp(-r2 / (2.0 * sigma * sigma)) * (r2 <= win_radius * win_radius).to(dt)
    weight_map = win * seg.unsqueeze(1)
    mass = weight_map.sum(dim=(-1, -2)) + 1e-6

    top_t = (weight_map * top.unsqueeze(1)).sum(dim=(-1, -2)) / mass
    bot_t = (weight_map * bottom.unsqueeze(1)).sum(dim=(-1, -2)) / mass
    height_t = (top_t - bot_t).clamp(min=0.0)
    z_t = 0.5 * (top_t + bot_t)

    reg_sel = _gather_reg(preg, topi)
    z_pred, h_pred = reg_sel[:, 6], reg_sel[:, 7]

    valid_height = (height_t >= min_height) & (height_t <= max_height)
    gate = ((topv > hm_thresh) & (mass > min_mass) & valid_height).to(dt)
    smooth_mass = (mass / (mass + float(min_mass))).clamp(0.0, 1.0)
    weights = gate * smooth_mass * topv.to(dt).clamp(0.0, 1.0)
    denom = weights.sum()
    if float(denom.detach().item()) <= 0.0:
        zero = preg.sum() * 0.0
        return zero, {
            "active_frac": 0.0,
            "mean_height_teacher": 0.0,
            "mean_z_teacher": 0.0,
            "mean_mass": 0.0,
            "degraded_teacher": float(degraded),
        }

    z_loss = F.smooth_l1_loss(z_pred, z_t, reduction="none", beta=0.2)
    h_loss = F.smooth_l1_loss(h_pred, height_t, reduction="none", beta=0.2)
    loss = (weights * (z_loss + h_loss)).sum() / denom.clamp(min=1e-6)
    active = gate.sum().clamp(min=1.0)
    info = {
        "active_frac": float(gate.mean().item()),
        "mean_height_teacher": float((height_t * gate).sum().item() / active.item()),
        "mean_z_teacher": float((z_t * gate).sum().item() / active.item()),
        "mean_mass": float((mass * gate).sum().item() / active.item()),
        "degraded_teacher": float(degraded),
    }
    return loss, info
