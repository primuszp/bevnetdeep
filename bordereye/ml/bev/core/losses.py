"""Training losses for the stable BEV three-head model contract."""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Per-sample reductions keep the batch axis and collapse (C, H, W). Every
# ``*_per_sample`` loss below returns a (B,) vector holding exactly the value
# the scalar loss would produce for that sample on its own, which lets the
# batch-weighted wrappers in ``core.data`` avoid a Python loop over the batch.
_SAMPLE_DIMS = (1, 2, 3)


def _focal_terms(pred: torch.Tensor, gt: torch.Tensor,
                 ignore: torch.Tensor | None,
                 pos_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (per-element focal contribution, per-element normaliser)."""
    pred_sig = torch.clamp(torch.sigmoid(pred), 1e-4, 1 - 1e-4)
    pos = (gt >= pos_threshold).float()
    neg = (gt < pos_threshold).float()
    if ignore is not None:
        valid = (1 - ignore).clamp(0, 1)
        pos, neg = pos * valid, neg * valid
    neg_w = torch.pow(1 - gt, 4)
    pos_l = torch.log(pred_sig) * torch.pow(1 - pred_sig, 2) * gt * pos
    neg_l = torch.log(1 - pred_sig) * torch.pow(pred_sig, 2) * neg_w * neg
    return pos_l + neg_l, pos * gt


def focal_loss(pred: torch.Tensor, gt: torch.Tensor,
               ignore: torch.Tensor | None = None,
               pos_threshold: float = 0.5) -> torch.Tensor:
    """Soft-label CenterNet focal loss for the centre heatmap."""
    terms, norm = _focal_terms(pred, gt, ignore, pos_threshold)
    return -terms.sum() / norm.sum().clamp(min=1.0)


def focal_loss_per_sample(pred: torch.Tensor, gt: torch.Tensor,
                          ignore: torch.Tensor | None = None,
                          pos_threshold: float = 0.5,
                          class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """(B,) vector of :func:`focal_loss` values, one per batch element."""
    terms, norm = _focal_terms(pred, gt, ignore, pos_threshold)
    if class_weights is not None:
        weights = class_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
        if weights.shape[1] != pred.shape[1]:
            raise ValueError(
                f"class_weights has {weights.shape[1]} values for {pred.shape[1]} classes")
        terms, norm = terms * weights, norm * weights
    return -terms.sum(dim=_SAMPLE_DIMS) / norm.sum(dim=_SAMPLE_DIMS).clamp(min=1.0)


def _reg_l1_terms(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor,
                  channel_weights: torch.Tensor | None,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the (per-element weighted |error|, per-element normaliser)."""
    m = mask.expand_as(pred)
    diff = torch.abs(pred - target) * m
    if channel_weights is not None:
        w = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
        return diff * w, m * w
    return diff, m


def reg_l1(pred: torch.Tensor, target: torch.Tensor,
           mask: torch.Tensor,
           channel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Masked L1 loss for OBB regression channels."""
    diff, norm = _reg_l1_terms(pred, target, mask, channel_weights)
    return diff.sum() / norm.sum().clamp(min=1.0)


def reg_l1_per_sample(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor,
                      channel_weights: torch.Tensor | None = None) -> torch.Tensor:
    """(B,) vector of :func:`reg_l1` values, one per batch element."""
    diff, norm = _reg_l1_terms(pred, target, mask, channel_weights)
    return diff.sum(dim=_SAMPLE_DIMS) / norm.sum(dim=_SAMPLE_DIMS).clamp(min=1.0)


def angle_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Direct doubled-angle loss on supervised OBB cells."""
    m = (mask[:, 0] if mask.ndim == 4 else mask) > 0
    if not bool(m.any()):
        return pred.sum() * 0.0
    sp, cp = pred[:, 4][m], pred[:, 5][m]
    st, ct = target[:, 4][m], target[:, 5][m]
    n = torch.sqrt((sp * sp + cp * cp).clamp_min(1e-12))
    cos_diff = (sp / n) * st + (cp / n) * ct
    return (1.0 - cos_diff).mean()


def class_aware_yaw_geodesic_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    doubled_yaw_ids=(),
    no_orientation_ids=(),
    macro_average: bool = False,
) -> torch.Tensor:
    """Direct angular-distance loss for mixed directed/pi-symmetric classes.

    Unlike cosine loss, the wrapped geodesic error retains a useful gradient
    for large yaw mistakes.  The encoded angle is divided by two only for
    classes represented as sin(2 theta), cos(2 theta).
    """
    active = (mask[:, 0] if mask.ndim == 4 else mask) > 0
    for class_id in no_orientation_ids:
        active &= class_ids != int(class_id)
    if not bool(active.any()):
        return pred.sum() * 0.0
    sp, cp = pred[:, 4][active], pred[:, 5][active]
    st, ct = target[:, 4][active], target[:, 5][active]
    # atan2(cross, dot) is the signed, wrapped difference of the encoded
    # unit vectors and remains scale-invariant for non-unit predictions.
    encoded = torch.atan2(sp * ct - cp * st, sp * st + cp * ct).abs()
    factors = torch.ones_like(encoded)
    active_classes = class_ids[active]
    for class_id in doubled_yaw_ids:
        factors = torch.where(active_classes == int(class_id), 2.0, factors)
    error = encoded / factors
    per_object = F.smooth_l1_loss(
        error, torch.zeros_like(error), beta=0.05, reduction="none")
    if macro_average:
        # WiSAR's mAOE is a macro average over oriented classes.  A plain
        # object mean lets frequent Box/Chair instances drown out rare classes
        # such as Canopy and Porta_potty, so mirror the evaluation contract.
        return torch.stack([
            per_object[active_classes == class_id].mean()
            for class_id in torch.unique(active_classes)
        ]).mean()
    return per_object.mean()


def vertical_overlap_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    class_ids: torch.Tensor | None = None, *, macro_average: bool = False,
) -> torch.Tensor:
    """Differentiable 1-D box-overlap loss for z centre and height."""
    active = (mask[:, 0] if mask.ndim == 4 else mask) > 0
    if not bool(active.any()):
        return pred.sum() * 0.0
    pz, tz = pred[:, 6][active], target[:, 6][active]
    ph = pred[:, 7][active].clamp(min=0.05)
    th = target[:, 7][active].clamp(min=0.05)
    intersection = (
        torch.minimum(pz + 0.5 * ph, tz + 0.5 * th)
        - torch.maximum(pz - 0.5 * ph, tz - 0.5 * th)
    ).clamp(min=0.0)
    iou = intersection / (ph + th - intersection).clamp(min=1e-6)
    # Smooth terms preserve a recovery gradient if intervals do not overlap.
    recovery = F.smooth_l1_loss(pz, tz, reduction="none", beta=0.1)
    recovery = recovery + 0.5 * F.smooth_l1_loss(ph, th, reduction="none", beta=0.1)
    per_object = (1.0 - iou) + recovery
    if macro_average:
        if class_ids is None:
            raise ValueError("class_ids are required for macro vertical overlap")
        active_classes = class_ids[active]
        return torch.stack([
            per_object[active_classes == class_id].mean()
            for class_id in torch.unique(active_classes)
        ]).mean()
    return per_object.mean()


def segmentation_valid_weight(weight: torch.Tensor,
                              ignore: torch.Tensor | None = None) -> torch.Tensor:
    """Combine confidence and ignore maps without changing target semantics."""
    w = weight.to(dtype=torch.float32).clamp(0.0, 1.0)
    if ignore is not None:
        w = w * (1.0 - ignore.to(dtype=w.dtype)).clamp(0.0, 1.0)
    return w


def weighted_seg_bce_per_sample(pred: torch.Tensor, target: torch.Tensor,
                                weight: torch.Tensor,
                                ignore: torch.Tensor | None = None) -> torch.Tensor:
    """Per-sample binary cross entropy over trusted pixels."""
    w = segmentation_valid_weight(weight, ignore).to(dtype=pred.dtype)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    return ((loss * w).sum(dim=_SAMPLE_DIMS)
            / w.sum(dim=_SAMPLE_DIMS).clamp(min=1.0))


def weighted_seg_dice_per_sample(pred: torch.Tensor, target: torch.Tensor,
                                 weight: torch.Tensor,
                                 ignore: torch.Tensor | None = None,
                                 eps: float = 1.0) -> torch.Tensor:
    """(B,) vector of Dice losses, one per batch element.

    Weight is applied exactly once to each sufficient statistic. Multiplying
    both probability and target first would accidentally square soft weights.
    """
    prob = torch.sigmoid(pred)
    w = segmentation_valid_weight(weight, ignore).to(dtype=pred.dtype)
    inter = (w * prob * target).sum(dim=_SAMPLE_DIMS)
    denom = (w * prob).sum(dim=_SAMPLE_DIMS) + (w * target).sum(dim=_SAMPLE_DIMS)
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def weighted_seg_dice(pred: torch.Tensor, target: torch.Tensor,
                      weight: torch.Tensor, ignore: torch.Tensor | None = None,
                      eps: float = 1.0) -> torch.Tensor:
    """Confidence-weighted Dice loss, averaged over the batch."""
    return weighted_seg_dice_per_sample(pred, target, weight, ignore, eps).mean()


def weighted_seg_tversky_per_sample(pred: torch.Tensor, target: torch.Tensor,
                                    weight: torch.Tensor,
                                    ignore: torch.Tensor | None = None,
                                    alpha: float = 0.7, beta: float = 0.3,
                                    eps: float = 1.0) -> torch.Tensor:
    """(B,) vector of Tversky losses, one per batch element."""
    prob = torch.sigmoid(pred)
    w = segmentation_valid_weight(weight, ignore).to(dtype=pred.dtype)
    tp = (w * prob * target).sum(dim=_SAMPLE_DIMS)
    fp = (w * prob * (1.0 - target)).sum(dim=_SAMPLE_DIMS)
    fn = (w * (1.0 - prob) * target).sum(dim=_SAMPLE_DIMS)
    score = (tp + eps) / (tp + float(alpha) * fp + float(beta) * fn + eps)
    return 1.0 - score


def weighted_seg_tversky(pred: torch.Tensor, target: torch.Tensor,
                         weight: torch.Tensor, ignore: torch.Tensor | None = None,
                         alpha: float = 0.7, beta: float = 0.3,
                         eps: float = 1.0) -> torch.Tensor:
    """Confidence-weighted Tversky loss with explicit FP/FN tradeoff."""
    return weighted_seg_tversky_per_sample(
        pred, target, weight, ignore, alpha, beta, eps).mean()
