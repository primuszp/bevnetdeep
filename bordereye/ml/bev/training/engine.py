"""Reusable training-engine state and numerical safety helpers."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossOut:
    """Named output of one forward/loss evaluation."""

    loss: object
    l_hm: float
    l_reg: float
    l_seg: float
    l_cons: float
    l_orient: float
    l_moment: float
    l_greg: float
    l_base: float
    l_zheight: float
    l_shadow: float
    l_seg_explain: float
    l_seg_center: float
    zheight_info: dict
    orient_info: dict
    seg_explain_info: dict
    seg_center_info: dict
    yaw_info: dict
    pred_reg: object
    pred_seg: object


def snapshot_bn_state(model):
    """Capture BatchNorm running stats before a potentially unstable batch."""
    state = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            state.append((
                module,
                module.running_mean.detach().clone() if module.running_mean is not None else None,
                module.running_var.detach().clone() if module.running_var is not None else None,
                module.num_batches_tracked.detach().clone()
                if getattr(module, "num_batches_tracked", None) is not None else None,
            ))
    return state


def restore_bn_state(state) -> None:
    """Restore a snapshot produced by :func:`snapshot_bn_state`."""
    for module, running_mean, running_var, num_batches_tracked in state:
        if running_mean is not None:
            module.running_mean.copy_(running_mean)
        if running_var is not None:
            module.running_var.copy_(running_var)
        if num_batches_tracked is not None:
            module.num_batches_tracked.copy_(num_batches_tracked)


def nonfinite_grad_summary(model, limit: int = 8) -> str:
    """Return a compact summary of parameters whose gradients are not finite."""
    bad = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        finite = torch.isfinite(grad)
        if bool(finite.all().item()):
            continue
        safe = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        bad.append(
            f"{name}: nonfinite={int((~finite).sum().item())}/{grad.numel()} "
            f"min={float(safe.min().cpu()):.4g} max={float(safe.max().cpu()):.4g}")
        if len(bad) >= limit:
            break
    return " | ".join(bad) if bad else "no per-parameter nonfinite gradients found"
