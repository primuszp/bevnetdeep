"""Model factory for BEV vehicle detector architectures."""
from __future__ import annotations

import torch.nn as nn

from .nn_backbones import BevNet, BevNetDeep
from .nn_contract import MODEL_ARCHES


def build_bev_model(
    arch: str,
    n_ch: int,
    *,
    cue_idx=None,
    cbam_max: bool = False,
    reg_gate: bool = True,
    base: int | None = None,
    dropout: float = 0.1,
    num_classes: int = 1,
) -> nn.Module:
    """Build a BEV detector by canonical architecture name."""
    arch = str(arch or "v1").lower()
    n_ch = int(n_ch)
    if n_ch <= 0:
        raise ValueError(f"n_ch must be positive, got {n_ch}")
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    cue_idx = tuple(int(i) for i in (cue_idx or ()))
    if len(set(cue_idx)) != len(cue_idx) or any(i < 0 or i >= n_ch for i in cue_idx):
        raise ValueError(f"cue_idx must contain unique indices in [0,{n_ch}), got {cue_idx}")
    if arch == "deep":
        deep_base = 64 if base is None else int(base)
        if deep_base <= 0 or deep_base % 8:
            raise ValueError(f"deep base must be positive and divisible by 8, got {deep_base}")
        return BevNetDeep(
            n_ch=n_ch,
            base=deep_base,
            dropout=dropout,
            cue_idx=cue_idx,
            cbam_max=cbam_max,
            reg_gate=reg_gate,
            num_classes=num_classes,
        )
    if arch in ("v1", "v1_hr", "v1_hr_classreg", "v1_hr_boxdetail"):
        compact_base = 48 if base is None else int(base)
        if compact_base <= 0:
            raise ValueError(f"v1 base must be positive, got {compact_base}")
        return BevNet(
            n_ch=n_ch,
            base=compact_base,
            dropout=dropout,
            cue_idx=cue_idx,
            cbam_max=cbam_max,
            reg_gate=reg_gate,
            num_classes=num_classes,
            high_resolution=(arch in ("v1_hr", "v1_hr_classreg", "v1_hr_boxdetail")),
            class_regression=(arch in ("v1_hr_classreg", "v1_hr_boxdetail")),
            box_detail=(arch == "v1_hr_boxdetail"),
        )
    raise ValueError(f"unknown BEV model architecture {arch!r}; expected one of {MODEL_ARCHES}")
