"""BEV backbones for multi-domain vehicle detection.

Both backbones implement the same public contract:

- input: normalized BEV tensor ``(B, C, H, W)``
- output: ``(hm, reg, seg)``
- output stride: ``STRIDE = 2``
- regression layout: ``REG_DIM = 8``

``BevNet`` is the compact BatchNorm baseline. ``BevNetDeep`` is the canonical
GroupNorm backbone for multi-domain training. Both use the same hm x seg gated
OBB-regression head when ``reg_gate`` is enabled.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_blocks import CBAM, ResBlock, ResBlockGN, down_bn, gn, proj_bn
from .nn_contract import REG_DIM


def _init_head_bias(hm_head: nn.Conv2d, seg_head: nn.Conv2d) -> None:
    nn.init.constant_(hm_head.bias, -2.19)
    nn.init.constant_(seg_head.bias, -2.19)


def _apply_cue_prior(cue_branch: nn.Module | None, cue_idx: tuple[int, ...],
                     x: torch.Tensor, hm: torch.Tensor) -> torch.Tensor:
    if cue_branch is None or not cue_idx:
        return hm
    prior = cue_branch(x[:, cue_idx])
    if prior.shape[2:] != hm.shape[2:]:
        prior = F.interpolate(prior, size=hm.shape[2:], mode="bilinear", align_corners=False)
    return hm + prior


def _gate_reg_features(feat: torch.Tensor, hm: torch.Tensor, seg: torch.Tensor,
                       alpha: torch.Tensor | None) -> torch.Tensor:
    if alpha is None:
        return feat
    # Multi-class detectors share one class-agnostic box-regression head.
    # Collapse class confidence before gating so it broadcasts over features.
    hm_gate = torch.sigmoid(hm.detach()).amax(dim=1, keepdim=True)
    gate = hm_gate * torch.sigmoid(seg.detach())
    return feat * (1.0 + torch.clamp(alpha, 0.0, 4.0) * gate)


class BevNet(nn.Module):
    """Compact BEV detector with BatchNorm and a /4 bottleneck.

    This backbone is kept as a smaller, interpretable baseline. It now shares the
    same hm x seg regression gate used by ``BevNetDeep`` so architectural
    comparisons are not confounded by different head logic.
    """

    def __init__(self, n_ch: int, base: int = 48, dropout: float = 0.1,
                 cue_idx=None, cbam_max: bool = False, reg_gate: bool = True,
                 num_classes: int = 1, high_resolution: bool = False,
                 class_regression: bool = False, box_detail: bool = False):
        super().__init__()
        b = int(base)
        self.cue_idx = tuple(int(i) for i in cue_idx) if cue_idx else ()
        self.cue_branch = None
        if self.cue_idx:
            c = len(self.cue_idx)
            self.cue_branch = nn.Sequential(
                nn.Conv2d(c, b, 3, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                ResBlock(b),
                nn.Conv2d(b, 1, 1),
            )
            nn.init.zeros_(self.cue_branch[-1].weight)
            nn.init.zeros_(self.cue_branch[-1].bias)

        self.stem = nn.Sequential(
            nn.Conv2d(n_ch, b, 3, 1, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
            ResBlock(b),
        )
        self.enc1 = nn.Sequential(down_bn(b, b * 2), ResBlock(b * 2))
        self.enc2 = nn.Sequential(
            down_bn(b * 2, b * 4),
            ResBlock(b * 4),
            ResBlock(b * 4),
            CBAM(b * 4, cbam_max=cbam_max),
        )
        self.lat2 = proj_bn(b * 4, b * 2)
        self.lat1 = proj_bn(b * 2, b * 2)
        self.fuse = nn.Sequential(
            nn.Conv2d(b * 2, b * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True),
            ResBlock(b * 2),
        )
        self.high_resolution = bool(high_resolution)
        self.output_stride = 1 if self.high_resolution else 2
        if self.high_resolution:
            # Preserve the /4 receptive field while restoring stem-level
            # spatial detail before prediction. Heads keep their channel
            # contract, so a v1 checkpoint can warm-start all existing layers.
            self.detail_proj = proj_bn(b, b * 2)
            self.detail_fuse = nn.Sequential(
                nn.Conv2d(b * 4, b * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(b * 2), nn.ReLU(inplace=True), ResBlock(b * 2),
            )
            self.detail_delta = nn.Conv2d(b * 2, b * 2, 1)
            nn.init.zeros_(self.detail_delta.weight)
            nn.init.zeros_(self.detail_delta.bias)
        self.drop = nn.Dropout2d(dropout)
        self.hm_head = nn.Conv2d(b * 2, int(num_classes), 1)
        self.reg_head = nn.Conv2d(b * 2, REG_DIM, 1)
        self.class_regression = bool(class_regression)
        self.num_classes = int(num_classes)
        if self.class_regression:
            self.reg_residual_head = nn.Conv2d(b * 2, self.num_classes * REG_DIM, 1)
            nn.init.zeros_(self.reg_residual_head.weight)
            nn.init.zeros_(self.reg_residual_head.bias)
        self.seg_head = nn.Conv2d(b * 2, 1, 1)
        self.box_detail = None
        if box_detail:
            if not high_resolution or not class_regression:
                raise ValueError('box_detail requires high-resolution class regression')
            # Independent, class-shared geometric residual from full-resolution
            # stem and contextual features. Zero start preserves the checkpoint.
            self.box_detail = nn.Sequential(
                nn.Conv2d(b * 3, b, 3, padding=1), nn.ReLU(),
                nn.Conv2d(b, b, 3, padding=1), nn.ReLU(),
                nn.Conv2d(b, REG_DIM, 1),
            )
            nn.init.zeros_(self.box_detail[-1].weight)
            nn.init.zeros_(self.box_detail[-1].bias)
        _init_head_bias(self.hm_head, self.seg_head)
        self.reg_gate = bool(reg_gate)
        if self.reg_gate:
            self.reg_gate_alpha = nn.Parameter(torch.tensor(0.0))

        if self.box_detail is not None:
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith('box_detail.'))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.box_detail is not None:
            # Keep inherited BatchNorm statistics and dropout fixed, too.
            for name, module in self.named_children():
                module.train(mode if name == 'box_detail' else False)
        return self

    def forward(self, x: torch.Tensor):
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        up = F.interpolate(self.lat2(e2), size=e1.shape[2:], mode="bilinear", align_corners=False)
        feat = self.fuse(up + self.lat1(e1))
        if self.high_resolution:
            coarse = F.interpolate(feat, size=s.shape[2:], mode="bilinear", align_corners=False)
            detail = self.detail_fuse(torch.cat((coarse, self.detail_proj(s)), dim=1))
            feat = coarse + self.detail_delta(detail)
        feat = self.drop(feat)
        pseg = self.seg_head(feat)
        hm = self.hm_head(feat)
        hm = _apply_cue_prior(self.cue_branch, self.cue_idx, x, hm)
        alpha = self.reg_gate_alpha if self.reg_gate else None
        reg_feat = _gate_reg_features(feat, hm, pseg, alpha)
        reg = self.reg_head(reg_feat)
        if self.class_regression:
            residual = self.reg_residual_head(reg_feat).view(
                x.shape[0], self.num_classes, REG_DIM, reg.shape[2], reg.shape[3])
            reg = reg.unsqueeze(1) + residual
        if self.box_detail is not None:
            delta = self.box_detail(torch.cat((s, reg_feat), dim=1))
            # Do not perturb yaw (channels 4,5) in this geometry experiment.
            mask = delta.new_tensor([1, 1, 1, 1, 0, 0, 1, 1])[None, :, None, None]
            reg = reg + (delta * mask).unsqueeze(1)
        return hm, reg, pseg


class BevNetDeep(nn.Module):
    """GroupNorm BEV detector with a /8 bottleneck and 3-level FPN."""

    def __init__(self, n_ch: int, base: int = 64, dropout: float = 0.1,
                 cue_idx=None, cbam_max: bool = True, reg_gate: bool = True,
                 num_classes: int = 1):
        super().__init__()
        b = int(base)
        b2 = b * 2
        b4 = b * 4
        self.reg_gate = bool(reg_gate)
        self.output_stride = 2

        self.cue_idx = tuple(int(i) for i in cue_idx) if cue_idx else ()
        self.cue_branch = None
        if self.cue_idx:
            c = len(self.cue_idx)
            self.cue_branch = nn.Sequential(
                nn.Conv2d(c, b2, 3, 2, 1, bias=False), gn(b2), nn.ReLU(inplace=True),
                ResBlockGN(b2),
                nn.Conv2d(b2, 1, 1),
            )
            nn.init.zeros_(self.cue_branch[-1].weight)
            nn.init.zeros_(self.cue_branch[-1].bias)

        self.stem = nn.Sequential(
            nn.Conv2d(n_ch, b, 3, 1, 1, bias=False), gn(b), nn.ReLU(inplace=True),
            ResBlockGN(b),
        )
        self.enc1 = nn.Sequential(
            nn.Conv2d(b, b2, 3, 2, 1, bias=False), gn(b2), nn.ReLU(inplace=True),
            ResBlockGN(b2),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(b2, b4, 3, 2, 1, bias=False), gn(b4), nn.ReLU(inplace=True),
            ResBlockGN(b4),
            ResBlockGN(b4),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(b4, b4, 3, 2, 1, bias=False), gn(b4), nn.ReLU(inplace=True),
            ResBlockGN(b4),
            ResBlockGN(b4),
            CBAM(b4, cbam_max=cbam_max),
        )
        self.lat3 = nn.Sequential(nn.Conv2d(b4, b4, 1, bias=False), gn(b4))
        self.lat2 = nn.Sequential(nn.Conv2d(b4, b4, 1, bias=False), gn(b4))
        self.fuse4 = nn.Sequential(
            nn.Conv2d(b4, b4, 3, 1, 1, bias=False), gn(b4), nn.ReLU(inplace=True),
            ResBlockGN(b4),
        )
        self.lat4to2 = nn.Sequential(nn.Conv2d(b4, b2, 1, bias=False), gn(b2))
        self.lat1 = nn.Sequential(nn.Conv2d(b2, b2, 1, bias=False), gn(b2))
        self.fuse2 = nn.Sequential(
            nn.Conv2d(b2, b2, 3, 1, 1, bias=False), gn(b2), nn.ReLU(inplace=True),
            ResBlockGN(b2),
        )
        self.drop = nn.Dropout2d(dropout)
        self.hm_head = nn.Conv2d(b2, int(num_classes), 1)
        self.reg_head = nn.Conv2d(b2, REG_DIM, 1)
        self.seg_head = nn.Conv2d(b2, 1, 1)
        _init_head_bias(self.hm_head, self.seg_head)
        if self.reg_gate:
            self.reg_gate_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor):
        s = self.stem(x)
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        up3 = F.interpolate(self.lat3(e3), size=e2.shape[2:], mode="bilinear", align_corners=False)
        p4 = self.fuse4(up3 + self.lat2(e2))
        up4 = F.interpolate(self.lat4to2(p4), size=e1.shape[2:], mode="bilinear", align_corners=False)
        feat = self.drop(self.fuse2(up4 + self.lat1(e1)))

        pseg = self.seg_head(feat)
        hm = self.hm_head(feat)
        hm = _apply_cue_prior(self.cue_branch, self.cue_idx, x, hm)
        alpha = self.reg_gate_alpha if self.reg_gate else None
        reg_feat = _gate_reg_features(feat, hm, pseg, alpha)
        return hm, self.reg_head(reg_feat), pseg
