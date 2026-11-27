"""Reusable neural-network blocks for BEV vehicle detectors."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GN_GROUPS = 8


def gn(ch: int) -> nn.GroupNorm:
    """GroupNorm used by domain-robust BEV backbones."""
    assert ch % GN_GROUPS == 0, f"channel count {ch} not divisible by {GN_GROUPS}"
    return nn.GroupNorm(GN_GROUPS, ch)


class ResBlock(nn.Module):
    """Residual block using BatchNorm for the compact legacy baseline."""

    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x, inplace=True)


class ResBlockGN(nn.Module):
    """Residual block using GroupNorm for multi-domain training."""

    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), gn(ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False), gn(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x, inplace=True)


class CBAM(nn.Module):
    """Convolutional Block Attention Module with optional avg+max channel pooling."""

    def __init__(self, ch: int, r: int = 8, cbam_max: bool = False):
        super().__init__()
        self.cbam_max = bool(cbam_max)
        mid = max(ch // r, 4)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

    def _channel_attn(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cbam_max:
            return self.channel(x)
        flat = self.channel[1]
        def mlp(t):
            return self.channel[4](self.channel[3](self.channel[2](t)))
        avg = flat(self.channel[0](x))
        mx = flat(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(mlp(avg) + mlp(mx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self._channel_attn(x).view(x.shape[0], -1, 1, 1)
        sa_in = torch.cat([x.max(1, keepdim=True).values, x.mean(1, keepdim=True)], dim=1)
        return x * self.spatial(sa_in)


def down_bn(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 3, 2, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


def proj_bn(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co))
