"""High-level reusable BEV vehicle segmentation and OBB prediction API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bordereye.ml.models.checkpoints import resolve_device

from .core.inference import load_bevnet_checkpoint, nms, run_sliding_window
from .core.normalization import normalize_bev_channels, validate_norm_config


@dataclass(frozen=True)
class BevPrediction:
    """Linked outputs from one BEV vehicle prediction."""

    detections: list[dict[str, Any]]
    heatmap: np.ndarray
    segmentation: np.ndarray


class BevPredictor:
    """Load one BEV checkpoint and reuse it for normalized predictions.

    Raw channels are normalized with the contract stored in the checkpoint.
    Pass ``normalized=True`` only when the caller has already applied that
    exact contract.
    """

    def __init__(
        self,
        weights: str | Path | Mapping[str, Any],
        *,
        device: str | torch.device | None = "auto",
        channel_names: Sequence[str] | None = None,
    ) -> None:
        self.device = resolve_device(device)
        expected = tuple(str(name) for name in (channel_names or ()))
        model, checkpoint, channel_scale, checkpoint_names = load_bevnet_checkpoint(
            weights,
            self.device,
            expected_channel_names=expected,
        )
        self.model = model
        # Retain lightweight metadata, not a second CPU copy of model and
        # optimizer tensors after the live model has been constructed.
        self.checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model", "optimizer", "scheduler"}
        }
        self.channel_scale = channel_scale
        self.channel_names = checkpoint_names or expected
        self.normalization = validate_norm_config(
            checkpoint.get("norm_config", {"mode": "legacy_tile_percentile"})
        )
        value = checkpoint.get("cell")
        self.cell = float(value) if value is not None else None

    @property
    def channel_count(self) -> int:
        return int(self.checkpoint.get("n_ch", len(self.channel_names) or len(self.channel_scale)))

    def normalize(self, channels: np.ndarray) -> np.ndarray:
        """Apply this checkpoint's channel normalization contract."""
        values = np.asarray(channels, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != self.channel_count:
            raise ValueError(
                f"channels must have shape ({self.channel_count},H,W), got {values.shape}"
            )
        scale = self.channel_scale if len(self.channel_scale) else None
        return normalize_bev_channels(values, scale, self.normalization)

    def predict(
        self,
        channels: np.ndarray,
        *,
        cell: float | None = None,
        channel_names: Sequence[str] | None = None,
        normalized: bool = False,
        origin: tuple[float, float] | np.ndarray = (0.0, 0.0),
        window: int = 512,
        overlap: int = 64,
        confidence: float = 0.30,
        score_mode: str = "hm",
        nms_iou: float = 0.20,
        amp: bool = True,
        batch_size: int = 0,
    ) -> BevPrediction:
        """Predict vehicle segmentation and NMS-filtered OBBs from BEV channels."""
        values = np.asarray(channels, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != self.channel_count:
            raise ValueError(
                f"channels must have shape ({self.channel_count},H,W), got {values.shape}"
            )
        supplied_names = tuple(str(name) for name in (channel_names or ()))
        if supplied_names and self.channel_names and supplied_names != self.channel_names:
            raise ValueError(
                "Prediction channel schema does not match the checkpoint:\n"
                f"  checkpoint: {list(self.channel_names)}\n"
                f"  supplied:   {list(supplied_names)}"
            )
        effective_cell = self.cell if cell is None else float(cell)
        if effective_cell is None:
            raise ValueError("cell is required because the checkpoint does not record it")
        if not normalized:
            values = self.normalize(values)
        detections, heatmap, segmentation = run_sliding_window(
            self.model,
            values,
            self.device,
            cell=effective_cell,
            win=window,
            overlap=overlap,
            conf=confidence,
            score_mode=score_mode,
            amp=amp,
            batch_size=batch_size,
            origin=origin,
        )
        return BevPrediction(nms(detections, nms_iou), heatmap, segmentation)


def predict_bev(
    weights: str | Path | Mapping[str, Any],
    channels: np.ndarray,
    *,
    device: str | torch.device | None = "auto",
    channel_names: Sequence[str] | None = None,
    **options: Any,
) -> BevPrediction:
    """Load a checkpoint and perform one BEV prediction.

    For repeated predictions, construct :class:`BevPredictor` once instead.
    """
    predictor = BevPredictor(weights, device=device, channel_names=channel_names)
    return predictor.predict(channels, channel_names=channel_names, **options)
