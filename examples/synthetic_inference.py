"""Run the 12-channel BEV detector on a small synthetic point cloud."""

import argparse
from pathlib import Path

import numpy as np
import torch

from bordereye.ml.bev.core.model_factory import build_bev_model
from bordereye.ml.bev.data.bev_channels import (
    GENERIC_V2_MODEL_CHANNELS_12,
    build_generic_v2,
)
from bordereye.ml.bev.predict import BevPredictor


CELL = 0.20
SIZE = 128
CUE_INDICES = (1, 3, 4, 2, 6, 8, 9)


def synthetic_cloud() -> tuple[np.ndarray, np.ndarray]:
    xy = (np.arange(SIZE) + 0.5) * CELL
    x, y = (grid.ravel() for grid in np.meshgrid(xy, xy))
    footprint = (np.abs(x - 12.8) < 2.2) & (np.abs(y - 12.8) < 0.9)
    roof_height = 1.5 + 0.15 * np.cos((x[footprint] - 12.8) * np.pi / 4.4)
    return np.column_stack((x, y, np.zeros_like(x)))[~footprint], np.column_stack(
        (x[footprint], y[footprint], roof_height)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, help="optional retained seed-123 checkpoint")
    options = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    ground, roof = synthetic_cloud()
    points = np.concatenate((ground, roof))
    names = GENERIC_V2_MODEL_CHANNELS_12
    extent = SIZE * CELL
    channels, _, _ = build_generic_v2(points, points[:, 2], CELL,
                                      (0.0, 0.0, extent, extent), channel_names=names)
    if options.weights is None:
        model = build_bev_model("deep", len(names), base=8, cbam_max=True, cue_idx=CUE_INDICES)
        weights = {
            "model": model.state_dict(),
            "arch": "deep",
            "arch_base": 8,
            "args": {"cbam_max": True},
            "cue_idx": CUE_INDICES,
            "reg_gate": True,
            "n_ch": len(names),
            "cell": CELL,
            "channel_names": names,
        }
        print("Synthetic smoke test with untrained weights")
    else:
        weights = options.weights
        print("Synthetic scene with the supplied checkpoint")
    predictor = BevPredictor(weights, device="cpu", channel_names=names)
    result = predictor.predict(channels, window=SIZE, overlap=32,
                               confidence=0.25, score_mode="hm_seg_center_obb",
                               nms_iou=0.20, amp=False, batch_size=1)
    if channels.shape != (12, SIZE, SIZE) or not np.any(channels[1] > 0):
        raise RuntimeError("The synthetic scene did not produce the expected BEV channels.")
    if not np.isfinite(result.heatmap).all() or not np.isfinite(result.segmentation).all():
        raise RuntimeError("Inference produced a non-finite output.")
    print(f"{len(points)} synthetic points, {channels.shape} BEV input, "
          f"{len(result.detections)} detections")


if __name__ == "__main__":
    main()
