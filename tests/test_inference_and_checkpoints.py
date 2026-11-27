"""Edge coverage and failure recovery for inference and training."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bordereye.ml.bev.core.inference import run_sliding_window
from bordereye.ml.bev.training.checkpointing import save_checkpoint
from bordereye.ml.bev.training.setup import build_and_resume_model


class SampleEverySecondPixel(torch.nn.Module):
    output_stride = 2

    def forward(self, tile):
        logits = tile[:, :1, ::2, ::2]
        return logits, logits.expand(-1, 8, -1, -1), logits


@pytest.mark.parametrize("overlap", [0, 1, 2])
def test_stitched_maps_match_full_image_on_odd_dimensions(overlap):
    channels = np.linspace(-2, 2, 9 * 11, dtype=np.float32).reshape(1, 9, 11)
    _, heatmap, segmentation = run_sliding_window(
        SampleEverySecondPixel(), channels, torch.device("cpu"),
        cell=0.2, win=4, overlap=overlap, batch_size=3, conf=1.1,
    )
    expected = torch.from_numpy(channels[0, ::2, ::2]).sigmoid().numpy()
    np.testing.assert_allclose(heatmap, expected)
    np.testing.assert_allclose(segmentation, expected)


def test_interrupted_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "last.pt"
    save_checkpoint({"epoch": 1}, checkpoint)

    def fail_during_save(payload, path):
        path.write_bytes(b"incomplete")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_during_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint({"epoch": 2}, checkpoint)
    assert torch.load(checkpoint, weights_only=True) == {"epoch": 1}
    assert list(tmp_path.iterdir()) == [checkpoint]


def test_missing_explicit_warm_start_fails(tmp_path):
    args = SimpleNamespace(
        arch="deep", arch_base=8, cbam_max=True, no_reg_gate=False,
        resume=str(tmp_path / "missing.pt"),
    )
    with pytest.raises(FileNotFoundError, match="--resume"):
        build_and_resume_model(args, 12, (), None, torch.device("cpu"), tmp_path)
