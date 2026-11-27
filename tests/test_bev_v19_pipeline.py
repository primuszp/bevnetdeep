"""Fast regression tests for the supported BEV3 v19 production path."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from bordereye.ml.bev.core.inference import decode_window, load_bevnet_checkpoint
from bordereye.ml.bev.core.model_factory import build_bev_model
from bordereye.ml.bev.core.nn_contract import REG_DIM, unpack_model_output
from bordereye.ml.bev.data.bev_channels import GENERIC_V2_MODEL_CHANNELS_12
from bordereye.ml.bev.data.paths import config_root
from bordereye.ml.bev.jeli import _checkpoint_meta, _features_from_names

V19_CONFIG = config_root() / "v19_final.json"


def _v19_checkpoint(path: Path) -> None:
    names = tuple(GENERIC_V2_MODEL_CHANNELS_12)
    model = build_bev_model("deep", 12, base=8, cbam_max=True)
    torch.save({
        "model": model.state_dict(),
        "arch": "deep",
        "arch_base": 8,
        "args": {"cbam_max": True},
        "reg_gate": True,
        "n_ch": 12,
        "channel_names": names,
        "cell": 0.1,
        "norm_config": {"mode": "legacy_tile_percentile"},
        "segmentation_contract": "rounded_box",
    }, path)


def test_v19_config_keeps_the_production_contract():
    cfg = json.loads(V19_CONFIG.read_text(encoding="utf-8"))
    assert cfg["arch"] == "deep"
    assert cfg["seg_target"] == "rounded_box"
    assert cfg["sources"] == "synth,synthsparse,dales,wisar"
    assert cfg["use_channels"] == "default"
    assert cfg["checkpoint_metric"] == "footprint_composite"


def test_v19_checkpoint_runs_inference_and_jeli_contract(tmp_path):
    checkpoint = tmp_path / "v19.pt"
    _v19_checkpoint(checkpoint)

    cell, names, norm = _checkpoint_meta(checkpoint)
    assert cell == 0.1
    assert len(names) == 12
    assert norm["mode"] == "legacy_tile_percentile"
    assert _features_from_names(names, "auto") == "generic_v2"

    model, metadata, _, _ = load_bevnet_checkpoint(checkpoint, torch.device("cpu"))
    assert metadata["n_ch"] == 12
    with torch.inference_mode():
        hm, reg, seg = unpack_model_output(model(torch.zeros(1, 12, 64, 64)))
    assert hm.shape == seg.shape == (1, 1, 32, 32)
    assert reg.shape == (1, REG_DIM, 32, 32)

    detections = decode_window(
        hm[0], reg[0], r0=0, c0=0, origin=(0.0, 0.0), cell=0.1,
        H=64, conf=1.1, seg_logits=seg[0], score_mode="hm_seg_center_obb",
    )
    assert detections == []
