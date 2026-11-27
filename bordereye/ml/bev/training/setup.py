"""Model construction, checkpoint resume, and channel setup helpers."""
from __future__ import annotations

import numpy as np
import torch

from bordereye.ml.bev.core.model_factory import build_bev_model
from bordereye.ml.bev.core.nn_contract import cue_indices
from bordereye.paths import resolve_checkpoint_path


def move_batch(batch, dev, pin):
    """Move one BevCrops batch to `dev`; det/seg source weights cast to float."""
    crop, hm, reg, mask, igm, seg, seg_w, det_w, seg_src_w = batch

    def to(t):
        return t.to(dev, non_blocking=pin)
    return (to(crop), to(hm), to(reg), to(mask), to(igm), to(seg),
            to(seg_w), to(det_w).float(), to(seg_src_w).float())


def build_and_resume_model(args, n_ch, ch_names, cue_idx, dev, out_dir):
    """Build BevNet/BevNetDeep and optionally warm-start it from a checkpoint.

    Returns (model, loaded_ck) on success, or None to signal that the run should
    abort (an explanatory message has already been printed).
    """
    arch = getattr(args, "arch", "v1")
    model = build_bev_model(
        arch,
        n_ch=n_ch,
        base=args.arch_base if arch == "deep" else None,
        cue_idx=cue_idx,
        cbam_max=args.cbam_max,
        reg_gate=not args.no_reg_gate,
        num_classes=getattr(args, "num_classes", 1),
    ).to(dev)
    loaded_ck = None
    if not args.resume:
        return model, loaded_ck
    ckpt_path = (
        out_dir / args.out_name
        if args.resume == "__best__"
        else resolve_checkpoint_path(args.resume)
    )
    if not ckpt_path.exists():
        if args.resume != "__best__":
            raise FileNotFoundError(f"Requested --resume checkpoint does not exist: {ckpt_path}")
        print(f"--resume: {ckpt_path} not found, starting fresh.")
        return model, loaded_ck

    # Keep the serialized model and optimizer state off the accelerator. Model
    # and optimizer ``load_state_dict`` copy/cast their tensors to the owning
    # parameter device, after which the checkpoint container can be released.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    loaded_ck = ck
    ck_names = tuple(str(x) for x in ck.get("channel_names", ()))
    if ck_names and ch_names and ck_names != ch_names:
        raise ValueError(
            "--resume checkpoint channel schema does not match the selected dataset channels:\n"
            f"  checkpoint: {list(ck_names)}\n"
            f"  selected:   {list(ch_names)}"
        )
    ck_args = ck.get("args", {}) if isinstance(ck.get("args", {}), dict) else {}
    ck_arch = str(ck.get("arch", ck_args.get("arch", "v1")))
    if ck_arch != arch and not args.allow_partial_resume:
        print("Resume checkpoint architecture differs from the requested run:")
        print(f"  checkpoint arch: {ck_arch}")
        print(f"  requested  arch: {arch}")
        print("Use --allow-partial-resume only if this is an intentional warm-start across model variants.")
        return None

    # Both backbones use the same hm x seg regression gate, controlled by
    # no_reg_gate, so those are the only architecture flags worth comparing.
    arch_keys = ("cbam_max", "no_reg_gate")
    arch_mismatch = {
        k: (bool(ck_args.get(k, False)), bool(getattr(args, k, False)))
        for k in arch_keys
        if bool(ck_args.get(k, False)) != bool(getattr(args, k, False))
    }
    if arch_mismatch and not args.allow_partial_resume:
        print("Resume checkpoint architecture flags differ from the requested run:")
        for k, (old, new) in arch_mismatch.items():
            print(f"  {k}: checkpoint={old} requested={new}")
        print("Use --allow-partial-resume only if this is an intentional warm-start across model variants.")
        return None
    # cue-prior head is an architecture variant too. New checkpoints record
    # it explicitly (cue_channels meta); old ones only carry cue_branch.* in
    # the state_dict. Validate by channel name (robust to index shifts) with
    # the state_dict presence as a fallback for pre-meta checkpoints.
    cue_now = [ch_names[i] for i in cue_idx]
    ck_cue = ck.get("cue_channels")
    if ck_cue is None:
        ck_has_cue = any(k.startswith("cue_branch") for k in ck["model"])
        ck_cue = cue_now if (ck_has_cue == bool(cue_idx)) else (
            ["<present>"] if ck_has_cue else [])
    if list(ck_cue) != list(cue_now) and not args.allow_partial_resume:
        print("Resume checkpoint cue-prior head differs from the requested run:")
        print(f"  checkpoint cue channels: {list(ck_cue)}")
        print(f"  requested  cue channels: {cue_now}")
        print("Use --allow-partial-resume for an intentional warm-start across cue variants.")
        return None
    state = ck["model"]
    if getattr(args, "reset_regression_head", False):
        # A target-schema or yaw-contract migration must never inherit a head
        # trained to predict different physical quantities.  Keep the visual
        # representation and detection/segmentation heads, but leave the new
        # regression head at the model factory's fresh initialisation.
        state = {
            key: value for key, value in state.items()
            if not (key.startswith("reg_head.") or key.startswith("reg_residual_head.")
                    or key == "reg_gate_alpha")
        }
        print("warm-start: reset regression head and regression gate calibration")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if getattr(args, "reset_regression_head", False):
        allowed = {
            "reg_head.weight", "reg_head.bias",
            "reg_residual_head.weight", "reg_residual_head.bias",
            "reg_gate_alpha",
        }
        unexpected_reset = set(unexpected)
        disallowed_missing = set(missing) - allowed
        if disallowed_missing or unexpected_reset:
            raise RuntimeError(
                "--reset-regression-head expected only regression parameters to be absent; "
                f"missing={sorted(disallowed_missing)} unexpected={sorted(unexpected_reset)}")
        missing, unexpected = [], []
    if (missing or unexpected) and not args.allow_partial_resume:
        print("Resume checkpoint architecture does not match the current model.")
        print(f"missing keys: {missing}")
        print(f"unexpected keys: {unexpected}")
        print("Use --allow-partial-resume only for intentional warm-starts across architecture changes.")
        return None
    if missing or unexpected:
        print(f"partial resume: missing={missing} unexpected={unexpected}")
    # The live model now owns its copy; the serialized state_dict is the
    # largest part of the checkpoint and is not consulted by the trainer.
    del ck["model"]
    print(f"resumed weights from {ckpt_path}")
    return model, loaded_ck


def resolve_cue_idx(ch_names, value: str):
    """Resolve --cue-channels into channel indices."""
    text = str(value or "auto").strip()
    if not text or text.lower() == "auto":
        return cue_indices(ch_names)
    if text.lower() in ("none", "off", "0", "false"):
        return ()
    names = [p.strip() for p in text.split(",") if p.strip()]
    missing = [n for n in names if n not in ch_names]
    if missing:
        raise ValueError(f"--cue-channels contains unknown channel(s): {missing}; "
                         f"available={list(ch_names)}")
    return tuple(ch_names.index(n) for n in names)


def channel_dropout_vector(ch_names, overrides: dict[str, float]):
    """Return per-channel dropout overrides aligned to ch_names, or None."""
    if not overrides:
        return None
    missing = [name for name in overrides if name not in ch_names]
    if missing:
        raise ValueError(f"--channel-dropout-overrides contains unknown channel(s): {missing}; "
                         f"available={list(ch_names)}")
    vals = np.zeros(len(ch_names), np.float32)
    for i, name in enumerate(ch_names):
        vals[i] = float(overrides.get(name, 0.0))
    return vals
