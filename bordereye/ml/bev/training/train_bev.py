"""
train_bev.py — BEV-YOLO trainer with seg<->OBB consistency.

This is the main training entry point:
  * data loading / augmentation        -> core.data.BevCrops
  * model + standard losses            -> core.model_factory / core.losses
  * seg<->OBB consistency loss          -> core.consistency

The consistency term lets sources that only have segmentation labels (no OBB GT)
still shape the box-regression head, by tying the predicted box to the predicted
(well-supervised) segmentation. See core.consistency for the rationale.

Serious runs write to output/runs/<run-name> by default. Smoke/test runs write
to output/_tests/<run-name>, so temporary artifacts are easy to delete. Use
--output-dir to override the run directory explicitly.

Run through the stable package workflow after building the dataset:
  python -m bordereye.ml.bev.workflow train --config recipe.json --smoke
"""

from __future__ import annotations

import contextlib
import csv
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import bordereye.ml.bev.core.data as T
from bordereye.ml.bev.core.consistency import (
    gaussian_obb_reg_loss,
    seg_component_heatmap_loss,
    seg_mask_obb_explanation_loss,
    seg_moment_obb_loss,
    seg_obb_consistency_loss,
    seg_orientation_loss,
    seg_z_height_consistency_loss,
    shadow_box_consistency_loss,
)
from bordereye.ml.bev.core.losses import (
    angle_loss,
    class_aware_yaw_geodesic_loss,
    vertical_overlap_loss,
)
from bordereye.ml.bev.core.mask_obb_teacher import VehiclePrior
from bordereye.ml.bev.core.nn_contract import (
    REG_DIM,
    channel_scale,
    output_stride_for_arch,
    select_class_regression,
    unpack_model_output,
)
from bordereye.ml.bev.core.normalization import (
    compute_train_stats_norm,
    load_norm_config,
    validate_norm_config,
)
from bordereye.ml.bev.core.orientation import (
    WISAR_NO_ORIENTATION_CLASS_IDS,
    WISAR_PI_SYMMETRIC_CLASS_IDS,
)
from bordereye.ml.bev.data.paths import project_root
from bordereye.ml.bev.data.selection import parse_csv_set
from bordereye.ml.bev.training.checkpointing import checkpoint_score, save_checkpoint, segmentation_contract
from bordereye.ml.bev.training.engine import (
    LossOut,
    nonfinite_grad_summary,
    restore_bn_state,
    snapshot_bn_state,
)
from bordereye.ml.bev.training.reporting import (
    append_experiment_log,
    make_curves_plot,
    resolve_output_dir,
    save_epoch_viz,
)
from bordereye.ml.bev.training.runtime import setup_diagnostics
from bordereye.ml.bev.training.setup import (
    build_and_resume_model,
    channel_dropout_vector,
    move_batch,
    resolve_cue_idx,
)
from bordereye.ml.bev.training.train_args import parse_args
from bordereye.ml.bev.training.train_console import (
    USE_COLOR,
    format_hms,
    print_epoch_summary,
    progress_bar,
)
from bordereye.ml.bev.training.train_splits import (
    manifest_train_val_split,
    det_supervised_val_tiles,
    stratified_train_val_split,
    tile_cell,
    tile_has_det_supervision,
    tile_has_viz_signal,
    tile_is_test_split,
    tile_schema,
    tile_source,
    validate_train_args,
)
from bordereye.ml.bev.training.validation import ValidationMetricAccumulator
from bordereye.ml.models.checkpoints import resolve_device
from bordereye.paths import datasets_root

# Set by main() so the __main__ crash handler can persist the traceback into
# the run's diagnostics.log even when the console is gone.
_ACTIVE_DIAG = None


def _positive_vector(text: str, count: int, option: str) -> np.ndarray | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        values = np.asarray([float(value.strip()) for value in raw.split(",")], np.float32)
    except ValueError as exc:
        raise ValueError(f"{option} must be comma-separated numbers") from exc
    if len(values) != count or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{option} requires exactly {count} positive finite values")
    return values / values.mean()


def _class_ids(text: str, count: int) -> tuple[int, ...]:
    raw = str(text or "").strip()
    if not raw:
        return ()
    try:
        values = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError("--copy-paste-classes must be comma-separated integers") from exc
    if len(set(values)) != len(values) or any(value < 0 or value >= count for value in values):
        raise ValueError(f"--copy-paste-classes must contain unique ids in [0,{count})")
    return values


def main(argv=None):
    # Make the formatted console output (box chars, °, progress bar) safe on
    # Windows code pages and when redirected to a log file.
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(__doc__, argv)
    root = project_root()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        validate_train_args(args)
    except ValueError as exc:
        print(exc)
        return
    if args.pseudo_peak != 0 and args.pseudo_peak < 0.5:
        print("--pseudo-peak must be 0 or >= 0.5.")
        return
    try:
        allowed_sources = parse_csv_set(args.sources)
        weak_seg_hole_sources = parse_csv_set(args.weak_seg_hole_sources) or set()
        source_weights = T.parse_source_weights(args.source_weights)
        det_loss_weights = T.parse_source_weights(args.det_loss_weights)
        seg_loss_weights = T.parse_source_weights(args.seg_loss_weights)
        channel_dropout_overrides = T.parse_source_weights(args.channel_dropout_overrides)
        class_sampling_weights = _positive_vector(
            args.class_sampling_weights, args.num_classes, "--class-sampling-weights")
        class_heatmap_values = _positive_vector(
            args.class_heatmap_weights, args.num_classes, "--class-heatmap-weights")
        copy_paste_classes = _class_ids(args.copy_paste_classes, args.num_classes)
    except ValueError as exc:
        print(exc)
        return

    dev = resolve_device(args.device)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Detached Windows processes can lose their inherited stdout handle after
    # the launcher exits.  A direct file stream keeps progress reporting from
    # aborting an otherwise healthy training run (WinError/OSError 9).
    console_log = os.environ.get("TRAIN_CONSOLE_LOG", "").strip()
    if console_log:
        console_path = Path(console_log)
        console_path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(console_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
        sys.stdout = stream
        sys.stderr = stream
    diag_path, diag = setup_diagnostics(out_dir)
    global _ACTIVE_DIAG
    _ACTIVE_DIAG = diag
    print(
        f"[{time.strftime('%H:%M:%S')}] run start  |  output: {out_dir}\n"
        f"           warnings/errors -> {diag_path}"
    )
    if args.include_test_split and not args.smoke:
        diag(
            "WARNING",
            "--include-test-split is enabled; this run is not suitable for final metrics.",
        )

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    all_paths = sorted(p for p in dataset_dir.glob("*.npz") if not p.name.endswith(".tmp.npz"))
    if not all_paths:
        # Support train/ + test/ subdirectory layout (build_dataset.py v3+ structure).
        all_paths = sorted(
            p
            for sub in ("train", "test")
            for p in (dataset_dir / sub).glob("*.npz")
            if not p.name.endswith(".tmp.npz")
        )
    if not all_paths:
        print(
            f"No tiles in {dataset_dir}. Run python -m bordereye.ml.bev.data.build_dataset first."
        )
        return

    # Exclude DALES test tiles (leakage guard).
    test_stems = {
        p.stem.replace("_new", "")
        for p in (datasets_root() / "DALESObjects" / "test").glob("*_new.ply")
    }

    def is_test(p: Path) -> bool:
        if p.stem.replace("ply_", "") in test_stems:
            return True
        return tile_is_test_split(p)

    paths = list(all_paths) if args.include_test_split else [p for p in all_paths if not is_test(p)]
    if allowed_sources is not None:
        kept = []
        for p in paths:
            try:
                if tile_source(p) in allowed_sources:
                    kept.append(p)
            except Exception as exc:
                diag("WARNING", f"Skipping unreadable tile during source filter: {p.name} ({exc})")
        paths = kept
    if args.tile_include_file:
        include_path = Path(args.tile_include_file)
        if not include_path.is_absolute():
            include_path = Path.cwd() / include_path
        with include_path.open("r", encoding="utf-8") as f:
            keep_names = {
                line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")
            }
        keep_names |= {Path(name).stem for name in list(keep_names)}
        before = len(paths)
        paths = [p for p in paths if p.name in keep_names or p.stem in keep_names]
        print(f"tile_include_file={include_path} kept {len(paths)}/{before} tile(s)")
    if not paths:
        print("No TRAIN tiles found.")
        return

    cells = {}
    for p in paths:
        try:
            cell_size = tile_cell(p)
            cells[cell_size] = cells.get(cell_size, 0) + 1
        except Exception as exc:
            diag("ERROR", f"Skipping unreadable tile during cell check: {p.name} ({exc})")
            return
    if len(cells) != 1:
        print(f"Mixed BEV cell sizes are not supported in one run: {cells}")
        return
    cell = float(next(iter(cells)))

    schema_by_path = []
    for p in paths:
        try:
            n_tile_ch, tile_names = tile_schema(p)
            schema_by_path.append((p, n_tile_ch, tile_names))
        except FileNotFoundError:
            diag("WARNING", f"Skipping transient/missing tile during schema scan: {p.name}")
    paths = [p for p, _, _ in schema_by_path]
    if not paths:
        print("No readable TRAIN tiles found.")
        return
    n_ch, ch_names = schema_by_path[0][1], schema_by_path[0][2]
    bad_ch = [(p, c) for p, c, _ in schema_by_path if c != n_ch]
    if bad_ch:
        print(
            f"Mixed channel counts in {dataset_dir}; expected {n_ch}. Regenerate incompatible .npz."
        )
        return
    bad_schema = [(p, names) for p, _, names in schema_by_path if names != ch_names]
    if bad_schema:
        print(f"Mixed BEV channel schemas in {dataset_dir}. Regenerate incompatible .npz.")
        return

    stored_ch_names = tuple(ch_names)
    channel_subset = None
    use_channels = getattr(args, "use_channels", "default").strip()
    if use_channels == "default":
        from bordereye.ml.bev.data.bev_channels import GENERIC_V2_MODEL_CHANNELS_12

        if set(GENERIC_V2_MODEL_CHANNELS_12) <= set(ch_names):
            use_channels = ",".join(GENERIC_V2_MODEL_CHANNELS_12)
        else:
            diag(
                "WARNING",
                "[channels] dataset schema lacks the canonical "
                "12-channel subset; training on all stored channels.",
            )
            use_channels = "all"
    if use_channels and use_channels != "all":
        want = [s.strip() for s in use_channels.split(",") if s.strip()]
        missing = [n for n in want if n not in ch_names]
        if missing:
            print(
                f"--use-channels names not in dataset schema: {missing}\n"
                f"available: {list(ch_names)}"
            )
            return
        channel_subset = tuple(ch_names.index(n) for n in want)
        ch_names = tuple(want)
        n_ch = len(want)
        print(
            f"channel subset: training on {n_ch}/{len(schema_by_path[0][2])} "
            f"stored channels: {list(ch_names)}"
        )

    if args.smoke:
        smoke_paths = [p for p in paths if tile_has_det_supervision(p)] or paths
        paths = smoke_paths[:1]
        args.epochs = 2
        args.crops_per_epoch = 16
        args.val_crops = 16
        args.batch = 2
        args.full_coverage = False
        tr, va = paths, paths
    elif getattr(args, "split_manifest", ""):
        tr, va = manifest_train_val_split(paths, args.split_manifest)
    else:
        tr, va = stratified_train_val_split(paths, args.val_frac, args.split_seed)
    if not tr or not va:
        print(
            f"Train/val split failed: train={len(tr)} val={len(va)}. "
            "Use more tiles or adjust --val-frac."
        )
        return
    det_val = det_supervised_val_tiles(va, det_loss_weights)
    if not det_val and not args.allow_no_det_val:
        print("Validation split has no detection-supervised tile with positive det loss weight.")
        print("This would select checkpoints on a non-detection fallback metric.")
        print(
            "Add det-labelled data to validation, adjust --sources/--val-frac, or pass "
            "--allow-no-det-val for an intentional segmentation-only run."
        )
        return

    if args.norm_stats:
        norm_config = load_norm_config(args.norm_stats)
    elif args.norm_mode == "train_stats":
        print(f"normalization: computing train_stats from {len(tr)} train tiles ...")
        norm_config = compute_train_stats_norm(
            tr, channel_scale, max_tiles=int(args.norm_stats_max_tiles)
        )
    else:
        norm_config = validate_norm_config({"mode": args.norm_mode})
    print(
        f"normalization: mode={norm_config['mode']}"
        + (f" stat_tiles={norm_config.get('stat_tiles')}" if "stat_tiles" in norm_config else "")
    )

    output_stride = output_stride_for_arch(args.arch)
    print(
        f"device={dev}  tiles train/val={len(tr)}/{len(va)}  channels={n_ch}  "
        f"cell={cell:g}m  crop={args.crop}  stride={output_stride}  workers={args.workers}"
    )
    print(f"diagnostics: {diag_path}")
    if ch_names:
        print("channel_schema=" + ",".join(ch_names))
    T.summarize_sources(tr, "train")
    T.summarize_sources(va, "val")
    print(
        f"val_det_supervised_tiles={len(det_val)}  split_seed={args.split_seed}  "
        f"training_seed={args.seed}"
    )

    def _sample_weights(paths_for_ds):
        if not source_weights:
            return None
        vals = []
        for p in paths_for_ds:
            try:
                vals.append(source_weights.get(tile_source(p), 1.0))
            except Exception:
                vals.append(1.0)
        return np.asarray(vals, np.float64)

    try:
        cue_idx = resolve_cue_idx(ch_names, args.cue_channels)
        ch_dropout_probs = channel_dropout_vector(ch_names, channel_dropout_overrides)
    except ValueError as exc:
        print(exc)
        return
    if ch_dropout_probs is not None:
        active = [(ch_names[i], float(v)) for i, v in enumerate(ch_dropout_probs) if v > 0]
        print(f"channel dropout overrides: {active}")

    # Every crop stream in this run — train, validation, visualisation and the
    # det-only fallback — must share one labelling and normalisation contract.
    # Building the shared arguments once keeps them from drifting apart, and a
    # newly added option only has to be threaded through in one place.
    crops_labelling = {
        "boundary_coordinate_augmentation": args.boundary_coordinate_augmentation,
        "num_classes": args.num_classes,
        "pseudo_peak": args.pseudo_peak,
        "load_retries": args.load_retries,
        "pseudo_reg_weight": args.pseudo_reg_weight,
        "pseudo_seg_weight": args.pseudo_seg_weight,
        "det_loss_weights": det_loss_weights,
        "seg_loss_weights": seg_loss_weights,
        "norm_config": norm_config,
        "seg_target": args.seg_target,
        "weak_seg_hole_policy": args.weak_seg_hole_policy,
        "weak_seg_hole_sources": weak_seg_hole_sources,
        "weak_seg_hole_soft_target": args.weak_seg_hole_soft_target,
        "weak_seg_hole_soft_weight": args.weak_seg_hole_soft_weight,
        "channel_subset": channel_subset,
        "stored_channel_names": stored_ch_names,
        "output_stride": output_stride,
        "copy_paste_prob": args.copy_paste_prob,
        "copy_paste_max": args.copy_paste_max,
        "copy_paste_classes": copy_paste_classes,
    }
    # Heatmap-radius and tile-ordering knobs. The visualisation stream reads
    # single tiles and deliberately keeps the BevCrops defaults instead.
    crops_sampling = {
        "hm_radius_min": args.hm_radius_min,
        "hm_radius_max": args.hm_radius_max,
        "hm_radius_scale": args.hm_radius_scale,
        "tile_block": args.tile_block,
        "tile_interleave": args.tile_interleave,
    }

    ds_tr = T.BevCrops(
        tr,
        cell,
        args.crop,
        args.crops_per_epoch,
        args.pos_frac,
        seed=args.seed,
        augment=True,
        cache_tiles=args.cache_tiles,
        sample_weights=_sample_weights(tr),
        class_sampling_weights=class_sampling_weights,
        channel_scale_jitter=args.channel_scale_jitter,
        channel_noise_std=args.channel_noise_std,
        channel_dropout_prob=args.channel_dropout_prob,
        channel_dropout_probs=ch_dropout_probs,
        full_coverage=bool(getattr(args, "full_coverage", False)),
        reg_support_radius=args.reg_support_radius,
        reg_support_max_size_m=args.reg_support_max_size_m,
        **crops_labelling,
        **crops_sampling,
    )
    if ds_tr.full_coverage and len(ds_tr) != int(args.crops_per_epoch):
        print(
            f"full-coverage: expanded train crops/epoch from "
            f"{args.crops_per_epoch} to {len(ds_tr)} for {len(tr)} tiles"
        )
    val_crops = args.val_crops if args.val_crops > 0 else max(40, args.crops_per_epoch // 5)
    ds_va = T.BevCrops(
        va,
        cell,
        args.crop,
        val_crops,
        args.pos_frac,
        seed=2,
        augment=False,
        cache_tiles=args.cache_tiles,
        deterministic=True,
        full_coverage=bool(getattr(args, "full_coverage", False)),
        **crops_labelling,
        **crops_sampling,
    )
    if ds_va.full_coverage:
        ds_va.set_epoch(0)  # deterministic val: cover every val tile each epoch
        if len(ds_va) != val_crops:
            print(
                f"full-coverage: expanded val crops from {val_crops} to "
                f"{len(ds_va)} for {len(va)} tiles"
            )

    pin = dev.type == "cuda" and not args.no_pin_memory
    dl_kwargs = {"batch_size": args.batch, "num_workers": max(0, args.workers), "pin_memory": pin}
    if args.workers > 0:
        dl_kwargs.update(
            {"persistent_workers": True, "prefetch_factor": 2, "worker_init_fn": T.seed_worker}
        )
    tr_kwargs = dict(dl_kwargs)
    if ds_tr.full_coverage and args.workers > 0:
        # workers must re-fork each epoch to pick up the new per-epoch tile plan
        tr_kwargs["persistent_workers"] = False
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    tr_kwargs["generator"] = loader_generator
    dl_tr = DataLoader(ds_tr, **tr_kwargs)
    dl_va = DataLoader(ds_va, **dl_kwargs)

    if cue_idx:
        print(f"cue-prior head on channels: {[ch_names[i] for i in cue_idx]}")
    else:
        print("cue-prior head disabled")
    shadow_idx = ch_names.index("ground_shadow") if "ground_shadow" in ch_names else None
    if args.shadow_weight > 0 and shadow_idx is None:
        diag("WARNING", "[shadow] --shadow-weight > 0 but no 'ground_shadow' channel; disabling.")
    built = build_and_resume_model(args, n_ch, ch_names, cue_idx, dev, out_dir)
    if built is None:
        raise ValueError("Training aborted because the requested checkpoint is incompatible.")
    model, loaded_ck = built
    if int(getattr(model, "output_stride", output_stride)) != output_stride:
        raise RuntimeError("model and target output strides disagree")

    if args.heatmap_only_finetune:
        from bordereye.ml.bev.training.finetune import freeze_for_heatmap_finetuning
        if loaded_ck is None or str(args.sources).lower() != "wisar":
            raise ValueError("heatmap fine-tuning requires a WiSAR checkpoint")
        count = freeze_for_heatmap_finetuning(model)
        print(f"heatmap-only fine-tuning: {count} trainable parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    use_amp = bool(args.amp) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.amp and not use_amp:
        diag("WARNING", "[amp] --amp requested but device is not CUDA; running in fp32.")
    sched = T.WarmupThenPlateau(opt, args.warmup_epochs, factor=0.5, patience=6, min_lr=1e-7)
    n_batches = (len(ds_tr) + args.batch - 1) // args.batch
    best = float("inf")
    best_seg = float("inf")
    bad = 0
    start_epoch = 0
    ema_val = None
    ramp_steps = 0.0  # plateau-aware teacher ramp progress (good-epoch counter)
    if args.resume_training_state and loaded_ck is not None:
        if "optimizer" in loaded_ck:
            opt.load_state_dict(loaded_ck["optimizer"])
        else:
            diag(
                "WARNING",
                "--resume-training-state requested, but checkpoint has no optimizer state.",
            )
        if "scheduler" in loaded_ck:
            sched.load_state_dict(loaded_ck["scheduler"])
        else:
            diag(
                "WARNING",
                "--resume-training-state requested, but checkpoint has no scheduler state.",
            )
        # best_val is the det-specific metric for checkpoints saved by this
        # version; resuming a pre-val_det checkpoint re-baselines once (its
        # best_val was on the old total-loss scale) — harmless, just a one-off
        # "improvement" on the first epoch that resets bad=0.
        best = float(loaded_ck.get("best_val", best))
        best_seg = float(loaded_ck.get("best_seg", best_seg))
        bad = int(loaded_ck.get("bad_epochs", bad))
        # ema_val is None in checkpoints saved before any finite save metric.
        # Keep it as None when the checkpoint has no finite EMA/best baseline;
        # otherwise an inf EMA would never improve after resume.
        ema_raw = loaded_ck.get("ema_val", None)
        if ema_raw is None:  # noqa: SIM108 - explicit checkpoint migration branch
            ema_val = best if math.isfinite(best) else None
        else:
            ema_val = float(ema_raw)
        if ema_val is not None and not math.isfinite(ema_val):
            # The checkpoint being resumed may have been saved during the
            # NaN-EMA-poisoning bug; fall back to best_val rather than
            # restarting from a permanently-NaN EMA.
            diag(
                "WARNING",
                f"[resume] checkpoint ema_val is non-finite ({ema_val}); resetting to best_val={best:.4f}.",
            )
            ema_val = best if math.isfinite(best) else None
        ramp_steps = float(loaded_ck.get("ramp_steps", ramp_steps))
        start_epoch = int(loaded_ck.get("epoch", -1)) + 1
        print(
            f"resumed training state: start_epoch={start_epoch} best={best:.4f} bad={bad} ramp_steps={ramp_steps:.0f}"
        )
        if start_epoch >= args.epochs:
            print(
                f"Checkpoint is already at epoch {start_epoch - 1}; increase --epochs to continue."
            )
            return
    elif loaded_ck is not None and args.resume:
        ck_ep = loaded_ck.get("epoch", None)
        if ck_ep is not None:
            print(
                "[resume] loaded model weights only; optimizer/scheduler/warmup start at epoch 0. "
                "Use --resume-training-state only when continuing the same run."
            )
        # Do NOT carry over ema_val / best_val / bad_epochs from the source
        # checkpoint — they reflect that run's metric scale and would make
        # bad_epochs count from a baseline the new run can never beat during
        # warmup, effectively freezing the teacher ramp from epoch 3.
        best = float("inf")
        bad = 0
        ema_val = None
    # Model/optimizer/scheduler state has been copied into the live objects.
    # Do not retain a second complete checkpoint for the rest of training.
    epoch_resume_state = None
    if getattr(args, "epoch_resume_snapshot", False) and args.resume_training_state:
        if loaded_ck is None or "epoch_resume_state" not in loaded_ck:
            raise ValueError("Restart requires an epoch-end resume snapshot")
        if "optimizer" not in loaded_ck or "scheduler" not in loaded_ck:
            raise ValueError("Restart snapshot lacks optimizer/scheduler")
        epoch_resume_state = loaded_ck["epoch_resume_state"]
    del loaded_ck
    vehicle_prior = VehiclePrior(
        length_m=float(args.prior_length),
        width_m=float(args.prior_width),
        length_sigma=float(args.prior_length_sigma),
        width_sigma=float(args.prior_width_sigma),
    )
    # Metric size of one output-grid cell; 0 disables anchor splitting.
    anchor_split_cell_m = cell * output_stride if args.mask_anchor_split else 0.0
    class_heatmap_weights = (
        torch.as_tensor(class_heatmap_values, device=dev)
        if class_heatmap_values is not None else None
    )
    reg_weights = torch.ones(REG_DIM, dtype=torch.float32, device=dev)
    reg_weights[4:6] = float(args.angle_weight)  # sin2θ, cos2θ
    reg_weights[6] = float(args.z_reg_weight)  # z (metres)
    reg_weights[7] = float(args.height_reg_weight)  # height (metres)
    log_path = out_dir / "train_log.csv"
    plot_path = out_dir / "training_curves.png"
    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.resume_training_state and log_path.exists():
        # Continue the same run_id so the curve plot (filtered by run_id) and
        # the CSV stay one continuous series instead of starting a new segment.
        try:
            with log_path.open("r", newline="", encoding="utf-8") as f:
                last_run_id = None
                for r in csv.DictReader(f):
                    if r.get("run_id"):
                        last_run_id = r["run_id"]
                if last_run_id:
                    run_id = last_run_id
        except Exception:
            pass
    # Per-run unique tag (from the checkpoint name) so logs/plots/viz of different
    # runs don't overwrite each other.
    smoke_default_name = args.out_name == "best_model_consistency.pt"
    ckpt_name = "smoke_model.pt" if args.smoke and smoke_default_name else args.out_name
    last_name = (
        "smoke_model_last.pt"
        if args.smoke and smoke_default_name
        else f"{Path(args.out_name).stem}_last.pt"
    )
    ckpt_out = out_dir / ckpt_name
    ckpt_last = out_dir / last_name

    with np.load(paths[0]) as d0:
        saved_ch_scale = channel_scale(d0)
        if channel_subset is not None:
            saved_ch_scale = saved_ch_scale[list(channel_subset)]

    # Fixed validation crops for per-epoch qualitative pictures: keep one
    # representative crop per source, so mixed-domain runs show each domain.
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    occ_idx = next(
        (ch_names.index(n) for n in ("veh_hmax", "all_hmax", "occ_all") if n in ch_names), 0
    )

    def _safe_source_name(name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))

    def _source_of(path: Path) -> str:
        return tile_source(path)

    source_order = []
    for raw in args.sources.split(","):
        s = raw.strip()
        if s:
            source_order.append(s)
    for p in paths:
        s = _source_of(p)
        if s not in source_order:
            source_order.append(s)

    viz_samples = []
    for src in source_order:
        src_paths = [p for p in va if _source_of(p) == src]
        if not src_paths:
            src_paths = [p for p in tr if _source_of(p) == src]
        if not src_paths:
            continue
        src_paths = sorted(src_paths, key=lambda p: not tile_has_viz_signal(p))
        sample = None
        tile_name = src_paths[0].name
        for p in src_paths:
            vds = T.BevCrops(
                [p],
                cell,
                args.crop,
                32,
                1.0,
                seed=1000 + len(viz_samples),
                augment=False,
                cache_tiles=1,
                deterministic=True,
                **crops_labelling,
            )
            fallback = None
            for vi in range(len(vds)):
                sample_tuple = vds[vi]
                _crop_t, hm_t, _reg_t, mask_t, _igm_t, seg_t, _seg_w_t, _det_w_t, _seg_src_w_t = (
                    sample_tuple
                )
                fallback = sample_tuple
                if seg_t.sum() > 0 or hm_t.sum() > 0 or mask_t.sum() > 0:
                    sample = fallback
                    tile_name = p.name
                    break
            if sample is not None:
                break
            if fallback is not None:
                sample = fallback
                tile_name = p.name
        if sample is not None:
            viz_samples.append((_safe_source_name(src), sample, tile_name))
    print(
        f"per-epoch viz -> {viz_dir}  (occ channel idx={occ_idx}; "
        f"sources={','.join(s for s, _, _ in viz_samples) or 'none'})"
    )

    def obb_orientation_metrics(preg, reg, mask, det_w, class_ids):
        """GT OBB yaw metrics under the dataset's class-aware angle contract."""
        with torch.no_grad():
            obj = (mask[:, 0] if mask.ndim == 4 else mask) > 0.5  # (B,H,W)
            if det_w.ndim == 1:
                obj = obj & (det_w.view(-1, 1, 1) > 0)
            wisar = str(args.sources).lower() == "wisar"
            if wisar:
                for class_id in WISAR_NO_ORIENTATION_CLASS_IDS:
                    obj = obj & (class_ids != int(class_id))
            if not torch.any(obj):
                return {"yaw_mae_deg": 0.0, "angle_vec_norm": 0.0, "count": 0}
            ps, pc = preg[:, 4][obj], preg[:, 5][obj]
            gs, gc = reg[:, 4][obj], reg[:, 5][obj]
            pn = torch.sqrt(ps * ps + pc * pc).clamp_min(1e-6)
            gn = torch.sqrt(gs * gs + gc * gc).clamp_min(1e-6)
            dot = (ps / pn) * (gs / gn) + (pc / pn) * (gc / gn)
            encoded_err = torch.acos(dot.clamp(-1.0, 1.0)) * (180.0 / np.pi)
            if wisar:
                doubled = torch.zeros_like(class_ids[obj], dtype=torch.bool)
                for class_id in WISAR_PI_SYMMETRIC_CLASS_IDS:
                    doubled |= class_ids[obj] == int(class_id)
                yaw_err_deg = torch.where(doubled, 0.5 * encoded_err, encoded_err)
            else:
                yaw_err_deg = 0.5 * encoded_err
            return {
                "yaw_mae_deg": float(yaw_err_deg.mean().item()),
                "angle_vec_norm": float(pn.mean().item()),
                "count": int(obj.sum().item()),
            }

    def run_losses(
        crop,
        hm,
        reg,
        mask,
        igm,
        seg,
        seg_w,
        det_w,
        seg_src_w,
        cons_w,
        orient_w,
        moment_w,
        zheight_w,
        greg_w,
        shadow_w=0.0,
        seg_explain_w=0.0,
        seg_center_w=0.0,
    ) -> LossOut:
        phm, preg_raw, pseg = unpack_model_output(model(crop))
        target_class_ids = hm.argmax(dim=1)
        preg = select_class_regression(preg_raw, target_class_ids)
        # per-source weighted supervised losses: det_w gates
        # the OBB/heatmap loss so seg-only sources are not penalised on absent GT.
        l_hm = T.batch_focal_loss(
            phm, hm, igm, det_w, class_weights=class_heatmap_weights)
        # WiSAR orientation semantics depend on the dataset, not on whether
        # regression is shared or class-conditioned (controlled head ablation).
        if str(args.sources).lower() == "wisar":
            l_reg = T.batch_reg_l1_class_aware(
                preg, reg, mask, reg_weights, det_w, target_class_ids,
                tuple(sorted(WISAR_NO_ORIENTATION_CLASS_IDS)),
            )
            yaw_valid = torch.ones_like(target_class_ids, dtype=mask.dtype)
            for class_id in WISAR_NO_ORIENTATION_CLASS_IDS:
                yaw_valid *= (target_class_ids != int(class_id)).to(mask.dtype)
            angle_mask = mask * yaw_valid[:, None]
        else:
            l_reg = T.batch_reg_l1(preg, reg, mask, reg_weights, det_w)
            angle_mask = mask
        l_seg = T.batch_seg_loss(
            pseg,
            seg,
            seg_w,
            igm,
            seg_src_w,
            dice_weight=args.seg_dice_weight,
            tversky_weight=args.seg_tversky_weight,
            tversky_alpha=args.seg_tversky_alpha,
            tversky_beta=args.seg_tversky_beta,
        )
        base_loss = l_hm + l_reg + float(args.seg_weight) * l_seg
        # Direct doubled-angle loss on confident cells: sharper angle than
        # L1 on sin2θ/cos2θ, removes 90° axis swaps. Gated to sources with GT boxes.
        det_gate = det_w.mean().clamp(min=0.0)
        if float(args.angle_loss_weight) > 0:
            base_loss = (
                base_loss + float(args.angle_loss_weight) * angle_loss(
                    preg, reg, angle_mask) * det_gate
            )
        if float(args.yaw_geodesic_weight) > 0:
            doubled_ids = WISAR_PI_SYMMETRIC_CLASS_IDS if str(args.sources).lower() == "wisar" else range(args.num_classes)
            no_yaw_ids = WISAR_NO_ORIENTATION_CLASS_IDS if str(args.sources).lower() == "wisar" else ()
            base_loss = base_loss + float(args.yaw_geodesic_weight) * class_aware_yaw_geodesic_loss(
                preg, reg, mask, target_class_ids,
                doubled_yaw_ids=doubled_ids, no_orientation_ids=no_yaw_ids,
                macro_average=bool(args.yaw_macro_average),
            ) * det_gate
        if float(args.vertical_overlap_weight) > 0:
            base_loss = base_loss + float(args.vertical_overlap_weight) * vertical_overlap_loss(
                preg, reg, mask, target_class_ids,
                macro_average=bool(args.vertical_macro_average),
            ) * det_gate
        loss = base_loss
        # Self-supervised teachers help only sources WITHOUT confident boxes; on
        # synth/DALES the exact reg supervision is better. Run the teachers on the
        # seg-only SUB-BATCH so det-supervised tiles receive no teacher gradient at
        # all (a batch-mean scalar previously leaked teacher gradients onto them).
        ss_sel = det_w < 1.0
        n_ss = int(ss_sel.sum().item())
        ss_scale = float((1.0 - det_w)[ss_sel].clamp(0.0, 1.0).mean().item()) if n_ss > 0 else 0.0
        teach = n_ss > 0 and ss_scale > 0.0
        if teach:
            phm_s, preg_s, pseg_s = phm[ss_sel], preg[ss_sel], pseg[ss_sel]
            crop_s, seg_s = crop[ss_sel], seg[ss_sel]

        l_greg_val = 0.0
        if greg_w > 0:
            wisar = str(args.sources).lower() == "wisar"
            l_greg = gaussian_obb_reg_loss(
                preg, reg, mask, cell=cell, stride=output_stride,
                min_box_m=(0.1 if wisar else 1.0),
                class_ids=target_class_ids,
                doubled_yaw_ids=(WISAR_PI_SYMMETRIC_CLASS_IDS if wisar else None),
            )
            loss = loss + float(greg_w) * l_greg * det_gate
            l_greg_val = float(l_greg.item())
        l_cons_val = 0.0
        if cons_w > 0 and teach:
            l_cons, _ = seg_obb_consistency_loss(
                phm_s,
                preg_s,
                pseg_s,
                cell=cell,
                stride=output_stride,
                topk=args.consistency_topk,
                hm_thresh=args.consistency_hm_thresh,
                sharp=args.consistency_sharp,
            )
            loss = loss + cons_w * l_cons * ss_scale
            l_cons_val = float(l_cons.item())
        l_orient_val = 0.0
        orient_info = {"active_frac": 0.0, "mean_aniso": 0.0}
        if orient_w > 0 and teach:
            l_orient, orient_info = seg_orientation_loss(
                phm_s,
                preg_s,
                pseg_s,
                topk=args.consistency_topk,
                hm_thresh=args.consistency_hm_thresh,
                win_radius=args.orient_win_radius,
                min_mass=args.orient_min_mass,
                min_aniso=args.orient_min_aniso,
            )
            loss = loss + orient_w * l_orient * ss_scale
            l_orient_val = float(l_orient.item())
        l_moment_val = 0.0
        if moment_w > 0 and teach:
            l_moment, _ = seg_moment_obb_loss(
                phm_s,
                preg_s,
                pseg_s,
                cell=cell,
                stride=output_stride,
                topk=args.consistency_topk,
                hm_thresh=args.consistency_hm_thresh,
                win_radius=args.moment_win_radius,
                min_mass=args.moment_min_mass,
                min_aniso=args.moment_min_aniso,
            )
            loss = (
                loss + moment_w * torch.clamp(l_moment, max=float(args.moment_loss_cap)) * ss_scale
            )
            l_moment_val = float(l_moment.item())
        l_zheight_val = 0.0
        zheight_info = {"active_frac": 0.0, "mean_height_teacher": 0.0}
        if zheight_w > 0 and teach:
            l_zheight, zheight_info = seg_z_height_consistency_loss(
                phm_s,
                preg_s,
                pseg_s,
                crop_s,
                ch_names,
                cell=cell,
                stride=output_stride,
                topk=args.consistency_topk,
                hm_thresh=args.consistency_hm_thresh,
                win_radius=args.z_height_win_radius,
                min_mass=args.z_height_min_mass,
                min_height=args.z_height_min_height,
                max_height=args.z_height_max_height,
            )
            loss = loss + zheight_w * l_zheight * ss_scale
            l_zheight_val = float(l_zheight.item())
        l_shadow_val = 0.0
        if shadow_w > 0 and shadow_idx is not None and teach:
            l_shadow, _ = shadow_box_consistency_loss(
                phm_s,
                preg_s,
                crop_s[:, shadow_idx : shadow_idx + 1],
                pseg_s,
                cell=cell,
                stride=output_stride,
                topk=args.consistency_topk,
                hm_thresh=args.consistency_hm_thresh,
                sharp=args.consistency_sharp,
                win_radius=args.shadow_win_radius,
                min_mass=args.shadow_min_mass,
                seg_gate=not args.shadow_no_seg_gate,
                precision_only=args.shadow_precision_only,
            )
            loss = loss + shadow_w * l_shadow * ss_scale
            l_shadow_val = float(l_shadow.item())
        l_seg_explain_val = 0.0
        seg_explain_info = {"active_frac": 0.0, "mean_area": 0.0}
        if seg_explain_w > 0 and teach:
            l_seg_explain, seg_explain_info = seg_mask_obb_explanation_loss(
                preg_s,
                seg_s,
                pseg_s,
                cell=cell,
                stride=output_stride,
                max_components=args.seg_explain_max_components,
                seg_thresh=args.seg_explain_thresh,
                min_area=args.seg_explain_min_area,
                max_area=args.seg_explain_max_area,
                sharp=args.consistency_sharp,
                min_aniso=args.seg_explain_min_aniso,
                reliability_power=args.mask_anchor_reliability_power,
                teacher=args.seg_explain_teacher,
                split_cell_m=anchor_split_cell_m,
                prior=vehicle_prior,
                size_prior_weight=args.seg_explain_size_prior_weight,
            )
            loss = loss + seg_explain_w * l_seg_explain * ss_scale
            l_seg_explain_val = float(l_seg_explain.item())
        l_seg_center_val = 0.0
        seg_center_info = {"active_frac": 0.0, "components": 0.0}
        if seg_center_w > 0 and teach:
            l_seg_center, seg_center_info = seg_component_heatmap_loss(
                phm_s,
                seg_s,
                pseg_s,
                max_components=args.seg_center_max_components,
                seg_thresh=args.seg_center_thresh,
                min_area=args.seg_center_min_area,
                max_area=args.seg_center_max_area,
                radius=args.seg_center_radius,
                peak=args.seg_center_peak,
                min_aniso=args.seg_center_min_aniso,
                reliability_power=args.mask_anchor_reliability_power,
                teacher=args.seg_center_teacher,
                split_cell_m=anchor_split_cell_m,
                prior=vehicle_prior,
            )
            loss = loss + seg_center_w * l_seg_center * ss_scale
            l_seg_center_val = float(l_seg_center.item())
        return LossOut(
            loss=loss,
            l_hm=float(l_hm.item()),
            l_reg=float(l_reg.item()),
            l_seg=float(l_seg.item()),
            l_cons=l_cons_val,
            l_orient=l_orient_val,
            l_moment=l_moment_val,
            l_greg=l_greg_val,
            l_base=float(base_loss.item()),
            l_zheight=l_zheight_val,
            l_shadow=l_shadow_val,
            l_seg_explain=l_seg_explain_val,
            l_seg_center=l_seg_center_val,
            zheight_info=zheight_info,
            orient_info=orient_info,
            seg_explain_info=seg_explain_info,
            seg_center_info=seg_center_info,
            yaw_info=obb_orientation_metrics(preg, reg, mask, det_w, target_class_ids),
            pred_reg=preg,
            pred_seg=pseg,
        )

    # Cascade breaker state: recent healthy train losses + rollback budget.
    # Every unstable run so far (v2_020 ep8, v3 ep24, v11 ep23, v15 ep19)
    # showed the same signature — train loss jumping to ~2x its recent level
    # within one epoch, then never recovering. Detecting that jump, restoring
    # the last healthy weights and permanently lowering the LR turns a dead
    # run into a slower-but-alive one.
    diverge_hist: list[float] = []
    diverge_rollbacks = 0

    if epoch_resume_state is not None:
        from bordereye.ml.bev.training.reproducibility import restore_random_state
        restore_random_state(epoch_resume_state["random"], ds_tr.rng, loader_generator, workers=args.workers)
        ds_va.rng.bit_generator.state = epoch_resume_state["validation_rng"]
        scaler.load_state_dict(epoch_resume_state["scaler"])
        diverge_hist = list(epoch_resume_state["diverge_hist"])
        diverge_rollbacks = int(epoch_resume_state["diverge_rollbacks"])

    t_train_start = time.time()
    for ep in range(start_epoch, args.epochs):
        ds_tr.set_epoch(ep)  # rebuild full-coverage tile plan (no-op otherwise)
        # Teacher ramp. For high-trust supervised runs the plateau-aware mode avoids
        # teacher losses growing after det validation has already gone bad. For v6
        # mask-to-OBB adaptation, however, det validation sees only synth/dales while
        # the teacher losses are the actual bridge from YUTO/ECLAIR/ISPRS masks to
        # OBB geometry, so use a monotonic epoch ramp there.
        if args.teacher_ramp_mode == "epoch":
            # Keep the teacher strength continuous across a full-state resume.
            # `ep` is the absolute zero-based epoch index; measuring relative
            # to `start_epoch` would incorrectly reset the ramp to 1 whenever
            # the process is restarted.
            ramp_steps = float(ep + 1)
        else:
            if bad <= 2:
                ramp_steps += 1.0
            elif bad > 6:
                ramp_steps = max(0.0, ramp_steps - 1.0)
            # 3..6: freeze (no change to ramp_steps)

        def _ramp(weight, warmup, steps=ramp_steps):
            if weight <= 0:
                return 0.0
            if warmup <= 0:
                return weight
            return weight * min(1.0, steps / warmup)

        cons_w = _ramp(args.consistency_weight, args.consistency_warmup_epochs)
        orient_w = _ramp(args.orient_weight, args.orient_warmup_epochs)
        moment_w = _ramp(args.moment_weight, args.moment_warmup_epochs)
        zheight_w = _ramp(args.z_height_weight, args.z_height_warmup_epochs)
        shadow_w = _ramp(args.shadow_weight, args.shadow_warmup_epochs)
        seg_explain_w = _ramp(args.seg_explain_weight, args.seg_explain_warmup_epochs)
        seg_center_w = _ramp(args.seg_center_weight, args.seg_center_warmup_epochs)
        model.train()
        if args.heatmap_only_finetune:
            model.eval()
        t0 = time.time()
        print(
            f"[{time.strftime('%H:%M:%S')}] epoch {ep + 1}/{args.epochs}  |  "
            f"training {n_batches} batches (batch={args.batch})  |  "
            f"lr {opt.param_groups[0]['lr']:.2e}  |  teacher ramp {ramp_steps:.0f}",
            flush=True,
        )
        tot = tbase = thm = treg = tseg = tcons = torient = tmoment = tzheight = tgreg = tshadow = (
            0.0
        )
        tseg_explain = tseg_explain_active = tseg_explain_area = 0.0
        tseg_explain_rel = tseg_explain_aniso = 0.0
        tseg_center = tseg_center_active = tseg_center_components = 0.0
        tseg_center_rel = tseg_center_aniso = 0.0
        tgrad = tzheight_active = tzheight_mean_h = 0.0
        skipped_nonfinite = 0
        skipped_nonfinite_loss = 0
        skipped_nonfinite_grad = 0
        retried_nonfinite_loss = 0
        torient_active = torient_aniso = 0.0
        tyaw_sum = 0.0
        tyaw_count = 0
        nb = 0
        for batch in dl_tr:
            crop, hm, reg, mask, igm, seg, seg_w, det_w, seg_src_w = move_batch(batch, dev, pin)
            opt.zero_grad()
            bn_state = snapshot_bn_state(model)
            with torch.amp.autocast("cuda", enabled=use_amp):
                r = run_losses(
                    crop,
                    hm,
                    reg,
                    mask,
                    igm,
                    seg,
                    seg_w,
                    det_w,
                    seg_src_w,
                    cons_w,
                    orient_w,
                    moment_w,
                    zheight_w,
                    args.gaussian_reg_weight,
                    shadow_w,
                    seg_explain_w,
                    seg_center_w,
                )
            if not torch.isfinite(r.loss):
                opt.zero_grad(set_to_none=True)
                restore_bn_state(bn_state)
                retried_nonfinite_loss += 1
                bn_state = snapshot_bn_state(model)
                with torch.amp.autocast("cuda", enabled=False):
                    r_retry = run_losses(
                        crop,
                        hm,
                        reg,
                        mask,
                        igm,
                        seg,
                        seg_w,
                        det_w,
                        seg_src_w,
                        cons_w,
                        orient_w,
                        moment_w,
                        zheight_w,
                        args.gaussian_reg_weight,
                        shadow_w,
                        seg_explain_w,
                        seg_center_w,
                    )
                if torch.isfinite(r_retry.loss):
                    if retried_nonfinite_loss <= 3:
                        diag(
                            "INFO",
                            f"nonfinite loss retry recovered: "
                            f"epoch={ep + 1} batch={nb + skipped_nonfinite + 1}/{n_batches}",
                        )
                    r = r_retry
                else:
                    r = r_retry
            if not torch.isfinite(r.loss):
                skipped_nonfinite += 1
                skipped_nonfinite_loss += 1
                restore_bn_state(bn_state)
                if skipped_nonfinite_loss <= 3:
                    with torch.no_grad():
                        phm_dbg, preg_dbg, pseg_dbg = unpack_model_output(model(crop))
                        l_hm_dbg = T.batch_focal_loss(
                            phm_dbg, hm, igm, det_w,
                            class_weights=class_heatmap_weights)
                        l_reg_dbg = T.batch_reg_l1(preg_dbg, reg, mask, reg_weights, det_w)
                        l_seg_dbg = T.batch_seg_loss(
                            pseg_dbg,
                            seg,
                            seg_w,
                            igm,
                            seg_src_w,
                            dice_weight=args.seg_dice_weight,
                            tversky_weight=args.seg_tversky_weight,
                            tversky_alpha=args.seg_tversky_alpha,
                            tversky_beta=args.seg_tversky_beta,
                        )
                        l_ang_dbg = angle_loss(preg_dbg, reg, mask)

                        def _stat(name, t):
                            finite = torch.isfinite(t)
                            safe = torch.nan_to_num(t.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                            return (
                                f"{name}:finite={bool(finite.all().item())}"
                                f" nonfinite={int((~finite).sum().item())}"
                                f" min={float(safe.min().cpu()):.4g}"
                                f" max={float(safe.max().cpu()):.4g}"
                            )

                    diag(
                        "ERROR",
                        "[nonfinite loss] "
                        f"epoch={ep + 1} batch={nb + skipped_nonfinite + 1}/{n_batches} "
                        f"loss={float(torch.nan_to_num(r.loss.detach()).cpu()):.6g} "
                        f"hm={r.l_hm:.6g} reg={r.l_reg:.6g} seg={r.l_seg:.6g} "
                        f"base={r.l_base:.6g} cons={r.l_cons:.6g} orient={r.l_orient:.6g} "
                        f"moment={r.l_moment:.6g} greg={r.l_greg:.6g} "
                        f"zheight={r.l_zheight:.6g} shadow={r.l_shadow:.6g} "
                        f"seg_explain={r.l_seg_explain:.6g} seg_center={r.l_seg_center:.6g}\n"
                        f"  recompute hm={float(torch.nan_to_num(l_hm_dbg).cpu()):.6g}"
                        f"/finite={bool(torch.isfinite(l_hm_dbg).item())} "
                        f"reg={float(torch.nan_to_num(l_reg_dbg).cpu()):.6g}"
                        f"/finite={bool(torch.isfinite(l_reg_dbg).item())} "
                        f"seg={float(torch.nan_to_num(l_seg_dbg).cpu()):.6g}"
                        f"/finite={bool(torch.isfinite(l_seg_dbg).item())} "
                        f"angle={float(torch.nan_to_num(l_ang_dbg).cpu()):.6g}"
                        f"/finite={bool(torch.isfinite(l_ang_dbg).item())}\n"
                        + "  "
                        + " | ".join(
                            [
                                _stat("crop", crop),
                                _stat("hm_t", hm),
                                _stat("reg_t", reg),
                                _stat("mask_t", mask),
                                _stat("seg_t", seg),
                                _stat("phm", phm_dbg),
                                _stat("preg", preg_dbg),
                                _stat("pseg", pseg_dbg),
                            ]
                        ),
                    )
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(r.loss).backward()
            scaler.unscale_(opt)  # unscale so grad-clip / grad-norm see true gradients
            # One pass over the parameters: clip_grad_norm_ returns the pre-clip
            # global norm, so the separate norm computation is unnecessary.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")
            )
            if not torch.isfinite(grad_norm):
                # Gradients may have been scaled by a non-finite clip factor,
                # but they are discarded below anyway.
                skipped_nonfinite += 1
                skipped_nonfinite_grad += 1
                if skipped_nonfinite_grad <= 3:
                    scale_now = scaler.get_scale() if use_amp else 1.0
                    diag(
                        "ERROR",
                        "[nonfinite grad] "
                        f"epoch={ep + 1} batch={nb + skipped_nonfinite + 1}/{n_batches} "
                        f"grad_norm={float(torch.nan_to_num(grad_norm.detach()).cpu()):.6g} "
                        f"amp={bool(use_amp)} scaler_scale={scale_now:.6g} "
                        f"loss={float(torch.nan_to_num(r.loss.detach()).cpu()):.6g} "
                        f"hm={r.l_hm:.6g} reg={r.l_reg:.6g} seg={r.l_seg:.6g} "
                        f"base={r.l_base:.6g} cons={r.l_cons:.6g} orient={r.l_orient:.6g} "
                        f"moment={r.l_moment:.6g} greg={r.l_greg:.6g} "
                        f"seg_explain={r.l_seg_explain:.6g} seg_center={r.l_seg_center:.6g}\n"
                        "  " + nonfinite_grad_summary(model),
                    )
                restore_bn_state(bn_state)
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(opt)
            scaler.update()
            tot += r.loss.item()
            thm += r.l_hm
            treg += r.l_reg
            tseg += r.l_seg
            tbase += r.l_base
            tcons += r.l_cons
            torient += r.l_orient
            tmoment += r.l_moment
            tzheight += r.l_zheight
            tgreg += r.l_greg
            tshadow += r.l_shadow
            tseg_explain += r.l_seg_explain
            tseg_explain_active += float(r.seg_explain_info.get("active_frac", 0.0))
            tseg_explain_area += float(r.seg_explain_info.get("mean_area", 0.0))
            tseg_explain_rel += float(r.seg_explain_info.get("mean_reliability", 0.0))
            tseg_explain_aniso += float(r.seg_explain_info.get("mean_aniso", 0.0))
            tseg_center += r.l_seg_center
            tseg_center_active += float(r.seg_center_info.get("active_frac", 0.0))
            tseg_center_components += float(r.seg_center_info.get("components", 0.0))
            tseg_center_rel += float(r.seg_center_info.get("mean_reliability", 0.0))
            tseg_center_aniso += float(r.seg_center_info.get("mean_aniso", 0.0))
            tgrad += float(grad_norm)
            tzheight_active += float(r.zheight_info.get("active_frac", 0.0))
            tzheight_mean_h += float(r.zheight_info.get("mean_height_teacher", 0.0))
            torient_active += float(r.orient_info.get("active_frac", 0.0))
            torient_aniso += float(r.orient_info.get("mean_aniso", 0.0))
            if r.yaw_info["count"] > 0:
                tyaw_sum += r.yaw_info["yaw_mae_deg"] * r.yaw_info["count"]
                tyaw_count += r.yaw_info["count"]
            nb += 1
            # One professional progress line. Live in-place bar on an
            # interactive terminal (or FORCE_COLOR); a sparse plain line
            # (~10 per epoch) otherwise so logs stay readable, never spammed.
            plain_every = max(args.progress_every, max(1, n_batches // 10))
            due = (
                nb == 1
                or nb == n_batches
                or nb % (args.progress_every if USE_COLOR else plain_every) == 0
            )
            if args.progress_every > 0 and due:
                el = time.time() - t0
                ips = nb / max(el, 1e-6)
                eta = (n_batches - nb) / max(ips, 1e-6)
                line = (
                    f"ep {ep + 1:>3}/{args.epochs}  {progress_bar(nb / n_batches)}  "
                    f"{nb:>4}/{n_batches}   "
                    f"obj {tot / nb:6.3f}  base {tbase / nb:6.3f}  "
                    f"[hm {thm / nb:.2f} reg {treg / nb:.2f} seg {tseg / nb:.2f}]  "
                    f"skip {skipped_nonfinite} retry {retried_nonfinite_loss}  "
                    f"{ips:4.1f} it/s  ETA {format_hms(eta)}"
                )
                if USE_COLOR:
                    print("\r\033[K  " + line, end="", flush=True)
                else:
                    print(f"  {time.strftime('%H:%M:%S')}  {line}", flush=True)
        if USE_COLOR:
            print("", flush=True)  # finish the live progress line
        skip_frac = skipped_nonfinite / max(n_batches, 1)
        skip_limit = float(getattr(args, "nonfinite_skip_rollback_frac", 0.25))
        if skip_limit > 0 and skipped_nonfinite > 0 and skip_frac >= skip_limit:
            max_rb = int(getattr(args, "max_divergence_rollbacks", 4))
            lr_scale = float(getattr(args, "divergence_lr_scale", 0.5))
            target = ckpt_out if ckpt_out.exists() else ckpt_last
            reason = (
                f"{skipped_nonfinite}/{n_batches} batches skipped "
                f"({skip_frac:.1%}; loss={skipped_nonfinite_loss}, "
                f"grad={skipped_nonfinite_grad})"
            )
            if target.exists() and diverge_rollbacks < max_rb:
                diverge_rollbacks += 1
                ck_roll = torch.load(target, map_location=dev, weights_only=False)
                model.load_state_dict(ck_roll["model"])
                if "optimizer" in ck_roll:
                    opt.load_state_dict(ck_roll["optimizer"])
                sched.scale_lr(lr_scale)
                bad = 0
                diverge_hist.clear()
                diag(
                    "WARNING",
                    f"[cascade-breaker] epoch {ep + 1}: {reason} — rolled back to "
                    f"epoch {int(ck_roll.get('epoch', -1)) + 1}, lr x{lr_scale} "
                    f"(rollback {diverge_rollbacks}/{max_rb})",
                )
                continue
            if target.exists():
                diag(
                    "WARNING",
                    f"[cascade-breaker] rollback budget exhausted ({diverge_rollbacks}); "
                    f"stopping after numeric instability: {reason}. Best checkpoint remains {ckpt_out}.",
                )
                break
            diag(
                "WARNING",
                f"[cascade-breaker] numeric instability before any rollback checkpoint exists: {reason}.",
            )
        if nb == 0:
            raise RuntimeError(
                "Training produced no successful optimizer steps this epoch: "
                f"nonfinite_loss_batches={skipped_nonfinite_loss}, "
                f"nonfinite_grad_batches={skipped_nonfinite_grad}, "
                f"total_skipped={skipped_nonfinite}. Lower LR/disable AMP/disable teacher "
                "losses or inspect the batch targets before continuing."
            )

        print(
            f"[{time.strftime('%H:%M:%S')}] epoch {ep + 1}/{args.epochs}  |  "
            f"train pass done in {format_hms(time.time() - t0)}  |  "
            f"validating {val_crops} crops ...",
            flush=True,
        )
        model.eval()
        vtot = vhm = vreg = vseg = 0.0
        vyaw_sum = 0.0
        vyaw_count = 0
        vnb = 0
        vdet_num = vdet_den = 0.0  # det-supervised hm+reg, sample-weighted
        vseg_lf_num = vseg_lf_den = 0.0  # seg loss on label-free val tiles only
        validation_metrics = ValidationMetricAccumulator()
        with torch.inference_mode():
            for batch in dl_va:
                crop, hm, reg, mask, igm, seg, seg_w, det_w, seg_src_w = move_batch(batch, dev, pin)
                # Validate on the *baseline* objective (no consistency/orientation) so the
                # metric stays comparable across experiments.
                with torch.amp.autocast("cuda", enabled=use_amp):
                    vr_out = run_losses(
                        crop,
                        hm,
                        reg,
                        mask,
                        igm,
                        seg,
                        seg_w,
                        det_w,
                        seg_src_w,
                        cons_w=0.0,
                        orient_w=0.0,
                        moment_w=0.0,
                        zheight_w=0.0,
                        greg_w=0.0,
                    )
                vh, vr, vs, yaw_info = vr_out.l_hm, vr_out.l_reg, vr_out.l_seg, vr_out.yaw_info
                if not (
                    math.isfinite(float(vr_out.loss))
                    and math.isfinite(float(vh))
                    and math.isfinite(float(vr))
                    and math.isfinite(float(vs))
                ):
                    with torch.amp.autocast("cuda", enabled=False):
                        vr_retry = run_losses(
                            crop,
                            hm,
                            reg,
                            mask,
                            igm,
                            seg,
                            seg_w,
                            det_w,
                            seg_src_w,
                            cons_w=0.0,
                            orient_w=0.0,
                            moment_w=0.0,
                            zheight_w=0.0,
                            greg_w=0.0,
                        )
                    vh, vr, vs, yaw_info = (
                        vr_retry.l_hm,
                        vr_retry.l_reg,
                        vr_retry.l_seg,
                        vr_retry.yaw_info,
                    )
                    if (
                        math.isfinite(float(vr_retry.loss))
                        and math.isfinite(float(vh))
                        and math.isfinite(float(vr))
                        and math.isfinite(float(vs))
                    ):
                        vr_out = vr_retry
                    else:
                        # A single bad crop (e.g. extreme box regression target) can
                        # poison the running sums with NaN/inf; skip it rather than
                        # contaminating every later metric this epoch.
                        continue
                vtot += float(vr_out.loss)
                vnb += 1
                vhm += vh
                vreg += vr
                vseg += vs
                validation_metrics.update(
                    vr_out.pred_seg,
                    seg,
                    seg_w,
                    vr_out.pred_reg,
                    reg,
                    mask,
                    cell=cell,
                    stride=output_stride,
                    class_ids=hm.argmax(dim=1),
                    orientation_source=("wisar" if str(args.sources).lower() == "wisar" else "generic"),
                    no_orientation_ids=(WISAR_NO_ORIENTATION_CLASS_IDS if str(args.sources).lower() == "wisar" else ()),
                )
                # vh/vr are already det_w-weighted batch means (batch_focal_loss /
                # batch_reg_l1 divide by det_w.sum()); re-weight by the effective
                # det-sample count so the metric reflects ONLY sources with real OBB
                # supervision and is not diluted by seg-only/pseudo tiles. This is
                # exact for binary det weights (0/1); with fractional det weights a
                # batch whose det mass < 1 is mildly off due to the clamp(min=1.0).
                det_n = float(
                    ((mask.sum(dim=(1, 2, 3)) > 0) & (det_w > 0)).to(torch.float32).sum().item()
                )
                vdet_num += (vh + vr) * det_n
                vdet_den += det_n
                lf_n = float((1.0 - det_w).clamp(0.0, 1.0).sum().item())
                vseg_lf_num += vs * lf_n
                vseg_lf_den += lf_n
                if yaw_info["count"] > 0:
                    vyaw_sum += yaw_info["yaw_mae_deg"] * yaw_info["count"]
                    vyaw_count += yaw_info["count"]
            if vdet_den <= 0 and det_val and not args.allow_no_det_val:
                # The mixed validation crop stream can occasionally miss all
                # detection-supervised crops even when det-labelled validation
                # tiles exist. Add a short deterministic det-only pass so
                # checkpointing remains based on the intended detection metric
                # instead of requiring --allow-no-det-val.
                det_crops = max(
                    args.batch, min(max(128, len(det_val) * 16), max(val_crops * 2, args.batch))
                )
                det_ds = T.BevCrops(
                    det_val,
                    cell,
                    args.crop,
                    det_crops,
                    1.0,
                    seed=10_000 + ep,
                    augment=False,
                    cache_tiles=args.cache_tiles,
                    deterministic=True,
                    **crops_labelling,
                    **crops_sampling,
                )
                det_dl = DataLoader(det_ds, batch_size=args.batch, num_workers=0, pin_memory=pin)
                diag(
                    "WARNING",
                    f"mixed validation sampled no det-supervised crop; "
                    f"running det-only fallback ({det_crops} crops).",
                )
                fallback_batches = 0
                fallback_pos = 0.0
                fallback_nonfinite = 0
                for batch in det_dl:
                    crop, hm, reg, mask, igm, seg, seg_w, det_w, seg_src_w = move_batch(
                        batch, dev, pin
                    )
                    fallback_batches += 1
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        vr_out = run_losses(
                            crop,
                            hm,
                            reg,
                            mask,
                            igm,
                            seg,
                            seg_w,
                            det_w,
                            seg_src_w,
                            cons_w=0.0,
                            orient_w=0.0,
                            moment_w=0.0,
                            zheight_w=0.0,
                            greg_w=0.0,
                        )
                    vh, vr, yaw_info = vr_out.l_hm, vr_out.l_reg, vr_out.yaw_info
                    if not (math.isfinite(float(vh)) and math.isfinite(float(vr))):
                        with torch.amp.autocast("cuda", enabled=False):
                            vr_retry = run_losses(
                                crop,
                                hm,
                                reg,
                                mask,
                                igm,
                                seg,
                                seg_w,
                                det_w,
                                seg_src_w,
                                cons_w=0.0,
                                orient_w=0.0,
                                moment_w=0.0,
                                zheight_w=0.0,
                                greg_w=0.0,
                            )
                        vh, vr, yaw_info = vr_retry.l_hm, vr_retry.l_reg, vr_retry.yaw_info
                        if not (math.isfinite(float(vh)) and math.isfinite(float(vr))):
                            fallback_nonfinite += 1
                            continue
                    det_n = float(
                        ((mask.sum(dim=(1, 2, 3)) > 0) & (det_w > 0)).to(torch.float32).sum().item()
                    )
                    fallback_pos += det_n
                    vdet_num += (vh + vr) * det_n
                    vdet_den += det_n
                    if yaw_info["count"] > 0:
                        vyaw_sum += yaw_info["yaw_mae_deg"] * yaw_info["count"]
                        vyaw_count += yaw_info["count"]
                diag(
                    "INFO",
                    f"det-only fallback summary: batches={fallback_batches} "
                    f"positive_crops={fallback_pos:.0f} nonfinite={fallback_nonfinite} "
                    f"vdet_den={vdet_den:.0f}",
                )
        vl = vtot / max(vnb, 1)
        # det-specific early-stop metric; only fall back for explicit
        # segmentation-only runs.
        if vdet_den <= 0 and not args.allow_no_det_val:
            diag(
                "WARNING",
                "no finite detection validation metric this epoch even after "
                "det-only fallback; skipping checkpoint metric update for this epoch.",
            )
        val_det = (vdet_num / vdet_den) if vdet_den > 0 else vl
        val_seg_lf = (vseg_lf_num / vseg_lf_den) if vseg_lf_den > 0 else float("nan")
        val_footprint, val_obb = validation_metrics.compute()
        # WiSAR's official mAOE is a macro average over oriented classes;
        # report and checkpoint on the same contract when it is available.
        if "yaw_abs_error_deg_macro" in val_obb:
            vyaw_sum = float(val_obb["yaw_abs_error_deg_macro"])
            vyaw_count = 1
        save_raw = checkpoint_score(
            args,
            val_det,
            val_seg_lf,
            val_footprint,
            val_obb,
            has_detection_metric=(vdet_den > 0 or args.allow_no_det_val),
        )
        # ---- cascade breaker -------------------------------------------------
        # Two failure modes, two detectors:
        #   SPIKE — train loss jumps > factor x its recent median in one epoch
        #     (v11/v15 style). Roll back to *_last.pt (the previous, still-good
        #     epoch), cut LR.
        #   SLOW  — validation drifts > val_ratio x the best-so-far while stuck
        #     (bad_epochs high) for several epochs (v17 style). The rolling
        #     median tracks the creep, so the spike test never fires; instead
        #     roll back to the BEST checkpoint (rolling back to *_last would
        #     just restore another degraded epoch) and cut LR.
        # Either way skip every stateful update of this epoch so the bad epoch
        # cannot poison EMA/best/checkpoints.
        train_loss_now = tot / max(nb, 1)
        div_factor = float(getattr(args, "divergence_factor", 1.75))
        val_ratio = float(getattr(args, "val_divergence_ratio", 1.6))
        slow_pat = int(getattr(args, "slow_divergence_patience", 5))
        max_rb = int(getattr(args, "max_divergence_rollbacks", 4))
        lr_scale = float(getattr(args, "divergence_lr_scale", 0.5))
        div_ready = len(diverge_hist) >= 3
        spike = div_factor > 0 and (
            not math.isfinite(train_loss_now)
            or not math.isfinite(vl)
            or (div_ready and train_loss_now > div_factor * float(np.median(diverge_hist)))
        )
        # bad >= slow_pat already implies several epochs since the best, so
        # `best` is established — no separate warmup guard needed (v17 diverged
        # *during* warmup, so guarding on warmup would have missed it).
        slow = (
            not spike
            and val_ratio > 0
            and best is not None
            and math.isfinite(best)
            and math.isfinite(save_raw)
            and bad >= slow_pat
            and save_raw > val_ratio * best
        )
        if (spike or slow) and diverge_rollbacks < max_rb:
            # Always roll back to BEST when available: the v17 run showed that
            # rolling a spike back to *_last just restores the already-degrading
            # pre-spike epoch and never escapes the bad basin. Best is the
            # safest known-good anchor; fall back to *_last only pre-first-best.
            target = ckpt_out if ckpt_out.exists() else ckpt_last
            if target.exists():
                diverge_rollbacks += 1
                ck_roll = torch.load(target, map_location=dev, weights_only=False)
                model.load_state_dict(ck_roll["model"])
                if "optimizer" in ck_roll:
                    opt.load_state_dict(ck_roll["optimizer"])
                sched.scale_lr(lr_scale)
                reason = (
                    f"train loss {train_loss_now:.3f} > {div_factor:.2f}x median"
                    if spike
                    else f"val {save_raw:.3f} > {val_ratio:.2f}x best {best:.3f} ({bad} bad epochs)"
                )
                bad = 0  # fresh chance from the restored best
                diverge_hist.clear()
                diag(
                    "WARNING",
                    f"[cascade-breaker] epoch {ep + 1}: {reason} — rolled back to "
                    f"epoch {int(ck_roll.get('epoch', -1)) + 1} "
                    f"({'spike' if spike else 'slow'}), lr x{lr_scale} "
                    f"(rollback {diverge_rollbacks}/{max_rb})",
                )
                continue
            diag(
                "WARNING",
                "[cascade-breaker] divergence detected but no "
                "rollback checkpoint exists yet; continuing.",
            )
        elif spike or slow:
            diag(
                "WARNING",
                f"[cascade-breaker] rollback budget exhausted "
                f"({diverge_rollbacks}); stopping training.",
            )
            break
        if math.isfinite(train_loss_now):
            diverge_hist.append(train_loss_now)
            del diverge_hist[:-5]

        # EMA smoothing for checkpointing — removes single-epoch lucky/unlucky
        # crop combinations from driving checkpoint selection.
        alpha = float(args.val_ema_alpha)
        if not math.isfinite(save_raw):
            # Keep the previous EMA rather than poisoning it with NaN forever
            # (a*finite + (1-a)*nan == nan for every future epoch otherwise).
            diag(
                "WARNING",
                f"epoch {ep} save_raw is non-finite ({save_raw}); "
                "keeping previous EMA instead of updating it.",
            )
        else:
            ema_val = (
                alpha * save_raw + (1.0 - alpha) * ema_val if ema_val is not None else save_raw
            )
        train_yaw_mae = tyaw_sum / max(tyaw_count, 1)
        val_yaw_mae = vyaw_sum / max(vyaw_count, 1)
        sched.step()
        sched.step_plateau(vl)
        lr_now = opt.param_groups[0]["lr"]
        star = ""
        ckpt_payload = {
            "model": model.state_dict(),
            "args": vars(args),
            "arch": getattr(args, "arch", "v1"),
            "arch_base": getattr(args, "arch_base", 64),
            "reg_gate": not getattr(args, "no_reg_gate", False),
            "ch_scale": saved_ch_scale,
            "stride": output_stride,
            "norm_config": norm_config,
            "cell": cell,
            "n_ch": n_ch,
            "channel_names": list(ch_names),
            # New WiSAR runs use directed yaw for classes whose official
            # protocol distinguishes θ from θ+π.  Persist this because older
            # checkpoints used doubled yaw for every class and must remain
            # decodable with their original semantics.
            "orientation_encoding": (
                "wisar_class_aware_v1" if str(args.sources).lower() == "wisar"
                else "generic_pi_symmetric_v1"
            ),
            "regression_layout": (
                "shared_plus_class_residual_v1"
                if getattr(args, "arch", "v1") in ("v1_hr_classreg", "v1_hr_boxdetail")
                else "shared_v1"
            ),
            "segmentation_contract": segmentation_contract(args),
            "validation_metrics": {
                "footprint": dict(val_footprint),
                "obb": dict(val_obb),
                "checkpoint_metric": args.checkpoint_metric,
                "checkpoint_score_raw": save_raw,
            },
            # explicit architecture meta so the model can be rebuilt without
            # re-deriving the cue head from the channel schema (see bev_inference).
            "cue_idx": list(cue_idx),
            "cue_channels": [ch_names[i] for i in cue_idx],
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": ep,
            "best_val": best,
            "bad_epochs": bad,
            "ema_val": ema_val,
            "best_seg": best_seg,
            "ramp_steps": ramp_steps,
            "split_seed": args.split_seed,
            "seed": args.seed,
            "val_crops": val_crops,
            "dataset_dir": str(dataset_dir),
            "train_tiles": [p.name for p in tr],
            "val_tiles": [p.name for p in va],
        }
        # Early-stop / checkpoint on the EMA-smoothed save metric: EMA removes
        # lucky/unlucky single-epoch crop variance (~0.05-0.13) from driving
        # checkpoint selection; raw val_det and val_seg_labelfree stay logged.
        # ema_val can still be None when every epoch so far produced a
        # non-finite save metric; such epochs neither save nor count as bad.
        if ema_val is not None and ema_val < best - 1e-4:
            best = ema_val
            bad = 0
            ckpt_payload["best_val"] = best
            ckpt_payload["bad_epochs"] = bad
            ckpt_out.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(ckpt_payload, ckpt_out)
            star = "  *saved"
        elif ema_val is not None:
            bad += 1
            ckpt_payload["bad_epochs"] = bad
        if args.save_best_seg:
            seg_select = val_seg_lf if math.isfinite(val_seg_lf) else (vseg / max(vnb, 1))
            if math.isfinite(seg_select) and seg_select < best_seg - 1e-4:
                best_seg = seg_select
                ckpt_payload["best_seg"] = best_seg
                seg_out = ckpt_out.with_name(ckpt_out.stem + "_best_seg" + ckpt_out.suffix)
                save_checkpoint(ckpt_payload, seg_out)
        ckpt_last.parent.mkdir(parents=True, exist_ok=True)
        if not getattr(args, "epoch_resume_snapshot", False):
            save_checkpoint(ckpt_payload, ckpt_last)
        if args.save_epoch_weights:
            # Numbered per-epoch weights for post-hoc epoch selection/eval.
            # Optimizer/scheduler state is left out (resume uses *_last.pt),
            # so each file is roughly a third of the full training checkpoint.
            ep_dir = out_dir / "epochs"
            ep_dir.mkdir(parents=True, exist_ok=True)
            ep_payload = {
                k: v for k, v in ckpt_payload.items() if k not in ("optimizer", "scheduler")
            }
            ep_stem = Path(ckpt_name).stem
            save_checkpoint(ep_payload, ep_dir / f"{ep_stem}_ep{ep + 1:03d}.pt")
            keep = int(args.epoch_weights_keep)
            if keep > 0:
                snaps = sorted(ep_dir.glob(f"{ep_stem}_ep*.pt"))
                for old_snap in snaps[:-keep]:
                    try:
                        old_snap.unlink()
                    except OSError as exc:
                        diag("WARNING", f"could not prune epoch checkpoint {old_snap.name}: {exc}")
        den = max(nb, 1)
        vden = max(vnb, 1)

        def r8(v):
            """Keep CSV values native and human-readable."""
            return round(float(v), 8)

        append_experiment_log(
            log_path,
            {
                "run_id": run_id,
                "epoch": ep,
                "train_objective": r8(tot / den),
                "train_baseline_loss": r8(tbase / den),
                "train_loss": r8(tot / den),
                "train_hm": r8(thm / den),
                "train_reg": r8(treg / den),
                "train_seg": r8(tseg / den),
                "train_cons": r8(tcons / den),
                "train_orient": r8(torient / den),
                "train_moment": r8(tmoment / den),
                "train_zheight": r8(tzheight / den),
                "train_shadow": r8(tshadow / den),
                "train_seg_explain": r8(tseg_explain / den),
                "train_seg_center": r8(tseg_center / den),
                "train_greg": r8(tgreg / den),
                "consistency_weight": r8(cons_w),
                "orient_weight": r8(orient_w),
                "moment_weight": r8(moment_w),
                "z_height_weight": r8(zheight_w),
                "seg_explain_weight": r8(seg_explain_w),
                "seg_center_weight": r8(seg_center_w),
                "gaussian_reg_weight": r8(args.gaussian_reg_weight),
                "zheight_active_frac": r8(tzheight_active / den),
                "zheight_mean_height_teacher": r8(tzheight_mean_h / den),
                "seg_explain_active_frac": r8(tseg_explain_active / den),
                "seg_explain_mean_area": r8(tseg_explain_area / den),
                "seg_explain_mean_reliability": r8(tseg_explain_rel / den),
                "seg_explain_mean_aniso": r8(tseg_explain_aniso / den),
                "seg_center_active_frac": r8(tseg_center_active / den),
                "seg_center_components": r8(tseg_center_components / den),
                "seg_center_mean_reliability": r8(tseg_center_rel / den),
                "seg_center_mean_aniso": r8(tseg_center_aniso / den),
                "train_yaw_mae_deg": r8(train_yaw_mae),
                "val_yaw_mae_deg": r8(val_yaw_mae),
                "train_orient_active_frac": r8(torient_active / den),
                "train_orient_mean_aniso": r8(torient_aniso / den),
                "val_loss": r8(vl),
                "val_det": r8(val_det),
                "val_save_metric": ("nan" if not math.isfinite(save_raw) else r8(save_raw)),
                "val_det_ema": ("nan" if ema_val is None else r8(ema_val)),
                "val_seg_labelfree": ("nan" if val_seg_lf != val_seg_lf else r8(val_seg_lf)),
                "val_hm": r8(vhm / vden),
                "val_reg": r8(vreg / vden),
                "val_seg": r8(vseg / vden),
                "val_observed_recall": r8(val_footprint["observed_recall"]),
                "val_obb_containment": r8(val_footprint["obb_containment"]),
                "val_outside_allowed_fraction": r8(val_footprint["outside_allowed_fraction"]),
                "val_footprint_fill_ratio": r8(val_footprint["footprint_fill_ratio"]),
                "val_center_error_m": r8(val_obb.get("center_error_m", 0.0)),
                "val_length_abs_error_m": r8(val_obb.get("length_abs_error_m", 0.0)),
                "val_width_abs_error_m": r8(val_obb.get("width_abs_error_m", 0.0)),
                "val_bev_iou": r8(val_obb.get("bev_iou", 0.0)),
                "val_iou3d": r8(val_obb.get("iou3d", 0.0)),
                "val_z_abs_error_m": r8(val_obb.get("z_abs_error_m", 0.0)),
                "val_height_abs_error_m": r8(val_obb.get("height_abs_error_m", 0.0)),
                "best_val": r8(best),
                "lr": r8(lr_now),
                "bad_epochs": bad,
                "grad_norm": r8(tgrad / den),
                "nonfinite_grad_batches": skipped_nonfinite,
                "nonfinite_loss_batches": skipped_nonfinite_loss,
                "nonfinite_grad_only_batches": skipped_nonfinite_grad,
                "retried_nonfinite_loss_batches": retried_nonfinite_loss,
                "grad_clip": r8(args.grad_clip),
                "seg_dice_weight": r8(args.seg_dice_weight),
                "seg_tversky_weight": r8(args.seg_tversky_weight),
                "seg_tversky_alpha": r8(args.seg_tversky_alpha),
                "seg_tversky_beta": r8(args.seg_tversky_beta),
                "channel_scale_jitter": r8(args.channel_scale_jitter),
                "channel_noise_std": r8(args.channel_noise_std),
                "channel_dropout_prob": r8(args.channel_dropout_prob),
                "norm_mode": norm_config["mode"],
                "moment_loss_cap": r8(args.moment_loss_cap),
                "epoch_seconds": round(time.time() - t0, 3),
                "train_tiles": len(tr),
                "val_tiles": len(va),
                "val_crops": val_crops,
                "val_det_tiles": len(det_val),
                "split_seed": args.split_seed,
                "seed": args.seed,
                "sources": args.sources,
                "checkpoint_saved": int(bool(star)),
            },
        )
        make_curves_plot(log_path, plot_path, run_id)
        for src_name, viz_sample, tile_name in viz_samples:
            try:
                # Render once (one model forward) then copy to the alias names,
                # instead of recomputing the forward+decode 2-4x per source.
                latest_only = bool(getattr(args, "viz_latest_only", False))
                target = viz_dir / (
                    f"{src_name}_latest.png" if latest_only else f"{src_name}_epoch_{ep:03d}.png"
                )
                save_epoch_viz(model, viz_sample, dev, target, args, occ_idx, ep, cell)
                if not latest_only:
                    shutil.copyfile(target, viz_dir / f"{src_name}_latest.png")
                # Backward-compatible names for single-source runs.
                if len(viz_samples) == 1:
                    shutil.copyfile(target, viz_dir / "latest.png")
                    if not latest_only:
                        shutil.copyfile(target, viz_dir / f"epoch_{ep:03d}.png")
            except Exception as exc:
                diag("WARNING", f"viz skipped for {src_name} ({tile_name}): {exc}")
        print_epoch_summary(
            ep,
            args.epochs,
            (
                thm / max(nb, 1),
                treg / max(nb, 1),
                tseg / max(nb, 1),
                tcons / max(nb, 1),
                torient / max(nb, 1),
                tmoment / max(nb, 1),
                tzheight / max(nb, 1),
                tshadow / max(nb, 1),
                tgreg / max(nb, 1),
                tot / max(nb, 1),
            ),
            vl,
            best,
            bool(star),
            (cons_w, orient_w, moment_w, zheight_w, shadow_w),
            lr_now,
            bad,
            args.patience,
            time.time() - t0,
            train_yaw_mae,
            val_yaw_mae,
            val_det=val_det,
            val_det_ema=ema_val,
            nonfinite_grad_batches=skipped_nonfinite,
            nonfinite_loss_batches=skipped_nonfinite_loss,
            nonfinite_grad_only_batches=skipped_nonfinite_grad,
            retried_nonfinite_loss_batches=retried_nonfinite_loss,
        )
        # Whole-run progress: elapsed, average epoch cost, and the projected
        # wall-clock finish assuming no early stop.
        if getattr(args, "epoch_resume_snapshot", False):
            from bordereye.ml.bev.training.reproducibility import capture_random_state
            import copy
            ckpt_payload["epoch_resume_state"] = {
                "random": capture_random_state(ds_tr.rng, loader_generator, workers=args.workers),
                "validation_rng": copy.deepcopy(ds_va.rng.bit_generator.state),
                "scaler": scaler.state_dict(),
                "diverge_hist": list(diverge_hist),
                "diverge_rollbacks": diverge_rollbacks,
            }
            # Keep the committed best weights with the restart checkpoint so
            # an interruption before the next commit cannot advance "best".
            if ckpt_out.exists():
                ckpt_payload["resume_best_checkpoint"] = torch.load(
                    ckpt_out, map_location="cpu", weights_only=False)
            # Commit after logging/visualization. A partial write leaves the
            # preceding epoch's restart checkpoint intact.
            save_checkpoint(ckpt_payload, ckpt_last)
        done_eps = ep - start_epoch + 1
        total_el = time.time() - t_train_start
        avg_ep = total_el / max(done_eps, 1)
        remaining = avg_ep * (args.epochs - ep - 1)
        finish = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remaining))
        print(
            f"   elapsed {format_hms(total_el)}  |  avg epoch {format_hms(avg_ep)}  |  "
            f"full-run ETA {format_hms(remaining)} (~{finish}, without early stop)",
            flush=True,
        )
        if bad >= args.patience and (ep + 1) >= args.min_epochs:
            print(
                f"[{time.strftime('%H:%M:%S')}] early stop at epoch {ep + 1} "
                f"(no EMA improvement for {bad} epochs)."
            )
            break

    print(f"Done. best val={best:.4f}  ckpt={ckpt_out}")
    print(f"Last: {ckpt_last}")
    print(f"Log:  {log_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted by user]")
        if _ACTIVE_DIAG is not None:
            _ACTIVE_DIAG("WARNING", "training interrupted by user (KeyboardInterrupt)")
    except Exception:
        import traceback

        tb = traceback.format_exc()
        if _ACTIVE_DIAG is not None:
            _ACTIVE_DIAG("ERROR", "training aborted by exception:\n" + tb)
            print("\n[fatal] traceback also written to diagnostics.log")
        raise
