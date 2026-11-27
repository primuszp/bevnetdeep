"""Command-line and JSON preset handling for BEV training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bordereye.ml.bev.data.paths import config_root, dataset_root

CONFIG_ROOT = config_root()


def _load_train_config(path_text: str) -> dict:
    """Load a JSON training preset and convert option names to argparse dests."""
    path = Path(path_text)
    if not path.is_absolute():
        packaged = CONFIG_ROOT / path
        path = packaged if packaged.exists() else path.resolve()
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"training config must be a JSON object: {path}")
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def _apply_config_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Apply JSON preset defaults before parsing the real command line.

    CLI arguments still win because argparse parses the command line after these
    defaults are installed.
    """
    cfg = _load_train_config(config_path)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    valid = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg if k not in valid)
    if unknown:
        raise ValueError(f"unknown training config key(s): {unknown}")
    parser.set_defaults(**cfg)


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Build the training parser without reading argv."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="",
                   help="JSON training preset. Command-line arguments override preset values.")
    # Supervised training knobs.
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--num-classes", type=int, default=1,
                   help="number of center-heatmap classes; legacy vehicle models use 1")
    p.add_argument("--class-sampling-weights", default="",
                   help="comma-separated positive crop-sampling weights, one per class")
    p.add_argument("--class-heatmap-weights", default="",
                   help="comma-separated positive focal-loss weights, one per class")
    p.add_argument("--copy-paste-prob", type=float, default=0.0,
                   help="probability of geometry-only weak-class copy-paste per training crop")
    p.add_argument("--copy-paste-max", type=int, default=0,
                   help="maximum duplicated objects per augmented crop")
    p.add_argument("--copy-paste-classes", default="",
                   help="comma-separated zero-based class ids eligible for copy-paste")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--crops-per-epoch", type=int, default=400)
    p.add_argument("--pos-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--val-crops", type=int, default=0,
                   help="fixed validation crops per epoch; 0 = max(40, crops_per_epoch/5)")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--split-manifest", default="",
                   help="JSON with fixed train_tiles and val_tiles basenames; overrides random splitting")
    p.add_argument("--seed", type=int, default=42,
                   help="training RNG seed (model init, augmentation, sampling and DataLoader); "
                        "keep --split-seed fixed when measuring repeated-run variance")
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--min-epochs", type=int, default=0)
    p.add_argument("--val-ema-alpha", type=float, default=0.3,
                   help="EMA weight for val_det smoothing used for checkpointing; "
                        "1.0 = no smoothing (raw val_det), 0.1 = very smooth")
    p.add_argument("--checkpoint-metric",
                   choices=("det", "det_plus_seg_lf", "footprint_composite"), default="det",
                   help="metric used for checkpointing/early stop. 'det' (default) uses only "
                        "GT-supervised detection sources (det_w>0, e.g. synth+dales). "
                        "det_plus_seg_lf ALSO adds the label-free seg loss from weak domains "
                        "(det_w==0, e.g. yuto/eclair/isprs) — CAUTION: since weak-domain GT is "
                        "sparse, the model finding MORE vehicles raises this loss, which would "
                        "PENALISE correct behaviour on those sources. Only use det_plus_seg_lf "
                        "when the weak-domain seg GT is complete (e.g. instance-mask runs).")
    p.add_argument("--checkpoint-seg-lf-weight", type=float, default=0.25,
                   help="weight of val_seg_labelfree when --checkpoint-metric=det_plus_seg_lf")
    p.add_argument("--checkpoint-center-weight", type=float, default=0.10)
    p.add_argument("--checkpoint-yaw-weight", type=float, default=0.05,
                   help="weight after normalizing yaw MAE by 45 degrees")
    p.add_argument("--checkpoint-bev-iou-weight", type=float, default=0.15)
    p.add_argument("--checkpoint-3d-iou-weight", type=float, default=0.10)
    p.add_argument("--checkpoint-observed-recall-weight", type=float, default=0.10)
    p.add_argument("--checkpoint-outside-weight", type=float, default=0.15)
    p.add_argument("--save-best-seg", action=argparse.BooleanOptionalAction, default=True,
                   help="also save a checkpoint selected by validation segmentation loss")
    p.add_argument("--save-epoch-weights", action=argparse.BooleanOptionalAction, default=True,
                   help="save numbered model weights after every epoch into "
                        "<output>/epochs/ (model + eval meta, no optimizer state)")
    p.add_argument("--epoch-weights-keep", type=int, default=0,
                   help="keep only the newest N numbered epoch checkpoints; 0 = keep all")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--cache-tiles", type=int, default=4)
    p.add_argument("--tile-block", type=int, default=8,
                   help="crops served per loaded tile before drawing new tiles; "
                        "1 = fully random tile per crop (slow: defeats the tile cache)")
    p.add_argument("--tile-interleave", type=int, default=4,
                   help="distinct tiles interleaved inside one sampling super-block "
                        "(clamped to --cache-tiles); keeps batches mixed across tiles")
    p.add_argument("--no-pin-memory", action="store_true")
    p.add_argument(
        "--resume", nargs="?", const="__best__", default=None,
        help="checkpoint path or checkpoint ID (e.g. training-warm-start)",
    )
    p.add_argument("--resume-training-state", action="store_true",
                   help="also restore optimizer/scheduler/epoch/best counters from a training checkpoint")
    p.add_argument("--epoch-resume-snapshot", action="store_true",
                   help="workers=0: atomically save epoch-end RNG/scaler/divergence state for restart")
    p.add_argument("--allow-partial-resume", action="store_true",
                   help="allow missing/unexpected checkpoint keys when warm-starting across architecture changes")
    p.add_argument("--reset-regression-head", action="store_true",
                   help="when warm-starting, retain backbone/heatmap/segmentation weights but "
                        "reinitialise reg_head and its gate calibration")
    p.add_argument("--out-name", default="best_model_consistency.pt")
    p.add_argument("--output-dir", default="",
                   help="directory for checkpoint/log/plot/viz artifacts; default is "
                        "output/runs/<out-name-stem> or output/_tests/<out-name-stem> with --smoke")
    p.add_argument("--hm-radius-min", type=int, default=1,
                   help="min heatmap Gaussian radius (stride cells); raise to densify "
                        "positive cells for recall ablations")
    p.add_argument("--hm-radius-max", type=int, default=2,
                   help="max heatmap Gaussian radius (stride cells); lower keeps "
                        "closely-parked cars as separate peaks")
    p.add_argument("--hm-radius-scale", type=float, default=0.25,
                   help="heatmap radius = scale * min(L,W)/cell/stride, clamped to "
                        "[--hm-radius-min, --hm-radius-max]")
    p.add_argument("--reg-support-radius", type=int, default=0,
                   help="training-only radius (output cells) for dense OBB regression; "
                        "0 preserves the legacy centre-cell target")
    p.add_argument("--reg-support-max-size-m", type=float, default=0.0,
                   help="apply dense regression only when max(length,width) is at most "
                        "this value; 0 applies it to every object")
    p.add_argument("--pseudo-peak", type=float, default=0.5)
    p.add_argument("--pseudo-reg-weight", type=float, default=0.25)
    p.add_argument("--angle-weight", type=float, default=2.0)
    p.add_argument("--z-reg-weight", type=float, default=1.0,
                   help="L1 weight on the reg z channel (metres); raise/lower to "
                        "rebalance vs offset/log-size terms")
    p.add_argument("--height-reg-weight", type=float, default=1.0,
                   help="L1 weight on the reg height channel (metres)")
    p.add_argument("--seg-weight", type=float, default=0.5)
    p.add_argument("--seg-dice-weight", type=float, default=0.5,
                   help="Dice term added to segmentation BCE; improves small-object masks")
    p.add_argument("--seg-tversky-weight", type=float, default=0.0,
                   help="Tversky term added to segmentation BCE; use >0 to penalise FP-heavy masks")
    p.add_argument("--seg-tversky-alpha", type=float, default=0.7,
                   help="Tversky false-positive weight")
    p.add_argument("--seg-tversky-beta", type=float, default=0.3,
                   help="Tversky false-negative weight")
    p.add_argument("--pseudo-seg-weight", type=float, default=1.0)
    p.add_argument("--channel-scale-jitter", type=float, default=0.08,
                   help="log-normal per-channel BEV scale jitter for LiDAR domain shift")
    p.add_argument("--channel-noise-std", type=float, default=0.01,
                   help="Gaussian noise added to normalized BEV channels during training")
    p.add_argument("--channel-dropout-prob", type=float, default=0.02,
                   help="probability of dropping a full BEV channel during training")
    p.add_argument("--channel-dropout-overrides", default="",
                   help="comma-separated per-channel dropout probabilities, e.g. "
                        "ground_shadow=0.25,shadow_edt=0.25. Values override the "
                        "global --channel-dropout-prob for matching channel names.")
    p.add_argument("--norm-mode",
                   choices=("legacy_tile_percentile", "physical", "train_stats"),
                   default="legacy_tile_percentile",
                   help="BEV input normalization contract. train_stats computes robust "
                        "per-channel train-set clipping limits and stores them in checkpoints.")
    p.add_argument("--norm-stats", default="",
                   help="optional JSON normalization contract to reuse instead of computing one")
    p.add_argument("--norm-stats-max-tiles", type=int, default=0,
                   help="limit tiles used to estimate train_stats; 0 = all training tiles")
    p.add_argument("--sources", default="all")
    p.add_argument("--dataset-dir", default=str(dataset_root()),
                   help="BEV .npz dataset directory; defaults to bordereye_lib/data/bev_dataset_v7")
    p.add_argument("--tile-include-file", default="",
                   help="optional text file with tile filenames/stems to keep for training")
    p.add_argument("--source-weights", default="")
    p.add_argument("--det-loss-weights", default="",
                   help="comma-separated detection loss weights by source, e.g. dales=1,yuto=0")
    p.add_argument("--seg-loss-weights", default="",
                   help="comma-separated segmentation loss weights by source, e.g. dales=0.75,yuto=1")
    p.add_argument("--seg-target", choices=("auto", "fp", "box", "rounded_box"), default="auto",
                   help="segmentation target selection: auto prefers seg_box_mask then fp_mask; "
                        "fp prefers fp_mask; box prefers stored seg_box_mask; rounded_box "
                        "renders a complete vehicle-like rounded rectangle from each OBB")
    p.add_argument("--weak-seg-hole-policy", choices=("none", "ignore", "soft"), default="none",
                   help="for weak sparse sources, handle footprint pixels with no original seg_mask "
                        "as normal positives, ignored pixels, or soft positives")
    p.add_argument("--weak-seg-hole-sources", default="yuto",
                   help="comma-separated sources where --weak-seg-hole-policy applies")
    p.add_argument("--weak-seg-hole-soft-target", type=float, default=0.35,
                   help="target value for weak footprint holes when --weak-seg-hole-policy=soft")
    p.add_argument("--weak-seg-hole-soft-weight", type=float, default=0.35,
                   help="loss weight for weak footprint holes when --weak-seg-hole-policy=soft")
    p.add_argument("--include-test-split", action="store_true")
    p.add_argument("--allow-no-det-val", action="store_true",
                   help="allow training when validation has no detection-supervised tile")
    p.add_argument("--load-retries", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                   help="enable CUDA automatic mixed precision (autocast + GradScaler)")

    # Segmentation-to-OBB consistency knobs.
    p.add_argument("--consistency-weight", type=float, default=0.5,
                   help="max weight of the seg<->OBB consistency loss (0 disables it)")
    p.add_argument("--consistency-warmup-epochs", type=int, default=8,
                   help="linearly ramp the consistency weight from 0 over this many epochs")
    p.add_argument("--teacher-ramp-mode", choices=("epoch", "plateau"), default="plateau",
                   help="epoch = monotonic warmup; plateau = back off teachers when validation degrades")
    p.add_argument("--consistency-topk", type=int, default=32)
    p.add_argument("--consistency-hm-thresh", type=float, default=0.3)
    p.add_argument("--consistency-sharp", type=float, default=2.0)

    # Orientation and geometry teacher knobs.
    p.add_argument("--orient-weight", type=float, default=1.0)
    p.add_argument("--orient-warmup-epochs", type=int, default=8)
    p.add_argument("--orient-win-radius", type=float, default=3.0)
    p.add_argument("--orient-min-mass", type=float, default=1.0)
    p.add_argument("--orient-min-aniso", type=float, default=0.05)
    p.add_argument("--moment-weight", type=float, default=0.25)
    p.add_argument("--moment-warmup-epochs", type=int, default=8)
    p.add_argument("--moment-win-radius", type=float, default=8.0)
    p.add_argument("--moment-min-mass", type=float, default=1.0)
    p.add_argument("--moment-min-aniso", type=float, default=0.03)
    p.add_argument("--moment-loss-cap", type=float, default=10.0)
    p.add_argument("--gaussian-reg-weight", type=float, default=0.25)

    # Architecture knobs.
    p.add_argument("--boundary-coordinate-augmentation", action="store_true",
                   help="use raster-boundary coordinates for box flips and rotations")
    p.add_argument("--heatmap-only-finetune", action="store_true",
                   help="WiSAR only: train class scores with fixed features and geometry")
    p.add_argument("--arch", choices=["v1", "v1_hr", "v1_hr_classreg", "v1_hr_boxdetail", "deep"], default="v1",
                   help="'v1' = BevNet; 'deep' = BevNetDeep with /8 bottleneck and GroupNorm")
    p.add_argument("--arch-base", type=int, default=64,
                   help="base channel count for --arch deep; 48 is a smaller BevNetDeep-style model")
    p.add_argument("--cue-channels", default="auto",
                   help="comma-separated channel names for the cue-prior heatmap branch. "
                        "'auto' uses the built-in defaults; 'none' disables the branch.")
    p.add_argument("--cbam-max", action="store_true")
    p.add_argument("--no-reg-gate", action="store_true")

    # Runtime and visualisation knobs.
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=5.0)
    # Cascade breaker: rollback-on-divergence guard (training/train_bev.py).
    p.add_argument("--divergence-factor", type=float, default=1.75,
                   help="trigger a rollback when train loss exceeds this multiple "
                        "of its recent 5-epoch median (0 disables the guard)")
    p.add_argument("--divergence-lr-scale", type=float, default=0.5,
                   help="permanent LR multiplier applied at each rollback")
    p.add_argument("--max-divergence-rollbacks", type=int, default=4,
                   help="stop training after this many rollbacks")
    p.add_argument("--nonfinite-skip-rollback-frac", type=float, default=0.25,
                   help="rollback when this fraction of train batches are skipped "
                        "for non-finite loss/gradients in one epoch; 0 disables")
    p.add_argument("--val-divergence-ratio", type=float, default=1.6,
                   help="slow-divergence trigger: roll back to BEST when the val "
                        "save metric exceeds this multiple of best-so-far while "
                        "stuck (0 disables)")
    p.add_argument("--slow-divergence-patience", type=int, default=5,
                   help="bad-epoch count required before the slow-divergence "
                        "check can fire")
    p.add_argument("--viz-latest-only", action="store_true",
                   help="write only the overwriting *_latest.png each epoch "
                        "instead of accumulating per-epoch numbered PNGs")
    p.add_argument("--viz-topn", type=int, default=8)
    p.add_argument("--viz-hm-thresh", type=float, default=0.08)

    # Experimental self-supervision knobs.
    p.add_argument("--z-height-weight", type=float, default=0.0)
    p.add_argument("--z-height-warmup-epochs", type=int, default=8)
    p.add_argument("--z-height-win-radius", type=float, default=8.0)
    p.add_argument("--z-height-min-mass", type=float, default=1.0)
    p.add_argument("--z-height-min-height", type=float, default=0.4)
    p.add_argument("--z-height-max-height", type=float, default=4.5)
    p.add_argument("--shadow-weight", type=float, default=0.0)
    p.add_argument("--shadow-warmup-epochs", type=int, default=8)
    p.add_argument("--shadow-win-radius", type=float, default=8.0)
    p.add_argument("--shadow-min-mass", type=float, default=1.0)
    p.add_argument("--shadow-no-seg-gate", action="store_true")
    p.add_argument("--shadow-precision-only", action="store_true")
    p.add_argument("--angle-loss-weight", type=float, default=1.0)
    p.add_argument("--yaw-geodesic-weight", type=float, default=0.0,
                   help="extra class-aware wrapped yaw loss; zero preserves legacy training")
    p.add_argument("--yaw-macro-average", action="store_true",
                   help="average yaw loss per present class before averaging classes")
    p.add_argument("--vertical-overlap-weight", type=float, default=0.0,
                   help="extra supervised z/height interval-IoU loss")
    p.add_argument("--vertical-macro-average", action="store_true",
                   help="average vertical overlap loss per present class")
    p.add_argument("--seg-explain-weight", type=float, default=0.0)
    p.add_argument("--seg-explain-warmup-epochs", type=int, default=20)
    p.add_argument("--seg-explain-max-components", type=int, default=24)
    p.add_argument("--seg-explain-min-area", type=float, default=4.0)
    p.add_argument("--seg-explain-max-area", type=float, default=260.0)
    p.add_argument("--seg-explain-thresh", type=float, default=0.4)
    p.add_argument("--seg-explain-min-aniso", type=float, default=0.05)
    p.add_argument("--mask-anchor-reliability-power", type=float, default=1.0)
    p.add_argument("--seg-explain-teacher", choices=("target", "pred"), default="target")
    # Vehicle-prior extensions for the mask anchors (core.mask_obb_teacher).
    p.add_argument("--mask-anchor-split", action=argparse.BooleanOptionalAction, default=False,
                   help="slice merged mask components (parking rows) into vehicle-length "
                        "anchors instead of dropping them via the max-area filter")
    p.add_argument("--prior-length", type=float, default=4.7,
                   help="vehicle length prior in metres (mask-anchor split + size prior)")
    p.add_argument("--prior-width", type=float, default=1.9,
                   help="vehicle width prior in metres (size prior)")
    p.add_argument("--prior-length-sigma", type=float, default=0.18,
                   help="log-space sigma of the length prior")
    p.add_argument("--prior-width-sigma", type=float, default=0.12,
                   help="log-space sigma of the width prior")
    p.add_argument("--seg-explain-size-prior-weight", type=float, default=0.0,
                   help="weight of the log-normal size prior inside the mask-anchor "
                        "OBB explanation loss (0 disables)")
    p.add_argument("--seg-center-weight", type=float, default=0.0)
    p.add_argument("--seg-center-warmup-epochs", type=int, default=30)
    p.add_argument("--seg-center-max-components", type=int, default=32)
    p.add_argument("--seg-center-min-area", type=float, default=4.0)
    p.add_argument("--seg-center-max-area", type=float, default=260.0)
    p.add_argument("--seg-center-thresh", type=float, default=0.4)
    p.add_argument("--seg-center-radius", type=float, default=1.5)
    p.add_argument("--seg-center-peak", type=float, default=0.75)
    p.add_argument("--seg-center-min-aniso", type=float, default=0.05)
    p.add_argument("--seg-center-teacher", choices=("target", "pred"), default="target")
    p.add_argument("--full-coverage", action="store_true",
                   help="guarantee every TRAIN tile is visited at least once per "
                        "epoch (source weights fill the remaining crops). Without "
                        "it the block sampler trains a fixed ~crops/tile_block "
                        "subset every epoch and never sees the rest.")
    p.add_argument("--use-channels", default="default",
                   help="comma-separated channel names to train on (a subset of "
                        "the dataset schema); the checkpoint records the subset "
                        "so eval selects the same channels. 'default' = the "
                        "canonical 12-channel model subset "
                        "(bordereye GENERIC_V2_MODEL_CHANNELS_12) when the "
                        "dataset schema contains it, otherwise all stored "
                        "channels. 'all' = all stored channels.")
    return p


def parse_args(description: str | None = None, argv=None):
    """Parse CLI arguments with optional JSON preset defaults."""
    parser = build_parser(description)
    pre_args, _ = parser.parse_known_args(argv)
    if pre_args.config:
        _apply_config_defaults(parser, pre_args.config)
    return parser.parse_args(argv)
