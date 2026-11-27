"""Shared BEV training data and loss helpers.

This module owns crop sampling, target encoding, source weighting, and the
small batch loss wrappers used by the main consistency trainer. It is separated
from runnable trainer scripts so training code can import stable library APIs.
"""
from __future__ import annotations

import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from torch.utils.data import Dataset, get_worker_info

from .footprint_targets import oriented_rounded_rect_mask
from .losses import (
    focal_loss_per_sample,
    reg_l1_per_sample,
    weighted_seg_bce_per_sample,
    weighted_seg_dice_per_sample,
    weighted_seg_tversky_per_sample,
)
from .nn_contract import REG_DIM, STRIDE, channel_scale
from .normalization import normalize_bev_channels
from .orientation import use_doubled_yaw

# WiSAR3D stores the official annotation tuple in a different order from the
# sensor-independent BEV contract used everywhere else in this package.  Keep
# the conversion at the input boundary: cached WiSAR NPZ files remain faithful
# to the released dataset while all crop/augmentation/target code sees exactly
# one canonical format.
#
# raw WiSAR: [cx, cy, z, length, width, height, yaw_rad]
# canonical: [cx, cy, length, width, yaw_deg, z, height]
WISAR_RAW_BOX_SCHEMA = "wisar3d_xy_z_lwh_yawrad_v1"
CANONICAL_BOX_SCHEMA = "bev_xy_lw_yawdeg_zh_v1"


def canonicalize_wisar_boxes(boxes: np.ndarray) -> np.ndarray:
    """Convert released WiSAR3D boxes to the shared BEV OBB contract.

    This deliberately does not mutate the source array.  The function is
    public and small so dataset builders, visualisers and tests can use the
    same unambiguous conversion instead of duplicating fragile column indices.
    """
    raw = np.asarray(boxes, np.float32).reshape(-1, 7)
    if not len(raw):
        return raw.copy()
    out = raw[:, [0, 1, 3, 4, 6, 2, 5]].copy()
    out[:, 4] = np.rad2deg(out[:, 4])
    return out


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def densify_seg_mask(mask: np.ndarray, cell: float = 0.20,
                     gap_m: float = 0.40, sever_m: float = 0.0) -> np.ndarray:
    """Make a sparse vehicle point mask solid without merging vehicles.

    Measured on the real ECLAIR/DALES seg_masks, pure binary_fill_holes adds
    ~0% because the in-vehicle black pixels are open scan-line gaps, not enclosed
    holes. So a minimal closing first bridges those gaps (gap_m, ~0.4 m == 1 px
    at 0.2 m doubles the mask and stitches the fragments of one vehicle), then
    fill_holes solidifies the interior.

    gap_m is deliberately small: the closing only merges two distinct vehicles
    when they are parked closer than ~gap_m, which is rare and unavoidable
    without instance/box information. gap_m=0 falls back to fill_holes only.
    sever_m>0 opens thin necks afterwards to cut accidental bridges.
    The already-solid OBB seg_box_mask is not passed through here.
    """
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return np.zeros(m.shape, np.uint8)
    out = m
    if gap_m > 0.0:
        r = max(1, round(float(gap_m) / float(cell)))
        out = ndi.binary_closing(out, structure=_disk(r), border_value=0)
    out = ndi.binary_fill_holes(out)
    if sever_m > 0.0:
        rs = max(1, round(float(sever_m) / float(cell)))
        out = ndi.binary_opening(out, structure=_disk(rs), border_value=0)
    return out.astype(np.uint8)


def gaussian2d(radius: int, sigma: float) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    return np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, radius: int,
                  peak: float = 1.0) -> None:
    """Draw a 2D Gaussian onto hm (in-place max-blend).

    peak controls the maximum value; 1.0 = confident GT, 0.5 = pseudo-label.
    Values are blended with np.maximum so a confident box (peak=1) always wins
    over a pseudo box (peak=0.5) at the same location.
    """
    radius = max(1, radius)
    g = gaussian2d(radius, max(radius / 3.0, 1e-6))
    if peak != 1.0:
        g = g * peak
    H, W = hm.shape
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    gx0 = x0 - (cx - radius)
    gy0 = y0 - (cy - radius)
    sub = hm[y0:y1, x0:x1]
    gsub = g[gy0:gy0 + (y1 - y0), gx0:gx0 + (x1 - x0)]
    np.maximum(sub, gsub, out=sub)


# Augmentation helpers.

def _rotate_px(lc: float, lr: float, thd: float, C: int, k: int, *, boundary_coordinates=False):
    """Rotate pixel coords + OBB angle by k*90 degrees CCW."""
    for _ in range(k % 4):
        lc, lr = lr, C - (0 if boundary_coordinates else 1) - lc
        thd += 90.0
    return lc, lr, thd % 360.0


def _rotate_ignore_corners(cols: np.ndarray, rows: np.ndarray, C: int, k: int, *, boundary_coordinates=False):
    """Apply k*90 CCW rotation to corner pixel coordinates."""
    for _ in range(k % 4):
        cols, rows = rows.copy(), C - (0 if boundary_coordinates else 1) - cols
    return cols, rows


def _draw_seg_box(seg: np.ndarray, weight_map: np.ndarray,
                  cx: float, cy: float, length: float, width: float,
                  theta: float, weight: float) -> None:
    """Rasterise an oriented box into stride-space segmentation target maps."""
    inside = oriented_rounded_rect_mask(
        seg.shape, cx, cy, length, width, theta,
        corner_radius_fraction=0.25)
    if inside.any():
        already_positive = inside & (seg > 0)
        new_positive = inside & ~already_positive
        weight_map[new_positive] = float(weight)
        weight_map[already_positive] = np.maximum(
            weight_map[already_positive], float(weight))
        seg[inside] = 1.0


# Dataset.

class BevCrops(Dataset):
    def __init__(self, npz_paths, cell, crop, n_crops, pos_frac,
                 seed=0, augment=True, cache_tiles=3, pseudo_peak=0.5,
                 load_retries=3, sample_weights=None, pseudo_reg_weight=0.25,
                 pseudo_seg_weight=1.0, det_loss_weights=None, seg_loss_weights=None,
                 channel_scale_jitter=0.0, channel_noise_std=0.0,
                 channel_dropout_prob=0.0, channel_dropout_probs=None,
                 deterministic=False,
                 norm_config=None,
                 seg_target="auto",
                 weak_seg_hole_policy="none", weak_seg_hole_sources=None,
                 weak_seg_hole_soft_target=0.35, weak_seg_hole_soft_weight=0.35,
                 densify_seg=True, densify_seg_gap=0.40, densify_seg_sever=0.0,
                 hm_radius_min=1, hm_radius_max=2, hm_radius_scale=0.25,
                 tile_block=8, tile_interleave=4, channel_subset=None,
                 full_coverage=False, stored_channel_names=None, num_classes=1,
                 output_stride=STRIDE, class_sampling_weights=None,
                 copy_paste_prob=0.0, copy_paste_max=0, copy_paste_classes=None,
                 reg_support_radius=0, reg_support_max_size_m=0.0,
                 boundary_coordinate_augmentation=False):
        # channel_subset: indices into the stored channel stack; lets a model
        # train on a named subset of a wider dataset schema without rebuilding
        # the dataset (teachers still see only the selected channels).
        self.channel_subset = (tuple(int(i) for i in channel_subset)
                               if channel_subset else None)
        self.stored_channel_names = (tuple(str(n) for n in stored_channel_names)
                                     if stored_channel_names else None)
        self.selected_channel_names = (
            tuple(self.stored_channel_names[i] for i in self.channel_subset)
            if self.stored_channel_names is not None and self.channel_subset is not None
            else None)
        # full_coverage: build a per-epoch tile plan that visits EVERY tile at
        # least once (source weights fill the remaining crops). Without it the
        # block sampler's per-index hash is fixed across epochs, so the same
        # ~crops/tile_block tiles train every epoch and the rest never do.
        self.full_coverage = bool(full_coverage)
        self._epoch = 0
        self._plan = None
        self.paths = list(npz_paths)
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")
        self.cell = cell
        self.crop = crop
        self.output_stride = int(output_stride)
        if self.output_stride not in (1, 2, 4) or self.crop % self.output_stride:
            raise ValueError(
                f"output_stride must divide crop and be one of 1,2,4; got "
                f"stride={self.output_stride}, crop={self.crop}")
        self.class_sampling_weights = None
        if class_sampling_weights is not None:
            weights = np.asarray(class_sampling_weights, np.float64).reshape(-1)
            if len(weights) != self.num_classes or not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError(
                    f"class_sampling_weights must contain {self.num_classes} positive finite values")
            self.class_sampling_weights = weights
        self.copy_paste_prob = float(np.clip(copy_paste_prob, 0.0, 1.0))
        self.copy_paste_max = max(0, int(copy_paste_max))
        self.copy_paste_classes = set(int(value) for value in (copy_paste_classes or ()))
        if any(value < 0 or value >= self.num_classes for value in self.copy_paste_classes):
            raise ValueError("copy_paste_classes contains an invalid class id")
        # Radius zero is the historical single-centre target used by all
        # retained vehicle models. Dense support is opt-in for WiSAR's tiny
        # objects and does not change the model/checkpoint contract.
        self.reg_support_radius = max(0, int(reg_support_radius))
        self.boundary_coordinate_augmentation = bool(boundary_coordinate_augmentation)
        self.reg_support_max_size_m = max(0.0, float(reg_support_max_size_m))
        requested_crops = int(n_crops)
        # Full coverage is a hard per-epoch contract, not a best effort. An
        # epoch shorter than the tile list cannot visit every tile, therefore
        # expand it rather than silently rotating coverage across epochs.
        self.n = max(requested_crops, len(self.paths)) if self.full_coverage else requested_crops
        self.pos_frac = pos_frac
        self.augment = augment
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.deterministic = bool(deterministic)
        self.cache_tiles = max(1, int(cache_tiles))
        self.pseudo_peak = float(pseudo_peak)
        self.pseudo_reg_weight = float(pseudo_reg_weight)
        self.pseudo_seg_weight = float(pseudo_seg_weight)
        self.det_loss_weights = dict(det_loss_weights or {})
        self.seg_loss_weights = dict(seg_loss_weights or {})
        self.channel_scale_jitter = max(0.0, float(channel_scale_jitter))
        self.channel_noise_std = max(0.0, float(channel_noise_std))
        self.channel_dropout_prob = max(0.0, min(1.0, float(channel_dropout_prob)))
        if channel_dropout_probs is None:
            self.channel_dropout_probs = None
        else:
            probs = np.asarray(channel_dropout_probs, np.float32).reshape(-1)
            self.channel_dropout_probs = np.clip(probs, 0.0, 1.0)
        self._channel_augmentation_active = bool(
            self.channel_scale_jitter > 0
            or self.channel_noise_std > 0
            or self.channel_dropout_prob > 0
            or (self.channel_dropout_probs is not None
                and np.any(self.channel_dropout_probs > 0))
        )
        self.norm_config = dict(norm_config or {"mode": "legacy_tile_percentile"})
        if seg_target not in ("auto", "fp", "box", "rounded_box"):
            raise ValueError(f"unknown seg_target={seg_target!r}")
        self.seg_target = seg_target
        if weak_seg_hole_policy not in ("none", "ignore", "soft"):
            raise ValueError(f"unknown weak_seg_hole_policy={weak_seg_hole_policy!r}")
        self.weak_seg_hole_policy = weak_seg_hole_policy
        self.weak_seg_hole_sources = set(weak_seg_hole_sources or [])
        self.weak_seg_hole_soft_target = float(np.clip(weak_seg_hole_soft_target, 0.0, 1.0))
        self.weak_seg_hole_soft_weight = float(np.clip(weak_seg_hole_soft_weight, 0.0, 1.0))
        self.load_retries = max(1, int(load_retries))
        self.densify_seg = bool(densify_seg)
        self.densify_seg_gap = max(0.0, float(densify_seg_gap))
        self.densify_seg_sever = max(0.0, float(densify_seg_sever))
        # Heatmap Gaussian radius (stride-grid cells). Tunable so the previously
        # fixed [1,2] range can be ablated for recall/instance-split trade-offs.
        self.hm_radius_min = max(1, int(hm_radius_min))
        self.hm_radius_max = max(self.hm_radius_min, int(hm_radius_max))
        self.hm_radius_scale = max(0.0, float(hm_radius_scale))
        # Cache-friendly tile sampling (see _choose_tile_index). tile_block is
        # the number of crops each loaded tile serves; tile_interleave is how
        # many distinct tiles alternate inside one super-block (>=2 keeps
        # batches mixed even when batch size exceeds tile_block).
        self.tile_block = max(1, int(tile_block))
        self.tile_interleave = max(1, int(tile_interleave))
        self.sample_weights = None
        if sample_weights is not None:
            w = np.asarray(sample_weights, np.float64)
            if w.shape[0] != len(self.paths):
                raise ValueError("sample_weights length must match npz_paths")
            if np.isfinite(w).all() and w.sum() > 0:
                self.sample_weights = w / w.sum()
        self._cache: OrderedDict = OrderedDict()

    def __len__(self):
        return self.n

    def _load(self, ti):
        if ti in self._cache:
            item = self._cache.pop(ti)
            self._cache[ti] = item
            return item
        last_exc = None
        for attempt in range(max(1, getattr(self, "load_retries", 3))):
            try:
                with np.load(self.paths[ti]) as d:
                    tile_names = tuple(str(x) for x in d["channel_names"].tolist()) \
                        if "channel_names" in d else ()
                    # A long-running Windows DataLoader may pickle a Dataset
                    # created before a hot code update, then import the newer
                    # class body in spawned workers. Keep new attributes
                    # backward compatible so such workers fail only on an
                    # actual schema mismatch, never on AttributeError.
                    stored_names = getattr(self, "stored_channel_names", None)
                    selected_names = getattr(self, "selected_channel_names", None)
                    if stored_names is not None and tile_names != stored_names:
                        raise ValueError(
                            f"BEV channel schema changed after startup for {self.paths[ti]}: "
                            f"expected {list(stored_names)}, got {list(tile_names)}")
                    ch = normalize_bev_channels(d["channels"], channel_scale(d), self.norm_config)
                    if self.channel_subset is not None:
                        # Resolve by name at load time. The equality check above
                        # turns a stale tile or schema drift into a precise
                        # error instead of an out-of-range or wrong-channel read.
                        if selected_names is not None:
                            idx = [tile_names.index(n) for n in selected_names]
                        else:
                            idx = list(self.channel_subset)
                        ch = ch[idx]
                    origin = d["origin"].astype(np.float64)
                    source = str(d["source"]) if "source" in d else "unknown"
                    boxes = (d["boxes_valid"] if "boxes_valid" in d else d["boxes"]).astype(np.float32)
                    box_classes = (d["box_classes"].astype(np.int64)
                                   if "box_classes" in d else np.zeros(len(boxes), np.int64))
                    if len(box_classes) != len(boxes):
                        raise ValueError(
                            f"box_classes length {len(box_classes)} != boxes {len(boxes)} "
                            f"for {self.paths[ti]}")
                    if len(box_classes) and (
                            box_classes.min() < 0 or box_classes.max() >= self.num_classes):
                        raise ValueError(
                            f"box class outside [0,{self.num_classes}) in {self.paths[ti]}")
                    pseudo_boxes = (d["boxes_pseudo"] if "boxes_pseudo" in d
                                    else np.zeros((0, 7), np.float32)).astype(np.float32)
                    # Optional per-box teacher quality in [0,1] (written by
                    # generated by data.build_dataset); scales the pseudo
                    # regression weight so uncertain fits supervise softly.
                    if ("boxes_pseudo_score" in d
                            and len(d["boxes_pseudo_score"]) == len(pseudo_boxes)):
                        pseudo_scores = np.clip(
                            d["boxes_pseudo_score"].astype(np.float32), 0.0, 1.0)
                    else:
                        pseudo_scores = np.ones(len(pseudo_boxes), np.float32)
                    ignore_boxes = (d["boxes_ignore"] if "boxes_ignore" in d
                                    else np.zeros((0, 7), np.float32)).astype(np.float32)
                    # WiSAR annotations intentionally stay in their released
                    # raw schema on disk.  Convert all OBB collections once at
                    # the ingestion boundary so cropping, copy-paste, masks and
                    # regression targets share the documented canonical order.
                    if source.lower() == "wisar":
                        boxes = canonicalize_wisar_boxes(boxes)
                        pseudo_boxes = canonicalize_wisar_boxes(pseudo_boxes)
                        ignore_boxes = canonicalize_wisar_boxes(ignore_boxes)
                    has_boxes_for_seg = len(boxes) > 0 or len(pseudo_boxes) > 0
                    seg_mask = None
                    # NPZ members may be compressed. Cache each mask after its
                    # first read so an existence probe cannot decompress the
                    # same dense array again when it is selected.

                    def _nonempty_mask(key: str):
                        if key not in d:
                            return None
                        value = np.asarray(d[key], dtype=np.uint8)
                        return value if value.any() else None

                    box_mask = (_nonempty_mask("seg_box_mask")
                                if self.seg_target in ("box", "auto") else None)
                    fp_mask = None
                    raw_seg_mask = None
                    if self.seg_target == "rounded_box" and has_boxes_for_seg:
                        # Leave seg_mask unset: __getitem__ renders the canonical
                        # rounded footprint after crop augmentation, one OBB at
                        # a time, so geometry and target stay exactly aligned.
                        seg_mask = None
                    elif self.seg_target in ("box", "auto") and box_mask is not None:
                        # DALES/synth: OBB-derived footprint unless overridden.
                        seg_mask = box_mask
                    else:
                        fp_mask = _nonempty_mask("fp_mask")
                        if fp_mask is not None:
                            # shadow-close (DALES-LAS) or seg-close
                            # (ECLAIR/YUTO/ISPRS)
                            seg_mask = fp_mask
                        else:
                            raw_seg_mask = _nonempty_mask("seg_mask")
                            if raw_seg_mask is not None:
                                # Legacy tiles may have only point-level masks.
                                seg_mask = raw_seg_mask
                                if self.densify_seg:
                                    seg_mask = densify_seg_mask(
                                        seg_mask, cell=float(self.cell),
                                        gap_m=self.densify_seg_gap,
                                        sever_m=self.densify_seg_sever)
                            elif has_boxes_for_seg and self.seg_target != "rounded_box":
                                import warnings
                                warnings.warn(
                                    f"Tile {self.paths[ti].name}: has OBBs but fp_mask is "
                                    "empty/missing - seg supervision dropped for this tile.",
                                    stacklevel=2)
                    weak_hole_mask = None
                    if (self.weak_seg_hole_policy != "none"
                            and source in self.weak_seg_hole_sources
                            and seg_mask is not None):
                        if raw_seg_mask is None:
                            raw_seg_mask = _nonempty_mask("seg_mask")
                        if raw_seg_mask is not None:
                            weak_hole_mask = (
                                seg_mask.astype(bool) & ~raw_seg_mask.astype(bool)
                            ).astype(np.uint8)
                    H, _W = ch.shape[1:]
                break
            except (zipfile.BadZipFile, EOFError, OSError) as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed to load NPZ after retries: {self.paths[ti]}") from last_exc

        def _to_px(b):
            if not len(b):
                return np.zeros((0, 2), np.float32)
            col = (b[:, 0] - origin[0]) / self.cell
            row = H - (b[:, 1] - origin[1]) / self.cell
            return np.column_stack([col, row]).astype(np.float32)

        bpx        = _to_px(boxes)
        bpx_pseudo = _to_px(pseudo_boxes)

        has_valid_boxes = len(boxes) > 0
        has_pseudo_boxes = len(pseudo_boxes) > 0
        item = (ch, bpx, boxes, box_classes, bpx_pseudo, pseudo_boxes, pseudo_scores,
                ignore_boxes, origin,
                seg_mask, weak_hole_mask, source, has_valid_boxes, has_pseudo_boxes)
        self._cache[ti] = item
        while len(self._cache) > self.cache_tiles:
            self._cache.popitem(last=False)
        return item

    def _rng_for_index(self, index: int):
        """Return a deterministic per-index RNG or the dataset stream RNG."""
        if self.deterministic:
            return np.random.default_rng(self.seed + int(index))
        return self.rng

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch and rebuild the full-coverage tile plan.

        Call before each epoch's DataLoader iteration. With persistent workers
        disabled (see the training loop), re-forked workers pick up the new
        plan; the plan is a deterministic function of (seed, epoch) so all
        workers agree on it.
        """
        self._epoch = int(epoch)
        if self.full_coverage:
            self._plan = self._build_coverage_plan()

    def _build_coverage_plan(self) -> np.ndarray:
        n = len(self.paths)
        rng = np.random.default_rng((self.seed, 0xC0FFEE, self._epoch))
        if self.n >= n:
            base = np.arange(n)
            n_extra = self.n - n
            if n_extra <= 0:
                extra = np.empty(0, np.int64)
            elif self.sample_weights is not None:
                # Each tile already got one base crop. Sample the extras with
                # weights compensated for that base so the FINAL marginal stays
                # close to sample_weights: desired_i = w_i*N, extra_i ∝
                # max(0, desired_i - 1). High-tile-count/low-weight sources
                # (already fully covered by the base) get few or no extras.
                desired = self.sample_weights * float(self.n)
                comp = np.clip(desired - 1.0, 0.0, None)
                if comp.sum() <= 0:
                    extra = rng.integers(0, n, size=n_extra)
                else:
                    extra = rng.choice(n, size=n_extra, p=comp / comp.sum())
            else:
                extra = rng.integers(0, n, size=n_extra)
            plan = np.concatenate([base, extra])
        else:
            # fewer crops than tiles: rotate a fixed shuffle across epochs so
            # every tile is covered over ceil(n / crops) consecutive epochs.
            full = np.arange(n)
            np.random.default_rng((self.seed, 0xC0FFEE)).shuffle(full)
            start = (self._epoch * self.n) % n
            plan = np.take(full, np.arange(start, start + self.n), mode="wrap")
        rng.shuffle(plan)
        return plan.astype(np.int64)

    def _choose_tile_index(self, rng, index: int) -> int:
        """Sample one tile index with cache-friendly block locality.

        A fully random tile per crop defeats the small LRU tile cache: nearly
        every crop pays a full NPZ decompress + normalize, which dominates the
        epoch time on multi-tile datasets. Instead, consecutive indices cycle
        through ``tile_interleave`` tiles drawn once per super-block of
        ``tile_block * tile_interleave`` indices, so each loaded tile serves
        about ``tile_block`` crops while a batch still mixes different tiles.
        The source-balancing weights apply to the per-super-block draw, so the
        marginal tile distribution is unchanged. ``tile_block=1`` restores the
        original fully random behavior.
        """
        n = len(self.paths)
        if self.full_coverage:
            if self._plan is None:
                self._plan = self._build_coverage_plan()
            return int(self._plan[int(index) % len(self._plan)])
        if self.tile_block <= 1 or n <= 1:
            if self.sample_weights is not None:
                return int(rng.choice(n, p=self.sample_weights))
            return int(rng.integers(n))
        group = max(1, min(self.tile_interleave, self.cache_tiles))
        super_len = group * self.tile_block
        block_rng = np.random.default_rng((self.seed, 0x9E3779B9, int(index) // super_len))
        if self.sample_weights is not None:
            tiles = block_rng.choice(n, size=group, p=self.sample_weights)
        else:
            tiles = block_rng.integers(0, n, size=group)
        return int(tiles[int(index) % group])

    def _source_loss_weights(self, source: str, has_valid_boxes: bool,
                             has_pseudo_boxes: bool) -> tuple[float, float]:
        """Return detection and segmentation loss weights for one tile source."""
        det_loss_weight = source_weight_lookup(self.det_loss_weights, source, 1.0)
        seg_loss_weight = source_weight_lookup(self.seg_loss_weights, source, 1.0)
        if det_loss_weight > 0 and not has_valid_boxes and not has_pseudo_boxes:
            det_loss_weight = 0.0
        return det_loss_weight, seg_loss_weight

    def _choose_crop_origin(self, rng, bpx: np.ndarray, box_classes: np.ndarray,
                            bpx_pseudo: np.ndarray,
                            H: int, W: int) -> tuple[int, int]:
        """Choose the upper-left crop origin, biased toward known/pseudo boxes."""
        C = self.crop
        all_bpx = bpx if len(bpx) else bpx_pseudo
        if len(all_bpx) and rng.random() < self.pos_frac:
            if len(bpx) and self.class_sampling_weights is not None:
                probabilities = self.class_sampling_weights[box_classes]
                probabilities = probabilities / probabilities.sum()
                k = int(rng.choice(len(bpx), p=probabilities))
            else:
                k = int(rng.integers(len(all_bpx)))
            cc, cr = all_bpx[k]
            max_c0 = max(W - C, 0)
            max_r0 = max(H - C, 0)
            c0 = int(np.clip(cc - C / 2 + rng.integers(-C // 4, C // 4), 0, max_c0))
            r0 = int(np.clip(cr - C / 2 + rng.integers(-C // 4, C // 4), 0, max_r0))
        else:
            c0 = int(rng.integers(0, max(W - C, 1)))
            r0 = int(rng.integers(0, max(H - C, 1)))
        return c0, r0

    def _crop_tile(self, ch: np.ndarray, seg_mask_full: np.ndarray | None,
                   c0: int, r0: int,
                   weak_hole_full: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Crop channel and segmentation tensors, padding edge crops to square."""
        C = self.crop
        n_ch = ch.shape[0]
        crop = ch[:, r0:r0 + C, c0:c0 + C]
        seg_crop = None
        weak_hole_crop = None
        if seg_mask_full is not None:
            seg_crop = seg_mask_full[r0:r0 + C, c0:c0 + C].astype(np.float32)
        if weak_hole_full is not None:
            weak_hole_crop = weak_hole_full[r0:r0 + C, c0:c0 + C].astype(np.float32)
        if crop.shape[1:] != (C, C):
            pad = np.zeros((n_ch, C, C), np.float32)
            pad[:, :crop.shape[1], :crop.shape[2]] = crop
            crop = pad
            if seg_crop is not None:
                seg_pad = np.zeros((C, C), np.float32)
                seg_pad[:seg_crop.shape[0], :seg_crop.shape[1]] = seg_crop
                seg_crop = seg_pad
            if weak_hole_crop is not None:
                hole_pad = np.zeros((C, C), np.float32)
                hole_pad[:weak_hole_crop.shape[0], :weak_hole_crop.shape[1]] = weak_hole_crop
                weak_hole_crop = hole_pad
        return crop, seg_crop, weak_hole_crop

    def _copy_paste_objects(self, crop, bpx, boxes, box_classes, c0, r0,
                            origin, full_height, rng):
        """Duplicate weak-class geometry inside a crop without copying terrain.

        Only above-ground/density/shape channels are pasted. A density support
        mask prevents the square source patch from erasing the destination
        background, while the duplicated canonical box supplies exact targets.
        """
        if (not self.augment or self.copy_paste_max <= 0
                or rng.random() >= self.copy_paste_prob or not len(boxes)):
            return crop, bpx, boxes, box_classes
        local = bpx - np.asarray([c0, r0], np.float32)
        candidates = [
            i for i, (xy, class_id) in enumerate(zip(local, box_classes, strict=True))
            if int(class_id) in self.copy_paste_classes
            and 0 <= xy[0] < self.crop and 0 <= xy[1] < self.crop
        ]
        if not candidates:
            return crop, bpx, boxes, box_classes
        names = self.selected_channel_names or self.stored_channel_names or ()
        paste_indices = [
            i for i, name in enumerate(names)
            if (name.startswith("veh_") or name.startswith("all_")
                or name in {"occ", "log_count", "hag_max", "hag_mean", "hag_std"})
        ]
        if not paste_indices:
            paste_indices = list(range(crop.shape[0]))
        density_candidates = [
            i for i, name in enumerate(names)
            if name in {"all_density", "veh_density", "occ", "occ_all"}
        ]
        density_index = density_candidates[0] if density_candidates else paste_indices[0]
        out_boxes = list(np.asarray(boxes, np.float32))
        out_bpx = list(np.asarray(bpx, np.float32))
        out_classes = list(np.asarray(box_classes, np.int64))
        pasted = 0
        for source_index in rng.permutation(candidates):
            if pasted >= self.copy_paste_max:
                break
            sx, sy = local[source_index]
            box = boxes[source_index]
            radius = max(3, int(np.ceil(max(float(box[2]), float(box[3])) / self.cell * 0.75)))
            sx0, sy0 = int(round(sx)) - radius, int(round(sy)) - radius
            sx1, sy1 = sx0 + 2 * radius + 1, sy0 + 2 * radius + 1
            if sx0 < 0 or sy0 < 0 or sx1 > self.crop or sy1 > self.crop:
                continue
            for _ in range(12):
                tx = int(rng.integers(radius, self.crop - radius))
                ty = int(rng.integers(radius, self.crop - radius))
                min_sep = max(4.0, radius * 1.5)
                if all(np.hypot(tx - xy[0], ty - xy[1]) >= min_sep for xy in local):
                    break
            else:
                continue
            tx0, ty0 = tx - radius, ty - radius
            tx1, ty1 = tx0 + 2 * radius + 1, ty0 + 2 * radius + 1
            support = crop[density_index, sy0:sy1, sx0:sx1] > 0
            if not support.any():
                support = np.ones((2 * radius + 1, 2 * radius + 1), bool)
            for channel_index in paste_indices:
                source = crop[channel_index, sy0:sy1, sx0:sx1]
                target = crop[channel_index, ty0:ty1, tx0:tx1]
                target[support] = np.maximum(target[support], source[support])
            duplicate = np.asarray(box, np.float32).copy()
            duplicate[0] = origin[0] + (c0 + tx) * self.cell
            duplicate[1] = origin[1] + (full_height - (r0 + ty)) * self.cell
            out_boxes.append(duplicate)
            out_bpx.append(np.asarray([c0 + tx, r0 + ty], np.float32))
            out_classes.append(int(box_classes[source_index]))
            local = np.vstack((local, [tx, ty]))
            pasted += 1
        return (crop, np.asarray(out_bpx, np.float32),
                np.asarray(out_boxes, np.float32), np.asarray(out_classes, np.int64))

    def __getitem__(self, index):
        rng = self._rng_for_index(index)
        ti = self._choose_tile_index(rng, index)
        (ch, bpx, boxes, box_classes, bpx_pseudo, pseudo_boxes, pseudo_scores,
         ignore_boxes, origin,
         seg_mask_full, weak_hole_full, source, has_valid_boxes, has_pseudo_boxes) = self._load(ti)
        det_loss_weight, seg_loss_weight = self._source_loss_weights(
            source, has_valid_boxes, has_pseudo_boxes)
        H, W = ch.shape[1:]
        C = self.crop

        c0, r0 = self._choose_crop_origin(rng, bpx, box_classes, bpx_pseudo, H, W)
        # World-to-pixel maps continuous coordinates from the raster boundary.
        # Flipping [0,C] maps x -> C-x, unlike integer indices x -> C-1-x.
        flip_extent = C if self.boundary_coordinate_augmentation else C - 1
        crop, seg_crop, weak_hole_crop = self._crop_tile(ch, seg_mask_full, c0, r0, weak_hole_full)
        crop, bpx, boxes, box_classes = self._copy_paste_objects(
            crop, bpx, boxes, box_classes, c0, r0, origin, H, rng)

        # --- augmentation: flip (fx/fy) then 90-degree rotation (rot_k) ---
        fx = self.augment and bool(rng.integers(2))
        fy = self.augment and bool(rng.integers(2))
        rot_k = int(rng.integers(4)) if self.augment else 0

        if fx:
            crop = crop[:, :, ::-1]
            if seg_crop is not None:
                seg_crop = seg_crop[:, ::-1]
            if weak_hole_crop is not None:
                weak_hole_crop = weak_hole_crop[:, ::-1]
        if fy:
            crop = crop[:, ::-1, :]
            if seg_crop is not None:
                seg_crop = seg_crop[::-1, :]
            if weak_hole_crop is not None:
                weak_hole_crop = weak_hole_crop[::-1, :]
        if rot_k:
            crop = np.rot90(crop, k=rot_k, axes=(1, 2))
            if seg_crop is not None:
                seg_crop = np.rot90(seg_crop, k=rot_k, axes=(0, 1))
            if weak_hole_crop is not None:
                weak_hole_crop = np.rot90(weak_hole_crop, k=rot_k, axes=(0, 1))
        crop = np.ascontiguousarray(crop)
        if seg_crop is not None:
            seg_crop = np.ascontiguousarray(seg_crop)
        if weak_hole_crop is not None:
            weak_hole_crop = np.ascontiguousarray(weak_hole_crop)
        if self.augment:
            crop = self._augment_channels(crop, rng)

        stride = self.output_stride
        OS = C // stride
        hm = np.zeros((self.num_classes, OS, OS), np.float32)
        reg = np.zeros((REG_DIM, OS, OS), np.float32)
        mask = np.zeros((1, OS, OS), np.float32)
        ignore = np.zeros((1, OS, OS), np.float32)
        seg = np.zeros((1, OS, OS), np.float32)
        seg_weight = np.ones((1, OS, OS), np.float32)
        reg_owner = (np.full((OS, OS), -np.inf, np.float32)
                     if self.reg_support_radius > 0 else None)
        reg_owner_class = (np.full((OS, OS), -1, np.int16)
                           if self.reg_support_radius > 0 else None)
        has_point_seg = seg_crop is not None
        if has_point_seg:
            seg[0] = seg_crop.reshape(OS, stride, OS, stride).max(axis=(1, 3))
            if weak_hole_crop is not None:
                hole = weak_hole_crop.reshape(OS, stride, OS, stride).max(axis=(1, 3)) > 0
                raw_pos = (seg_crop > 0) & (weak_hole_crop <= 0)
                raw_pos_os = raw_pos.reshape(OS, stride, OS, stride).max(axis=(1, 3))
                hole &= ~raw_pos_os
                if self.weak_seg_hole_policy == "ignore":
                    seg_weight[0, hole] = 0.0
                elif self.weak_seg_hole_policy == "soft":
                    seg[0, hole] = self.weak_seg_hole_soft_target
                    seg_weight[0, hole] *= self.weak_seg_hole_soft_weight

        # Mark ignore boxes in the output map.
        for ib in ignore_boxes:
            cx_w, cy_w, L, Wd, thd_w = ib[0], ib[1], ib[2], ib[3], float(ib[4])
            th = np.deg2rad(thd_w)
            ca, sa = np.cos(th), np.sin(th)
            hl, hw = L / 2, Wd / 2
            local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], np.float32)
            verts = local @ np.array([[ca, -sa], [sa, ca]], np.float32).T + np.array([cx_w, cy_w])
            cols = (verts[:, 0] - origin[0]) / self.cell - c0
            rows = H - (verts[:, 1] - origin[1]) / self.cell - r0
            if fx:
                cols = flip_extent - cols
            if fy:
                rows = flip_extent - rows
            if rot_k:
                cols, rows = _rotate_ignore_corners(cols, rows, C, rot_k,
                    boundary_coordinates=self.boundary_coordinate_augmentation)
            x0i = int(np.floor(np.clip(cols.min() / stride, 0, OS - 1)))
            x1i = int(np.ceil(np.clip(cols.max() / stride, 0, OS - 1)))
            y0i = int(np.floor(np.clip(rows.min() / stride, 0, OS - 1)))
            y1i = int(np.ceil(np.clip(rows.max() / stride, 0, OS - 1)))
            if x1i >= x0i and y1i >= y0i:
                ignore[0, y0i:y1i + 1, x0i:x1i + 1] = 1.0

        def _encode_box(b_row, b_col_f, b_row_f, peak: float, quality: float = 1.0,
                        class_id: int = 0):
            """Encode one box into hm / reg / mask.

            peak=1.0 for valid boxes, <1 for pseudo. quality (0..1] additionally
            scales the pseudo regression weight per box (teacher fit score)."""
            lc = b_col_f - c0
            lr = b_row_f - r0
            L, Wd, thd = b_row[2], b_row[3], float(b_row[4])
            if fx:
                lc = flip_extent - lc
                thd = 180.0 - thd
            if fy:
                lr = flip_extent - lr
                thd = -thd
            if rot_k:
                lc, lr, thd = _rotate_px(lc, lr, thd, C, rot_k,
                    boundary_coordinates=self.boundary_coordinate_augmentation)
            if not (0 <= lc < C and 0 <= lr < C):
                return
            th = np.deg2rad(thd)
            if not has_point_seg:
                # th is the WORLD-frame angle (y up); the seg target lives on the
                # IMAGE grid (row down), where the world->pixel mapping flips y
                # (row = H - y/cell). A vehicle at world angle th appears at -th in
                # pixel space, so the rectangle must be rasterised with -th to match
                # the actual footprint (verified: IoU 1.0 vs box_corners, 0.30 with +th)
                # and to stay consistent with render_obb_soft, which also negates th.
                _draw_seg_box(seg[0], seg_weight[0], lc / stride, lr / stride,
                              L / self.cell / stride, Wd / self.cell / stride,
                              -th, weight=1.0 if peak >= 1.0 else self.pseudo_seg_weight)
            ox, oy = lc / stride, lr / stride
            ix, iy = int(ox), int(oy)
            if not (0 <= ix < OS and 0 <= iy < OS):
                return
            # Tight peak so closely-parked cars keep SEPARATE maxima (instance split):
            # small radius (<=2 cells) -> crisp per-car peak that does not merge.
            rad = max(self.hm_radius_min,
                      min(self.hm_radius_max,
                          int(self.hm_radius_scale * min(L, Wd) / self.cell / stride)))
            draw_gaussian(hm[class_id], ix, iy, rad, peak=peak)
            if self.reg_support_radius == 0:
                # Preserve the retained vehicle-model target path exactly.
                if hm[class_id, iy, ix] >= peak - 1e-6:
                    yaw_factor = 2.0 if use_doubled_yaw(source, class_id) else 1.0
                    reg[:, iy, ix] = [ox - ix, oy - iy,
                                      np.log(max(L, 0.1)), np.log(max(Wd, 0.1)),
                                      np.sin(yaw_factor * th), np.cos(yaw_factor * th),
                                      b_row[5], b_row[6]]
                    mask[0, iy, ix] = (peak if peak >= 1.0
                                       else peak * self.pseudo_reg_weight * float(quality))
                ignore[0, iy, ix] = 0.0
                return
            dense = self.reg_support_radius
            if (self.reg_support_max_size_m > 0.0
                    and max(float(L), float(Wd)) > self.reg_support_max_size_m):
                dense = 0
            yaw_factor = 2.0 if use_doubled_yaw(source, class_id) else 1.0
            reg_weight = (peak if peak >= 1.0
                          else peak * self.pseudo_reg_weight * float(quality))
            for dy in range(-dense, dense + 1):
                for dx in range(-dense, dense + 1):
                    if dx * dx + dy * dy > dense * dense:
                        continue
                    jx, jy = ix + dx, iy + dy
                    if not (0 <= jx < OS and 0 <= jy < OS):
                        continue
                    # GT owns a cell before pseudo labels; equally confident
                    # targets are resolved by centre distance. This prevents
                    # contradictory dense labels in crowded strips.
                    ownership = float(peak) - 1e-3 * float(dx * dx + dy * dy)
                    if ownership <= float(reg_owner[jy, jx]):
                        continue
                    reg_owner[jy, jx] = ownership
                    reg_owner_class[jy, jx] = int(class_id)
                    reg[:, jy, jx] = [ox - jx, oy - jy,
                                      np.log(max(L, 0.1)), np.log(max(Wd, 0.1)),
                                      np.sin(yaw_factor * th), np.cos(yaw_factor * th),
                                      b_row[5], b_row[6]]
                    mask[0, jy, jx] = reg_weight
                    ignore[0, jy, jx] = 0.0

        # --- pseudo-OBBs first (lower peak) so valid boxes can override ---
        if self.pseudo_peak > 0:
            for k in range(len(pseudo_boxes)):
                _encode_box(pseudo_boxes[k], bpx_pseudo[k, 0], bpx_pseudo[k, 1],
                            peak=self.pseudo_peak,
                            quality=float(pseudo_scores[k]))

        # GT boxes (peak=1.0) always override pseudo at the same location.
        for k in range(len(boxes)):
            _encode_box(boxes[k], bpx[k, 0], bpx[k, 1], peak=1.0,
                        class_id=int(box_classes[k]))

        if self.reg_support_radius > 0:
            # Class-specific regression is selected from hm.argmax in the
            # trainer. Never retain a dense target underneath another class's
            # stronger heatmap response.
            class_mismatch = (reg_owner_class >= 0) & (hm.argmax(axis=0) != reg_owner_class)
            mask[0, class_mismatch] = 0.0

        return (torch.from_numpy(crop), torch.from_numpy(hm),
                torch.from_numpy(reg), torch.from_numpy(mask),
                torch.from_numpy(ignore), torch.from_numpy(seg),
                torch.from_numpy(seg_weight),
                torch.tensor(det_loss_weight, dtype=torch.float32),
                torch.tensor(seg_loss_weight, dtype=torch.float32))

    def _augment_channels(self, crop: np.ndarray, rng=None) -> np.ndarray:
        """Apply sensor/domain jitter to normalized BEV channels."""
        if not getattr(self, "_channel_augmentation_active", True):
            return crop
        rng = self.rng if rng is None else rng
        out = crop.astype(np.float32, copy=True)
        if self.channel_scale_jitter > 0:
            scale = rng.lognormal(
                mean=0.0, sigma=self.channel_scale_jitter, size=(out.shape[0], 1, 1)
            ).astype(np.float32)
            out *= scale
        if self.channel_noise_std > 0:
            out += rng.normal(0.0, self.channel_noise_std, size=out.shape).astype(np.float32)
        if self.channel_dropout_prob > 0 or self.channel_dropout_probs is not None:
            if self.channel_dropout_probs is not None and len(self.channel_dropout_probs) == out.shape[0]:
                probs = np.maximum(
                    self.channel_dropout_prob,
                    self.channel_dropout_probs.astype(np.float32),
                ).reshape(-1, 1, 1)
            else:
                probs = np.full((out.shape[0], 1, 1), self.channel_dropout_prob, np.float32)
            keep = rng.random((out.shape[0], 1, 1)) >= probs
            out *= keep.astype(np.float32)
        return np.ascontiguousarray(out)


# Learning-rate schedule.

class WarmupThenPlateau:
    """Linear warmup, then ReduceLROnPlateau.

    Call sched.step() each epoch unconditionally (no val_loss arg needed;
    the plateau scheduler is updated separately via sched.step_plateau(val)).
    """
    def __init__(self, opt, warmup_epochs: int, factor: float = 0.5,
                 patience: int = 6, min_lr: float = 1e-7):
        self._warmup_epochs = warmup_epochs
        self._ep = 0
        # LambdaLR applies its epoch-0 multiplier during construction. A
        # requested zero-length warmup must therefore be a no-op, not a silent
        # 1%-learning-rate freeze.
        self._warmup = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda ep: (1.0 if warmup_epochs <= 0
                             else max(ep / warmup_epochs, 1e-2)))
        self._plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=factor, patience=patience,
            min_lr=min_lr)

    def step(self):
        if self._ep < self._warmup_epochs:
            self._warmup.step()
        self._ep += 1

    def step_plateau(self, val_loss: float):
        if self._ep >= self._warmup_epochs:
            self._plateau.step(val_loss)

    def scale_lr(self, factor: float):
        """Permanently scale the learning rate (cascade-breaker rollback).

        Scales both the live param-group LRs and the warmup base LRs, so the
        reduction survives whether we are still warming up (LambdaLR rewrites
        group LRs from base_lrs each step) or already in the plateau phase
        (which only ever multiplies the current group LR downwards).
        """
        for g in self._plateau.optimizer.param_groups:
            g["lr"] = max(float(g["lr"]) * factor, self._plateau.min_lrs[0])
        self._warmup.base_lrs = [b * factor for b in self._warmup.base_lrs]

    def state_dict(self):
        return {"ep": self._ep,
                "warmup": self._warmup.state_dict(),
                "plateau": self._plateau.state_dict()}

    def load_state_dict(self, d):
        self._ep = d.get("ep", 0)
        if "warmup" in d:
            self._warmup.load_state_dict(d["warmup"])
        if "plateau" in d:
            self._plateau.load_state_dict(d["plateau"])


# Miscellaneous helpers.

def seed_worker(worker_id: int) -> None:
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng((info.seed + worker_id) % (2 ** 32))


def _weighted_batch_mean(per_sample: torch.Tensor,
                         sample_weights: torch.Tensor) -> torch.Tensor:
    """Combine a (B,) loss vector with (B,) sample weights.

    The per-sample losses are normalised inside each loss function, so the
    batch value is the sample-weighted sum over the summed weights.
    """
    weights = sample_weights.to(device=per_sample.device, dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1.0)


def batch_focal_loss(pred: torch.Tensor, gt: torch.Tensor, ignore: torch.Tensor,
                     sample_weights: torch.Tensor,
                     class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return _weighted_batch_mean(
        focal_loss_per_sample(pred, gt, ignore, class_weights=class_weights), sample_weights)


def batch_reg_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                 channel_weights: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    return _weighted_batch_mean(
        reg_l1_per_sample(pred, target, mask, channel_weights), sample_weights)


def batch_reg_l1_class_aware(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    channel_weights: torch.Tensor,
    sample_weights: torch.Tensor,
    class_ids: torch.Tensor,
    no_orientation_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Masked regression L1 with class-dependent yaw supervision."""
    weights = channel_weights.to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1)
    active = mask.expand_as(pred).to(dtype=pred.dtype).clone()
    if no_orientation_ids:
        no_yaw = torch.zeros_like(class_ids, dtype=torch.bool)
        for class_id in no_orientation_ids:
            no_yaw |= class_ids == int(class_id)
        active[:, 4:6] *= (~no_yaw).to(dtype=pred.dtype)[:, None]
    weighted = active * weights
    per_sample = (torch.abs(pred - target) * weighted).sum(dim=(1, 2, 3))
    per_sample = per_sample / weighted.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return _weighted_batch_mean(per_sample, sample_weights)


def batch_seg_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                   ignore: torch.Tensor, sample_weights: torch.Tensor,
                   dice_weight: float = 0.0,
                   tversky_weight: float = 0.0,
                   tversky_alpha: float = 0.7,
                   tversky_beta: float = 0.3) -> torch.Tensor:
    per_sample = weighted_seg_bce_per_sample(pred, target, weight, ignore)
    if dice_weight > 0:
        per_sample = per_sample + float(dice_weight) * weighted_seg_dice_per_sample(
            pred, target, weight, ignore)
    if tversky_weight > 0:
        per_sample = per_sample + float(tversky_weight) * weighted_seg_tversky_per_sample(
            pred, target, weight, ignore, alpha=tversky_alpha, beta=tversky_beta)
    return _weighted_batch_mean(per_sample, sample_weights)


def parse_source_weights(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad --source-weights item: {part!r}; expected source=weight")
        name, weight = part.split("=", 1)
        out[name.strip()] = float(weight)
    return out


def source_weight_lookup(weights: dict[str, float], source: str, default: float) -> float:
    return float(weights.get(source, default))


def summarize_sources(paths: list[Path], label: str) -> None:
    counts: dict[str, int] = {}
    valid: dict[str, int] = {}
    pseudo: dict[str, int] = {}
    for p in paths:
        with np.load(p) as d:
            src = str(d["source"]) if "source" in d else "unknown"
            counts[src] = counts.get(src, 0) + 1
            vb = d["boxes_valid"] if "boxes_valid" in d else d.get("boxes", np.zeros((0, 7)))
            pb = d["boxes_pseudo"] if "boxes_pseudo" in d else np.zeros((0, 7))
            valid[src] = valid.get(src, 0) + len(vb)
            pseudo[src] = pseudo.get(src, 0) + len(pb)
    parts = []
    for src in sorted(counts):
        parts.append(f"{src}:tiles={counts[src]} valid={valid.get(src, 0)} pseudo={pseudo.get(src, 0)}")
    print(f"{label}_sources: " + "; ".join(parts))
