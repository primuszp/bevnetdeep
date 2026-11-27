"""
bev_dataset.py — BEV multi-channel tensor + rotated-box labels.

Sensor-INDEPENDENT geometric channels only (no intensity), 0.25 m cells.
One "vehicle" class.

──────────────────────────────── FEATURE CHANNELS ────────────────────────────
"generic_v2" (18 stored channels, default) — audited v18 schema:
   0  ground_occ       ground occupancy
   1  ground_shadow    laser shadow under vehicle (strongest cue)
   2  ground_edt       distance inside solid ground (shadow = 0)
   3  shadow_edt       soft footprint (shadow-EDT, normalised)
   4  dtm_slope        terrain slope magnitude
   5  veh_h_p05        5th-percentile HAG in vehicle band
   6  veh_h_p50        50th-percentile HAG in vehicle band
   7  veh_h_p95        95th-percentile HAG in vehicle band
   8  veh_h_span       height span in vehicle band
   9  veh_density      log vehicle-point density
  10  veh_dog          Difference-of-Gaussians blob
  11  veh_sobel        Sobel gradient magnitude (OBB edge)
  12  veh_h_p95_hillshade_multi
  13  veh_h_p95_aargau_like
  14  all_h_p95        95th-percentile full HAG (building/tree context)
  15  all_sobel        Sobel gradient magnitude (full height)
  16  all_density      log all-point density
  17  veh_width_hole   vehicle-width gated ground-occlusion hole

"generic" (15 channels) — older banded schema (kept for legacy datasets):
  talaj-sav: ground_occ ground_shadow dtm_slope ground_edt shadow_edt
  jarmu-sav: veh_hmax veh_dog veh_laplacian veh_tophat veh_sobel
  teljes:    all_hmax all_dog all_laplacian all_tophat all_sobel

"legacy" (6 channels): occ log_count hag_max hag_mean hag_std height_step

──────────────────────────────── LABEL FIELDS ────────────────────────────────
Each box row: [cx, cy, length, width, theta_deg, z_center_hag, height]

Saved in NPZ:
  boxes_valid  (N,7) f32  — confident GT (DALESObjects instance OBBs)
  boxes_pseudo (P,7) f32  — weak supervision (semantic cluster pseudo-OBBs)
  boxes_ignore (M,7) f32  — ignored noisy/oversized clusters
  seg_mask     (H,W) u8   — raw vehicle point mask (sparse)
  fp_mask      (H,W) u8   — processed footprint (shadow-close or seg-close)
  seg_box_mask (H,W) u8   — OBB-rasterised footprint (DALES only, highest conf)
  seg_box_weight (H,W) f16 — confidence weights for OBB footprint edges

──────────────────────────────── SUPPORTED SOURCES ───────────────────────────
  dales          DALESObjects PLY — instance OBBs → boxes_valid
  yuto           YutoSemantic    — semantic → pseudo-OBBs (tiled 500 m windows)
  eclair         ECLAIR LAZ      — requires lazrs backend (pip install lazrs)
  isprs          Vaihingen PTS   — semantic → pseudo-OBBs

Use :mod:`bordereye.ml.bev.data.build_dataset` as the single dataset CLI.
"""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi

from bordereye import RasterGrid
from bordereye.geometry import oriented_bounding_box_2d
from bordereye.ml.bev.core.nn_contract import box_row_corners, world_to_pixel
from bordereye.ml.bev.data import bev_channels
from bordereye.rect_search import RectSearchConfig, instance_ids_from_rect

# ── class codes ────────────────────────────────────────────────────────────────
# DALESObjects PLY semantic classes used for instance-box validation.
VEHICLE_CLASSES = (3, 4)

# ── geometry constants ─────────────────────────────────────────────────────────
SINGLE_MAX_W = 3.2
SINGLE_MAX_L = 16.0
_RECT_CFG = RectSearchConfig()

HAG_FLOOR = 0.20
HAG_CAP   = 30.0
STEP_WIN  = 3

VALID_L_MIN = 2.0
VALID_L_MAX = 8.5
VALID_W_MIN = 1.2
VALID_W_MAX = 3.2
VALID_H_MIN = 0.5
VALID_H_MAX = 3.5
VALID_ASPECT_MIN = 1.5
VALID_ASPECT_MAX = 5.5
LOW_MIN = 0.25
LOW_MAX = 4.0
GND_MAX = 0.30
VEH_MIN = 0.30
VEH_MAX = 4.0

GENERIC_CHANNEL_NAMES = [
    "ground_occ", "ground_shadow", "dtm_slope", "ground_edt", "shadow_edt",
    "veh_hmax", "veh_dog", "veh_laplacian", "veh_tophat", "veh_sobel",
    "all_hmax", "all_dog", "all_laplacian", "all_tophat", "all_sobel",
]
# Canonical order lives in bordereye.ml.bev.data.bev_channels (single source of truth).
GENERIC_V2_CHANNEL_NAMES = list(bev_channels.GENERIC_V2_CHANNEL_NAMES)
LEGACY_CHANNEL_NAMES = ["occ", "log_count", "hag_max", "hag_mean", "hag_std", "height_step"]


# ── raster helpers ─────────────────────────────────────────────────────────────

def dense(raster: RasterGrid, stat: str) -> np.ndarray:
    return raster.to_dense(stat)


def _raster_stat(xyz, values, cell, bounds, stat, fill=0.0):
    if len(values) == 0:
        ref = RasterGrid.from_point_cloud(
            np.array([[bounds[0], bounds[1], 0.0], [bounds[2], bounds[3], 0.0]]),
            cell_size=cell, bounds=bounds)
        return np.full(ref.shape, fill, np.float32), ref
    pts = np.column_stack([xyz[:, 0], xyz[:, 1], values])
    r = RasterGrid.from_point_cloud(pts, cell_size=cell, bounds=bounds)
    return np.nan_to_num(dense(r, stat), nan=fill).astype(np.float32), r


def _fill_nearest(arr, mask_valid):
    """Fill ~mask_valid cells with the value of the nearest valid cell."""
    if mask_valid.all():
        return arr
    if not mask_valid.any():
        return np.zeros_like(arr)
    _, idx = ndi.distance_transform_edt(~mask_valid, return_indices=True)
    return arr[tuple(idx)]


# ── feature channel builders ───────────────────────────────────────────────────

def _surface_ops(surf, cell, veh_se):
    """DoG + Laplacian + top-hat + Sobel on a height surface raster."""
    s1 = max(1.0 / cell * 0.25, 0.6)
    s2 = max(1.0 / cell * 1.0, 2.4)
    dog = (ndi.gaussian_filter(surf, s1) - ndi.gaussian_filter(surf, s2)).astype(np.float32)
    lap = ndi.gaussian_laplace(surf, sigma=1.5).astype(np.float32)
    tophat = ndi.white_tophat(surf, size=veh_se).astype(np.float32)
    gx = ndi.sobel(surf, axis=1)
    gy = ndi.sobel(surf, axis=0)
    sob = np.hypot(gx, gy).astype(np.float32)
    return dog, lap, tophat, sob


def build_generic_v2_channels(xyz, hag, cell, bounds, channel_names=None):
    """Audited canonical stored BEV schema.

    Delegates to ``bordereye.ml.bev.data.bev_channels`` — the single source of truth with
    per-channel builders (verified bit-identical to the previous inline
    implementation). Generate individual channels via
    a versioned 18-channel construction contract.
    """
    channels, r, names = bev_channels.build_generic_v2(
        xyz, hag, cell, bounds, channel_names=channel_names
    )
    return channels, r, tuple(names)


def build_generic_channels(xyz, hag, cell, bounds):
    """Legacy banded schema (generic): top-5 operators per band."""
    veh_se = max(round(4.0 / cell), 3)

    valid = np.isfinite(hag) & (hag >= HAG_FLOOR)
    all_hmax, r = _raster_stat(xyz[valid], np.clip(hag[valid], 0, HAG_CAP),
                               cell, bounds, "max")

    ground_pt = np.isfinite(hag) & (hag < GND_MAX)
    g_occ, _ = _raster_stat(xyz[ground_pt], np.ones(int(ground_pt.sum()), np.float32),
                            cell, bounds, "max")
    ground_occ_b = g_occ > 0
    ground_occ = ground_occ_b.astype(np.float32)

    raw_shadow = ndi.binary_fill_holes(ground_occ_b) & ~ground_occ_b
    min_shadow_cells = max(round(1.0 / (cell * cell)), 4)
    lab, n = ndi.label(raw_shadow)
    if n > 0:
        sizes = ndi.sum(raw_shadow, lab, range(1, n + 1))
        ground_shadow = np.isin(lab, 1 + np.nonzero(sizes >= min_shadow_cells)[0]).astype(np.float32)
    else:
        ground_shadow = np.zeros(ground_occ_b.shape, np.float32)

    gz, _ = _raster_stat(xyz[ground_pt], xyz[ground_pt, 2], cell, bounds, "mean", fill=np.nan)
    gz = _fill_nearest(gz, np.isfinite(gz))
    dtm_slope = (np.hypot(ndi.sobel(gz, axis=1), ndi.sobel(gz, axis=0)) / (8.0 * cell)).astype(np.float32)
    ground_edt = (ndi.distance_transform_edt(ground_occ_b).astype(np.float32) * cell)
    shadow_edt = (ndi.distance_transform_edt(ground_shadow > 0).astype(np.float32) * cell)

    veh_m = np.isfinite(hag) & (hag >= VEH_MIN) & (hag <= VEH_MAX)
    veh_hmax, _ = _raster_stat(xyz[veh_m], np.clip(hag[veh_m], 0, VEH_MAX),
                               cell, bounds, "max")
    veh_dog, veh_lap, veh_tophat, veh_sobel = _surface_ops(veh_hmax, cell, veh_se)
    all_dog, all_lap, all_tophat, all_sobel = _surface_ops(all_hmax, cell, veh_se)

    channels = np.stack([
        ground_occ, ground_shadow, dtm_slope, ground_edt, shadow_edt,
        veh_hmax, veh_dog, veh_lap, veh_tophat, veh_sobel,
        all_hmax, all_dog, all_lap, all_tophat, all_sobel,
    ], axis=0).astype(np.float32)
    assert channels.shape[0] == len(GENERIC_CHANNEL_NAMES)
    return channels, r, GENERIC_CHANNEL_NAMES


def build_legacy_channels(xyz, hag, cell, bounds):
    off  = np.isfinite(hag) & (hag >= HAG_FLOOR)
    oxyz = np.column_stack([xyz[off, 0], xyz[off, 1], np.clip(hag[off], 0, HAG_CAP)])
    r    = RasterGrid.from_point_cloud(oxyz, cell_size=cell, bounds=bounds)
    r2   = RasterGrid.from_point_cloud(
        np.column_stack([oxyz[:, 0], oxyz[:, 1], oxyz[:, 2] ** 2]),
        cell_size=cell, bounds=bounds)
    count    = dense(r, "count").astype(np.float32)
    hag_max  = np.nan_to_num(dense(r, "max"),  nan=0.0).astype(np.float32)
    hag_mean = np.nan_to_num(dense(r, "mean"), nan=0.0).astype(np.float32)
    mean2    = np.nan_to_num(dense(r2, "mean"), nan=0.0)
    hag_std  = np.sqrt(np.clip(mean2 - hag_mean ** 2, 0, None)).astype(np.float32)
    occ      = (count > 0).astype(np.float32)
    log_cnt  = np.log1p(count).astype(np.float32)
    step     = (ndi.grey_dilation(hag_max, size=STEP_WIN)
                - ndi.grey_erosion(hag_max, size=STEP_WIN)).astype(np.float32)
    channels = np.stack([occ, log_cnt, hag_max, hag_mean, hag_std, step], axis=0)
    return channels, r, LEGACY_CHANNEL_NAMES


def build_channels(xyz, hag, cell, bounds, features, channel_names=None):
    if features == "legacy":
        return build_legacy_channels(xyz, hag, cell, bounds)
    if features == "generic_v2":
        return build_generic_v2_channels(xyz, hag, cell, bounds, channel_names)
    return build_generic_channels(xyz, hag, cell, bounds)


# ── OBB extraction ─────────────────────────────────────────────────────────────

def _box_row(xy, h):
    obb = oriented_bounding_box_2d(xy)
    h   = h[np.isfinite(h)]
    z_c, height = (float((h.min() + h.max()) / 2), float(h.max() - h.min())) if len(h) else (0.0, 0.0)
    return [float(obb.center[0]), float(obb.center[1]),
            float(obb.length), float(obb.width),
            float(obb.rotation_deg), z_c, height], obb


def is_valid_vehicle_box(row) -> bool:
    L, W, H = float(row[2]), float(row[3]), float(row[6])
    aspect = L / max(W, 1e-6)
    return (VALID_L_MIN <= L <= VALID_L_MAX and
            VALID_W_MIN <= W <= VALID_W_MAX and
            VALID_H_MIN <= H <= VALID_H_MAX and
            VALID_ASPECT_MIN <= aspect <= VALID_ASPECT_MAX)


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 7), np.float32)


def boxes_from_instances(xyz, sem, ins, hag):
    """DALESObjects instance IDs → high-confidence OBBs (boxes_valid)."""
    valid_rows, ignore_rows = [], []
    for iid in np.unique(ins[ins > 0]):
        m = ins == iid
        if int(np.bincount(sem[m]).argmax()) not in VEHICLE_CLASSES:
            continue
        xy = xyz[m, :2]
        h = hag[m]
        try:
            row, obb = _box_row(xy, h)
        except Exception:
            continue
        if obb.width <= SINGLE_MAX_W and obb.length <= SINGLE_MAX_L:
            (valid_rows if is_valid_vehicle_box(row) else ignore_rows).append(row)
            continue
        hh = np.where(np.isfinite(h), h, 0.0)
        sub_ids, dets = instance_ids_from_rect(
            np.column_stack([xy[:, 0], xy[:, 1], hh]), config=_RECT_CFG)
        if len(dets) <= 1:
            ignore_rows.append(row)
            continue
        split_count = 0
        for k in np.unique(sub_ids):
            mm = sub_ids == k
            if int(mm.sum()) < 8:
                continue
            try:
                srow, _ = _box_row(xy[mm], h[mm])
            except Exception:
                continue
            (valid_rows if is_valid_vehicle_box(srow) else ignore_rows).append(srow)
            split_count += 1
        if split_count == 0:
            ignore_rows.append(row)
    return (np.asarray(valid_rows,  np.float32).reshape(-1, 7),
            np.asarray(ignore_rows, np.float32).reshape(-1, 7))


def boxes_from_semantic_components(xyz, hag, vehicle_mask, cell, bounds,
                                   min_points: int = 12):
    """Semantic vehicle points → pseudo-OBBs via connected components in BEV."""
    pseudo_rows, ignore_rows = [], []
    veh = vehicle_mask & np.isfinite(hag) & (hag >= LOW_MIN) & (hag <= LOW_MAX)
    if not veh.any():
        return _empty_boxes(), _empty_boxes()

    vxyz, vhag = xyz[veh], hag[veh]
    occ, r = _raster_stat(vxyz, np.ones(int(veh.sum()), np.float32), cell, bounds, "max")
    occ = occ > 0
    occ = ndi.binary_closing(occ, structure=np.ones((3, 3), bool), iterations=1)
    occ = ndi.binary_dilation(occ, structure=np.ones((3, 3), bool), iterations=1)
    labels, n_lab = ndi.label(occ)
    if n_lab == 0:
        return _empty_boxes(), _empty_boxes()

    Hg, Wg = labels.shape
    cols = np.floor((vxyz[:, 0] - r.origin[0]) / cell).astype(np.int64)
    rows = np.floor(Hg - (vxyz[:, 1] - r.origin[1]) / cell).astype(np.int64)
    keep = (rows >= 0) & (rows < Hg) & (cols >= 0) & (cols < Wg)
    comp_id = np.zeros(len(vxyz), np.int32)
    comp_id[keep] = labels[rows[keep], cols[keep]]

    for cid in np.unique(comp_id[comp_id > 0]):
        m = comp_id == cid
        if int(m.sum()) < min_points:
            continue
        try:
            row, _ = _box_row(vxyz[m, :2], vhag[m])
        except Exception:
            continue
        (pseudo_rows if is_valid_vehicle_box(row) else ignore_rows).append(row)

    return (np.asarray(pseudo_rows, np.float32).reshape(-1, 7),
            np.asarray(ignore_rows, np.float32).reshape(-1, 7))


# ── visualisation helpers ──────────────────────────────────────────────────────

# ── NPZ serialisation ──────────────────────────────────────────────────────────

def vehicle_point_seg_mask(xyz, hag, vehicle_mask, r, cell) -> np.ndarray:
    """Rasterise semantic/instance vehicle points into a full-resolution BEV mask."""
    H, W = r.shape
    seg = np.zeros((H, W), np.uint8)
    m = vehicle_mask & np.isfinite(hag) & (hag >= LOW_MIN) & (hag <= LOW_MAX)
    if not m.any():
        return seg
    cols = np.floor((xyz[m, 0] - r.origin[0]) / cell).astype(np.int64)
    rows = (H - 1 - np.floor((xyz[m, 1] - r.origin[1]) / cell)).astype(np.int64)
    keep = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    seg[rows[keep], cols[keep]] = 1
    return seg


def vehicle_box_seg_mask(boxes, r, cell) -> np.ndarray:
    """Rasterise valid vehicle OBB footprints into a full-resolution BEV mask."""
    H, W = r.shape
    seg = np.zeros((H, W), np.uint8)
    if boxes is None or len(boxes) == 0:
        return seg
    for b in np.asarray(boxes, np.float32):
        verts = world_to_pixel(box_row_corners(b), r.origin, cell, H)
        x0 = max(0, int(np.floor(verts[:, 0].min())))
        x1 = min(W - 1, int(np.ceil(verts[:, 0].max())))
        y0 = max(0, int(np.floor(verts[:, 1].min())))
        y1 = min(H - 1, int(np.ceil(verts[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        px = xx.astype(np.float32) + 0.5
        py = yy.astype(np.float32) + 0.5
        cx = (float(b[0]) - r.origin[0]) / cell
        cy = H - (float(b[1]) - r.origin[1]) / cell
        theta = -np.deg2rad(float(b[4]))
        ca, sa = np.cos(-theta), np.sin(-theta)
        dx, dy = px - cx, py - cy
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        length = float(b[2]) / cell
        width = float(b[3]) / cell
        inside = (np.abs(lx) <= length / 2.0) & (np.abs(ly) <= width / 2.0)
        seg[y0:y1 + 1, x0:x1 + 1][inside] = 1
    return seg


def vehicle_box_seg_weight(seg_box_mask, cell, edge_band_m=0.50, edge_weight=0.40) -> np.ndarray:
    """Confidence weights for OBB-derived footprint: down-weight edge band."""
    mask = np.asarray(seg_box_mask).astype(bool)
    weight = np.ones(mask.shape, np.float32)
    if not mask.any():
        return weight
    band = max(float(edge_band_m), float(cell))
    inside_dist = ndi.distance_transform_edt(mask).astype(np.float32) * float(cell)
    outside_dist = ndi.distance_transform_edt(~mask).astype(np.float32) * float(cell)
    inner_conf = np.clip(inside_dist / band, 0.0, 1.0)
    weight[mask] = float(edge_weight) + (1.0 - float(edge_weight)) * inner_conf[mask]
    outer_band = (~mask) & (outside_dist <= band)
    weight[outer_band] = np.minimum(weight[outer_band], float(edge_weight))
    return weight.astype(np.float32)


def save_bev_npz(out_path, channels, r, boxes_valid, boxes_pseudo,
                 boxes_ignore, seg_mask, channel_names, args, source, source_split,
                 seg_box_mask=None, seg_box_weight=None, fp_mask=None, extra=None):
    out_path = Path(out_path)
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    payload = dict(
        channels=channels.astype(np.float32),
        origin=np.asarray(r.origin, np.float64),
        cell=np.float64(args.cell),
        shape=np.asarray(channels.shape[1:], np.int64),
        boxes=boxes_valid,           # backward-compat alias
        boxes_valid=boxes_valid,
        boxes_pseudo=boxes_pseudo,
        boxes_ignore=boxes_ignore,
        seg_mask=seg_mask.astype(np.uint8),
        channel_names=np.asarray(channel_names),
        features=np.asarray(args.features),
        source=np.asarray(source),
        source_split=np.asarray(source_split),
        channel_builder_version=np.asarray(bev_channels.CHANNEL_BUILDER_VERSION),
    )
    for attr in ("dataset_version", "bev_schema", "mask_policy", "pseudo_obb_policy"):
        if hasattr(args, attr):
            payload[attr] = np.asarray(getattr(args, attr))
    if seg_box_mask is not None:
        payload["seg_box_mask"] = seg_box_mask.astype(np.uint8)
    if seg_box_weight is not None:
        payload["seg_box_weight"] = seg_box_weight.astype(np.float16)
    if fp_mask is not None:
        payload["fp_mask"] = fp_mask.astype(np.uint8)
    if extra:
        payload.update(extra)
    # Uncompressed NPZ loads several times faster in the training crop
    # pipeline; compression stays the default for archival/sync friendliness.
    if getattr(args, "npz_compress", True):
        np.savez_compressed(tmp_path, **payload)
    else:
        np.savez(tmp_path, **payload)
    try:
        tmp_path.replace(out_path)
    except PermissionError:
        shutil.copy2(tmp_path, out_path)
        with contextlib.suppress(PermissionError):
            tmp_path.unlink(missing_ok=True)
