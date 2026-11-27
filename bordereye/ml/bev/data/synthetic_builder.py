"""Synthetic vehicle-scene generation for BEV dataset construction."""
from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from bordereye.geometry import rect_corners_2d, to_box_frame_2d
from bordereye.ml.bev.core.footprint_masks import fp_mask_from_boxes
from bordereye.ml.bev.core.nn_contract import box_row_corners
from bordereye.ml.bev.data import bev_dataset as DS
from bordereye.ml.bev.data.build_contract import (
    save_args as _save_args,
)
from bordereye.ml.bev.data.build_contract import (
    should_skip_existing as _should_skip_existing,
)

ARCHETYPES = {
    "sedan":  dict(L=(4.0, 5.0), W=(1.7, 2.0), h_low=(0.65, 0.85), h_high=(1.35, 1.5),  cab=(0.30, 0.78)),
    "suv":    dict(L=(4.4, 5.2), W=(1.9, 2.1), h_low=(0.75, 0.95), h_high=(1.5,  1.7),  cab=(0.22, 0.85)),
    "van":    dict(L=(4.8, 5.8), W=(1.9, 2.2), h_low=(0.9,  1.1),  h_high=(1.75, 2.05), cab=(0.18, 0.95)),
    "pickup": dict(L=(5.0, 5.8), W=(1.9, 2.1), h_low=(0.8,  1.0),  h_high=(1.55, 1.75), cab=(0.12, 0.50)),
    "truck":  dict(L=(6.5, 10.0),W=(2.3, 2.55),h_low=(1.0,  1.3),  h_high=(2.6,  3.4),  cab=(0.0,  1.0)),
}
_ARCH_NAMES = list(ARCHETYPES.keys())
_ARCH_PROB  = np.array([0.52, 0.15, 0.08, 0.10, 0.15])
_PROFILE    = dict(low=0.18, belt=0.54, roof=0.25, upper=0.03)


def _vehicle_top_points(rng, L, W, arch, density):
    p = ARCHETYPES[arch]
    h_low  = rng.uniform(*p["h_low"])
    h_high = rng.uniform(*p["h_high"])
    cab0   = float(np.clip(p["cab"][0] + rng.uniform(-0.04, 0.04), 0.02, 0.9))
    cab1   = float(np.clip(p["cab"][1] + rng.uniform(-0.04, 0.04), cab0 + 0.12, 0.98))
    n      = max(int(density * L * W), 12)
    fr     = _PROFILE
    n_low  = int(n * fr["low"])
    n_belt = int(n * fr["belt"])
    n_roof = int(n * fr["roof"])
    n_up   = max(n - n_low - n_belt - n_roof, 0)

    def U(a, b, k): return rng.uniform(a, b, k)
    bx = (U(0.0, 1.0, n_belt) - 0.5) * L
    by = U(-0.5, 0.5, n_belt) * W
    bz = rng.normal(h_low, 0.12, n_belt)
    rx = (U(cab0, cab1, n_roof) - 0.5) * L
    ry = U(-0.42, 0.42, n_roof) * W
    rz = rng.normal(h_high, 0.09, n_roof)
    side = rng.random(n_low) < 0.6
    lx = np.where(side, U(-0.5, 0.5, n_low) * L, rng.choice([-1, 1], n_low) * 0.5 * L)
    ly = np.where(side, rng.choice([-1, 1], n_low) * U(0.38, 0.5, n_low) * W, U(-0.5, 0.5, n_low) * W)
    lz = U(0.12, 0.5, n_low)
    ux = (U(cab0, cab1, n_up) - 0.5) * L
    uy = U(-0.35, 0.35, n_up) * W
    uz = U(h_high, h_high + 0.25, n_up)
    x  = np.concatenate([bx, rx, lx, ux]) + rng.normal(0, 0.03, n_low + n_belt + n_roof + n_up)
    y  = np.concatenate([by, ry, ly, uy]) + rng.normal(0, 0.03, len(x))
    z  = np.concatenate([bz, rz, lz, uz]) + rng.normal(0, 0.02, len(x))
    return np.column_stack((x, y, z)), h_low, h_high


def _obb_corners(cx, cy, L, W, theta_deg):
    return rect_corners_2d(cx, cy, L, W, np.deg2rad(theta_deg))


def _in_obb(pts_xy, cx, cy, L, W, theta_deg, margin=0.0):
    along_l, along_w = to_box_frame_2d(pts_xy, cx, cy, np.deg2rad(theta_deg))
    return (np.abs(along_l) <= L / 2 + margin) & (np.abs(along_w) <= W / 2 + margin)


def _obb_overlap(a, b, eps=0.05):
    ca, cb = _obb_corners(*a[:5]), _obb_corners(*b[:5])
    for poly in (ca, cb):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis) + 1e-9
            pa, pb = ca @ axis, cb @ axis
            if pa.min() >= pb.max() - eps or pb.min() >= pa.max() - eps:
                return False
    return True


def _parking_block(rng, size, cx, cy, theta, arch):
    """A 2D grid of parked vehicles (dense parking lot).

    1D rows never produce the pattern that fails most in the real data:
    several PARALLEL rows of side-by-side cars packed into a block. Within a
    row cars share their width edge (neighbour pitch = width + small gap);
    rows are stacked along the car-length axis, separated either back-to-back
    or by a driving lane. Every car in the block shares one orientation, so
    the block is a coherent lattice the fitter/teacher must learn to split.
    """
    th = np.deg2rad(theta)
    across = np.array([np.cos(th), np.sin(th)])       # neighbours within a row
    along = np.array([-np.sin(th), np.cos(th)])       # row-to-row direction
    n_cols = int(rng.integers(3, 11))                 # cars per row
    n_rows = int(rng.integers(2, 6))                  # number of rows
    car_theta = (theta + 90.0) % 180.0                # length âź‚ to the row line
    Wc = rng.uniform(*ARCHETYPES[arch]["W"])
    Lc = rng.uniform(*ARCHETYPES[arch]["L"])
    col_pitch = Wc + rng.uniform(0.1, 0.5)            # side-by-side gap
    lane = rng.random() < 0.5
    row_pitch = Lc + (rng.uniform(3.0, 6.0) if lane else rng.uniform(0.3, 0.9))
    out = []
    for r in range(n_rows):
        for c in range(n_cols):
            off = along * (r * row_pitch) + across * (c * col_pitch)
            px, py = cx + off[0], cy + off[1]
            if not (6 < px < size - 6 and 6 < py < size - 6):
                continue
            out.append((px, py, Lc * rng.uniform(0.95, 1.05),
                        Wc * rng.uniform(0.95, 1.05),
                        car_theta + rng.uniform(-1.5, 1.5), arch))
    return out


def _place_vehicles(rng, size, n_target, parking_block_prob=0.18):
    placed = []
    attempts = 0
    while len(placed) < n_target and attempts < n_target * 30:
        attempts += 1
        arch = rng.choice(_ARCH_NAMES, p=_ARCH_PROB)
        L = rng.uniform(*ARCHETYPES[arch]["L"])
        W = rng.uniform(*ARCHETYPES[arch]["W"])
        theta = rng.uniform(0, 180)
        cx = rng.uniform(6, size - 6)
        cy = rng.uniform(6, size - 6)
        if rng.random() < parking_block_prob:
            placed.extend(_parking_block(rng, size, cx, cy, theta, arch))
            continue
        row = [(cx, cy, L, W, theta, arch)]
        if rng.random() < 0.6:
            k = rng.integers(3, 13)
            th = np.deg2rad(theta)
            longitudinal = rng.random() < 0.4
            tight = rng.random() < 0.5
            if longitudinal:
                ux, uy = np.cos(th), np.sin(th)
                row_gap = rng.uniform(0.05, 0.3) if tight else rng.uniform(0.1, 0.9)
            else:
                ux, uy = -np.sin(th), np.cos(th)
                row_gap = rng.uniform(0.05, 0.35) if tight else rng.uniform(0.1, 1.2)
            prev_ext = L if longitudinal else W
            for _ in range(k - 1):
                Lj = rng.uniform(*ARCHETYPES[arch]["L"])
                Wj = rng.uniform(*ARCHETYPES[arch]["W"])
                ext_j = Lj if longitudinal else Wj
                step = 0.5 * prev_ext + 0.5 * ext_j + row_gap
                cx += ux * step
                cy += uy * step
                if not (6 < cx < size - 6 and 6 < cy < size - 6):
                    break
                row.append((cx, cy, Lj, Wj, theta + rng.uniform(-1.5, 1.5), arch))
                prev_ext = ext_j
        placed.extend(row)
    kept = []
    for b in placed:
        if all(not _obb_overlap(b, k) for k in kept):
            kept.append(b)
    return kept


def _house_points(rng, L, W, density):
    n = max(int(density * L * W), 40)
    u = rng.uniform(-0.5, 0.5, n)
    v = rng.uniform(-0.5, 0.5, n)
    x, y = u * L, v * W
    eave = rng.uniform(2.6, 4.5)
    if rng.random() < 0.5:
        z = np.full(n, eave + rng.uniform(0.0, 1.5))
    else:
        z = eave + rng.uniform(1.5, 3.5) * (1.0 - np.abs(v) * 2.0)
    z += rng.normal(0, 0.04, n)
    x += rng.normal(0, 0.05, n)
    y += rng.normal(0, 0.05, n)
    return np.column_stack((x, y, z))


def _make_synth_scene(rng, cfg):
    """Generate one synthetic LiDAR scene. Returns (xyz, hag, veh_mask, boxes_valid, bounds, meta)."""
    size = cfg["tile_size"]
    bounds = (0.0, 0.0, float(size), float(size))
    rho_g  = rng.uniform(*cfg["ground_density"])
    rho_v  = rng.uniform(*cfg["veh_density"])
    slope  = rng.uniform(0, cfg["max_slope"])
    saz    = rng.uniform(0, 2 * np.pi)
    plane  = lambda X, Y: slope * (np.cos(saz) * X + np.sin(saz) * Y)  # noqa

    veh = _place_vehicles(rng, size, rng.integers(*[x + (1 if i else 0) for i, x in enumerate(cfg["veh_per_scene"])]))
    houses, bushes, roofclutter = [], [], []
    for _ in range(rng.integers(*[x + (1 if i else 0) for i, x in enumerate(cfg["houses_per_scene"])])):
        hl = rng.uniform(6.0, 16.0)
        hw = rng.uniform(5.0, 12.0)
        hx = rng.uniform(hl, size - hl)
        hy = rng.uniform(hl, size - hl)
        hth = rng.uniform(0, 180)
        if veh and any(_in_obb(np.array([[hx, hy]]), c, cy, L + hl, W + hw, t)[0] for (c, cy, L, W, t, _) in veh):
            continue
        houses.append((hx, hy, hl, hw, hth))
    def _clear(px, py, pad=1.5):
        return not (veh and any(_in_obb(np.array([[px, py]]), c, cy, L + pad, W + pad, t)[0] for (c, cy, L, W, t, _) in veh))
    for _ in range(rng.integers(*[x + (1 if i else 0) for i, x in enumerate(cfg["bushes_per_scene"])])):
        bx, by = rng.uniform(3, size - 3), rng.uniform(3, size - 3)
        if _clear(bx, by):
            bushes.append((bx, by, rng.uniform(0.8, 2.6), rng.uniform(0.2, 0.4), rng.uniform(1.2, 3.6)))
    for _ in range(rng.integers(*[x + (1 if i else 0) for i, x in enumerate(cfg["roof_clutter_per_scene"])])):
        rL = rng.uniform(2.0, 5.0)
        rW = rng.uniform(1.5, 3.0)
        rth = rng.uniform(0, 180)
        if houses and rng.random() < 0.6:
            hx, hy, hl, hw, _ = houses[rng.integers(len(houses))]
            rx = hx + rng.uniform(-0.3, 0.3) * hl
            ry = hy + rng.uniform(-0.3, 0.3) * hw
            zb = rng.uniform(2.6, 4.0)
        else:
            rx, ry = rng.uniform(4, size - 4), rng.uniform(4, size - 4)
            if not _clear(rx, ry, pad=2.0):
                continue
            zb = 0.0
        roofclutter.append((rx, ry, rL, rW, rth, zb, zb + rng.uniform(1.6, 3.2)))

    # vehicle points
    vx, vins_list = [], []
    boxes = []
    for iid, (cx, cy, L, W, theta, arch) in enumerate(veh, start=1):
        loc, _h_low, h_high = _vehicle_top_points(rng, L, W, arch, rho_v)
        th = np.deg2rad(theta)
        c, s = np.cos(th), np.sin(th)
        vx.append(np.column_stack((c * loc[:, 0] - s * loc[:, 1] + cx,
                                    s * loc[:, 0] + c * loc[:, 1] + cy,
                                    loc[:, 2] + plane(cx, cy))))
        vins_list.append(np.full(len(loc), iid))
        boxes.append((cx, cy, L, W, theta % 180.0, h_high / 2.0, h_high))
    vxyz = np.vstack(vx) if vx else np.zeros((0, 3))

    # ground with occlusion shadow
    n_g = int(rho_g * size * size)
    gx = rng.uniform(0, size, n_g)
    gy = rng.uniform(0, size, n_g)
    keep = np.ones(n_g, bool)
    gxy = np.column_stack((gx, gy))
    for (cx, cy, L, W, theta, _) in veh:
        keep &= ~_in_obb(gxy, cx, cy, L, W, theta, margin=0.05)
    for (hx, hy, hl, hw, hth) in houses:
        keep &= ~_in_obb(gxy, hx, hy, hl, hw, hth)
    for (bx, by, rad, _, _h1) in bushes:
        keep &= ~(((gxy[:, 0] - bx) ** 2 + (gxy[:, 1] - by) ** 2) <= rad * rad)
    for (rx, ry, rL, rW, rth, zb, _) in roofclutter:
        if zb == 0.0:
            keep &= ~_in_obb(gxy, rx, ry, rL, rW, rth)
    gx, gy = gx[keep], gy[keep]
    gz = plane(gx, gy) + rng.normal(0, cfg["ground_noise"], gx.shape[0])
    gxyz = np.column_stack((gx, gy, gz))

    # clutter: trees, houses, bushes, rooftop
    cxyz = []
    for _ in range(rng.integers(*[x + (1 if i else 0) for i, x in enumerate(cfg["trees_per_scene"])])):
        tx, ty = rng.uniform(3, size - 3), rng.uniform(3, size - 3)
        if veh and any(_in_obb(np.array([[tx, ty]]), c, cy, L + 1.5, W + 1.5, t)[0] for (c, cy, L, W, t, _) in veh):
            continue
        rad = rng.uniform(1.2, 3.5)
        base = rng.uniform(1.0, 3.0)
        top = rng.uniform(4.0, 11.0)
        nt = int(rng.uniform(200, 800))
        ang = rng.uniform(0, 2 * np.pi, nt)
        rr = rad * np.sqrt(rng.uniform(0, 1, nt))
        cz = rng.uniform(base, top, nt)
        cxyz.append(np.column_stack((tx + rr * np.cos(ang), ty + rr * np.sin(ang), plane(tx, ty) + cz)))
    for (hx, hy, hl, hw, hth) in houses:
        loc = _house_points(rng, hl, hw, rng.uniform(40, 120))
        th = np.deg2rad(hth)
        c, s = np.cos(th), np.sin(th)
        cxyz.append(np.column_stack((c * loc[:, 0] - s * loc[:, 1] + hx,
                                      s * loc[:, 0] + c * loc[:, 1] + hy,
                                      loc[:, 2] + plane(hx, hy))))
    for (bx, by, rad, h0, h1) in bushes:
        nb = int(rng.uniform(120, 420))
        ang = rng.uniform(0, 2 * np.pi, nb)
        zf = rng.uniform(0, 1, nb)
        rr = rad * np.sqrt(rng.uniform(0, 1, nb)) * (1.0 - 0.4 * zf)
        cz = h0 + (h1 - h0) * zf + rng.normal(0, 0.08, nb)
        cxyz.append(np.column_stack((bx + rr * np.cos(ang), by + rr * np.sin(ang),
                                      plane(bx, by) + np.clip(cz, 0.05, None))))
    for (rx, ry, rL, rW, rth, _zb, zt) in roofclutter:
        nr = int(rng.uniform(60, 200))
        u = rng.uniform(-0.5, 0.5, nr) * rL
        v = rng.uniform(-0.5, 0.5, nr) * rW
        th = np.deg2rad(rth)
        c, s = np.cos(th), np.sin(th)
        cxyz.append(np.column_stack((c * u - s * v + rx, s * u + c * v + ry,
                                      plane(rx, ry) + zt + rng.normal(0, 0.04, nr))))
    clut = np.vstack(cxyz) if cxyz else np.zeros((0, 3))

    xyz = np.vstack((gxyz, vxyz, clut))
    ground_z = plane(xyz[:, 0], xyz[:, 1])
    hag = (xyz[:, 2] - ground_z).astype(np.float64)
    veh_mask = np.zeros(len(xyz), bool)
    veh_mask[len(gxyz):len(gxyz) + len(vxyz)] = True
    boxes_valid = np.asarray(boxes, np.float32).reshape(-1, 7)
    return xyz, hag, veh_mask, boxes_valid, bounds, {
        "rho_v": rho_v, "n_house": len(houses), "n_bush": len(bushes),
        "n_roof": len(roofclutter), "n_veh": len(boxes),
    }


# ── viz helper ─────────────────────────────────────────────────────────────────

def _save_viz(path, channels, channel_names, r, boxes_valid, boxes_pseudo=None,
              fp_mask=None, seg_box_mask=None, background=None):
    """Save a compact PNG preview: height-band background + OBB/fp overlays.

    The default background is the veh_h_* HEIGHT-BAND input channel (every
    point with HAG in the vehicle band, vegetation and walls included) — i.e.
    what the network actually sees. Pass ``background`` as an explicit (H, W)
    array (e.g. the semantic seg_mask) to render a different base layer.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
    except ImportError:
        return

    if background is not None:
        img = np.asarray(background, np.float32)
    else:
        ch = list(channel_names)
        preferred = ("veh_h_p50", "veh_h_p95", "veh_hmax", "veh_density")
        vis_idx = next((ch.index(n) for n in preferred if n in ch), min(7, len(ch) - 1))
        img = channels[vis_idx]
    H = img.shape[0]
    vmax = float(np.percentile(img[img > 0], 99)) if (img > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")

    def _to_px(xy):
        col = (xy[:, 0] - r.origin[0]) / r.cell_size[0]
        row = H - (xy[:, 1] - r.origin[1]) / r.cell_size[1]
        return np.column_stack([col, row])

    for b in np.asarray(boxes_valid).reshape(-1, 7):
        ax.add_patch(MplPoly(_to_px(box_row_corners(b)), closed=True,
                              fill=False, edgecolor="lime", linewidth=1.2))
    if boxes_pseudo is not None:
        for b in np.asarray(boxes_pseudo).reshape(-1, 7):
            ax.add_patch(MplPoly(_to_px(box_row_corners(b)), closed=True,
                                  fill=False, edgecolor="cyan", linewidth=0.9))
    if fp_mask is not None and fp_mask.any():
        ax.contour(fp_mask, levels=[0.5], colors=["orangered"], linewidths=0.8)
    elif seg_box_mask is not None and seg_box_mask.any():
        ax.contour(seg_box_mask.astype(float), levels=[0.5], colors=["yellow"], linewidths=0.8)

    n_v = len(np.asarray(boxes_valid).reshape(-1, 7))
    n_p = len(np.asarray(boxes_pseudo).reshape(-1, 7)) if boxes_pseudo is not None else 0
    ax.set_title(f"{path.stem}  valid={n_v} pseudo={n_p}", fontsize=9)
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── worker functions (top-level for pickling) ──────────────────────────────────

def _worker_synth(job):
    """Process one synthetic scene and save NPZ + optional viz.

    With cfg["scan_profile"] set (synthsparse source) the dense scene is
    degraded to airborne scan-line evidence BEFORE rasterisation, while the
    labels stay the perfect scene-generator GT. fp_mask then uses the same
    solid box-fill policy as the final weak-source dataset contract,
    so the segmentation target is a contiguous blob, not the scan stripes.
    """
    idx, split, cfg, out_dir, viz_dir = job
    source_name = cfg.get("synth_name", "synth")
    tile_id = f"{source_name}_{split}_{idx:04d}"
    out_path = Path(out_dir) / f"{tile_id}.npz"
    if _should_skip_existing(out_path, cfg):
        return tile_id, {"skipped": "exists"}
    try:
        rng = np.random.default_rng(cfg["seed"] + idx + (10000 if split == "test" else 0))
        xyz, hag, veh_mask, boxes_valid, bounds, meta = _make_synth_scene(rng, cfg)
        profile_name = cfg.get("scan_profile")
        if profile_name:
            from bordereye.ml.bev.core.scan_degrade import (
                MIX_PROFILES,
                PROFILES,
                sample_passes,
                scan_keep_mask,
            )
            name = (profile_name if profile_name != "mix"
                    else MIX_PROFILES[int(rng.integers(len(MIX_PROFILES)))])
            keep = scan_keep_mask(xyz[:, :2], rng, sample_passes(rng, PROFILES[name]))
            xyz, hag, veh_mask = xyz[keep], hag[keep], veh_mask[keep]
            meta = dict(meta, scan_profile=name, kept_frac=float(keep.mean()))
        save_args = _save_args(cfg)
        channels, r, channel_names = DS.build_channels(xyz, hag, cfg["cell"], bounds, cfg["features"])
        seg_mask     = DS.vehicle_point_seg_mask(xyz, hag, veh_mask, r, cfg["cell"])
        boxes_ignore = np.zeros((0, 7), np.float32)
        if profile_name and len(boxes_valid):
            # A box the scan left (nearly) no returns in is unlearnable from
            # this tile — real surveys simply would not have it labelled.
            # Move such boxes to ignore so they cost neither hm nor recall.
            from bordereye.ml.bev.core.mask_obb_teacher import candidate_support
            supported = np.array([
                candidate_support(seg_mask > 0, r.origin, cfg["cell"],
                                  float(b[0]), float(b[1]), float(b[2]),
                                  float(b[3]), float(np.deg2rad(b[4])))
                * (float(b[2]) * float(b[3]) / cfg["cell"] ** 2) >= 3.0
                for b in boxes_valid])
            boxes_ignore = boxes_valid[~supported]
            boxes_valid = boxes_valid[supported]
            meta = dict(meta, n_veh_ignored=len(boxes_ignore))
        seg_box_mask = DS.vehicle_box_seg_mask(boxes_valid, r, cfg["cell"])
        seg_box_wt   = DS.vehicle_box_seg_weight(seg_box_mask, cfg["cell"])
        fp_mask      = fp_mask_from_boxes(seg_mask, boxes_valid, r.origin, cfg["cell"],
                                          solid_fill=bool(profile_name))
        empty        = np.zeros((0, 7), np.float32)
        DS.save_bev_npz(
            out_path, channels, r,
            boxes_valid, empty, boxes_ignore, seg_mask, channel_names,
            save_args, source_name, split,
            seg_box_mask=seg_box_mask, seg_box_weight=seg_box_wt, fp_mask=fp_mask,
        )
        if viz_dir is not None:
            _save_viz(viz_dir / f"{tile_id}.png", channels, channel_names, r,
                      boxes_valid, fp_mask=fp_mask, seg_box_mask=seg_box_mask)
        return tile_id, {"n_veh": meta["n_veh"], "pts": len(xyz), "split": split}
    except Exception:
        return f"synth_{split}_{idx:04d}", {"error": traceback.format_exc()}
