"""Mask-derived OBB teacher with vehicle length/width priors.

This module is the single source of truth for vehicle dimension priors
(``VehiclePrior`` / ``DEFAULT_PRIOR``). It is torch-free on purpose so the
dataset tools and the training-time mask anchors (core.consistency) can share
the same geometry without pulling in the model stack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi


@dataclass(frozen=True)
class VehiclePrior:
    """Log-normal vehicle footprint prior (metres; sigmas in log-space).

    The defaults describe a passenger car and are intentionally tight: on
    weakly labelled domains the mask constrains position and orientation well
    but barely the size, so the prior carries most of the size information.
    """
    length_m: float = 4.7
    width_m: float = 1.9
    length_sigma: float = 0.18
    width_sigma: float = 0.12
    min_length_m: float = 2.0
    max_length_m: float = 8.5
    min_width_m: float = 1.2
    max_width_m: float = 3.2


DEFAULT_PRIOR = VehiclePrior()


def split_component_by_length(comp: np.ndarray, cell_m: float,
                              prior: VehiclePrior = DEFAULT_PRIOR,
                              *,
                              split_factor: float = 1.6,
                              max_slots: int = 16,
                              min_pixels: int = 3):
    """Split one connected mask component into vehicle-length slots.

    Merged parking rows are exactly where detection supervision matters most,
    yet a single principal-axis fit on them is meaningless. This helper slices
    the component along its principal axis into ``round(extent / length_m)``
    slots so each slot is a plausible single-vehicle anchor.

    Args:
        comp:   (H, W) bool mask of ONE connected component on any grid.
        cell_m: metric size of one grid cell (e.g. ``cell * stride`` on the
                model output grid, or ``cell`` on the raw BEV grid).
        prior:  vehicle dimension prior; ``prior.length_m`` sets the slot pitch.
        split_factor: components with principal extent below
                ``split_factor * length_m`` are returned unchanged.

    Returns:
        List of ``(sub_mask_bool, cy, cx)`` with float pixel centroids. Always
        non-empty; degenerate inputs come back as a single unsplit entry.
    """
    rr, cc = np.nonzero(comp)
    if len(rr) == 0:
        return [(comp, 0.0, 0.0)]
    if len(rr) < 4:
        return [(comp, float(rr.mean()), float(cc.mean()))]
    xy = np.column_stack([cc, rr]).astype(np.float64)   # x = col, y = row
    mu = xy.mean(axis=0)
    cov = np.cov((xy - mu).T)
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, int(np.argmax(vals))]
    u = (xy - mu) @ axis
    u0, u1 = np.percentile(u, [2, 98])
    extent_m = max(float(u1 - u0), 1e-6) * float(cell_m)
    if extent_m <= split_factor * prior.length_m:
        return [(comp, float(mu[1]), float(mu[0]))]
    n_slots = int(np.clip(round(extent_m / max(prior.length_m, 1e-3)), 2, max_slots))
    edges = np.linspace(u0, u1, n_slots + 1)
    out = []
    for k in range(n_slots):
        sel = (u >= edges[k]) & (u <= edges[k + 1])
        if int(sel.sum()) < min_pixels:
            continue
        sub = np.zeros_like(comp, dtype=bool)
        sub[rr[sel], cc[sel]] = True
        out.append((sub, float(rr[sel].mean()), float(cc[sel].mean())))
    return out or [(comp, float(mu[1]), float(mu[0]))]


def mask_from_tile_npz(d, key: str = "auto") -> np.ndarray | None:
    keys = (("fp_mask", "seg_mask", "seg_box_mask") if key == "auto" else (key,))
    for k in keys:
        if k in d and np.asarray(d[k]).sum() > 0:
            return np.asarray(d[k]).astype(bool)
    return None


def candidate_support(mask: np.ndarray, origin: np.ndarray, cell: float,
                      cx: float, cy: float, length: float, width: float,
                      theta: float) -> float:
    H, W = mask.shape
    rad = int(np.ceil(0.5 * np.hypot(length, width) / cell)) + 2
    col = round((cx - origin[0]) / cell)
    row = round(H - (cy - origin[1]) / cell)
    r0, r1 = max(0, row - rad), min(H, row + rad + 1)
    c0, c1 = max(0, col - rad), min(W, col + rad + 1)
    if r1 <= r0 or c1 <= c0:
        return 0.0
    yy, xx = np.mgrid[r0:r1, c0:c1]
    x = origin[0] + (xx + 0.5) * cell
    y = origin[1] + (H - yy - 0.5) * cell
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(theta), np.sin(theta)
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    inside = (np.abs(lx) <= length * 0.5) & (np.abs(ly) <= width * 0.5)
    if not inside.any():
        return 0.0
    return float(mask[r0:r1, c0:c1][inside].mean())


def length_prior_boxes_from_mask(
    mask: np.ndarray,
    origin: np.ndarray,
    cell: float,
    *,
    car_length: float = 4.7,
    car_width: float = 1.9,
    support_thresh: float = 0.12,
    max_slots_per_component: int = 64,
    detect_cross_parking: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit vehicle-sized OBBs that explain a binary vehicle footprint mask.

    Merged components (parking rows) are sliced into vehicle slots along the
    component's principal axis. With ``detect_cross_parking`` (default) the
    fitter first checks how vehicles stand INSIDE the row: local PCA windows
    along the axis vote whether the per-vehicle elongation is in line with the
    row (nose-to-tail) or across it (side-by-side). Side-by-side rows get a
    car-width slot pitch and boxes rotated 90° to the row axis — without this
    every side-by-side row was fitted with crosswise boxes.

    Returns:
        boxes: ``(N,7)`` rows [cx, cy, length, width, theta_deg, z, height].
        scores: per-box mask support in [0,1].
    """
    lab, n = ndi.label(mask)
    rows_out = []
    scores = []
    H = mask.shape[0]
    for cid in range(1, n + 1):
        rr, cc = np.nonzero(lab == cid)
        if len(rr) < 4:
            continue
        xy = np.column_stack([
            origin[0] + (cc.astype(np.float64) + 0.5) * cell,
            origin[1] + (H - rr.astype(np.float64) - 0.5) * cell,
        ])
        mu = xy.mean(axis=0)
        cov = np.cov((xy - mu).T)
        vals, vecs = np.linalg.eigh(cov)
        axis = vecs[:, int(np.argmax(vals))]
        if axis[0] < 0:
            axis = -axis
        ortho = np.array([-axis[1], axis[0]], np.float64)
        u = (xy - mu) @ axis
        v = (xy - mu) @ ortho
        u0, u1 = np.percentile(u, [2, 98])
        v0, v1 = np.percentile(v, [5, 95])
        extent_l = max(float(u1 - u0), cell)
        extent_w = max(float(v1 - v0), cell)
        theta_deg = float(np.rad2deg(np.arctan2(axis[1], axis[0])) % 180.0)
        if extent_l <= car_length * 1.8 and extent_w <= car_width * 2.2:
            center = mu + axis * (0.5 * (u0 + u1))
            length = float(np.clip(extent_l, 2.0, 8.5))
            width = float(np.clip(extent_w, 1.2, 3.2))
            support = candidate_support(mask, origin, cell, center[0], center[1],
                                        length, width, np.deg2rad(theta_deg))
            if support >= support_thresh:
                rows_out.append([center[0], center[1], length, width, theta_deg, 1.0, 1.5])
                scores.append(support)
            continue
        # Vote on in-row vehicle orientation: local PCA in car-length windows.
        cross = False
        if detect_cross_parking and extent_w >= 0.9 * car_width:
            n_probe = int(np.clip(round(extent_l / max(car_length, 1e-3)), 1,
                                  max_slots_per_component))
            angles = []
            for cu in np.linspace(u0 + 0.5 * car_length, u1 - 0.5 * car_length,
                                  max(n_probe, 1)):
                local = (u >= cu - 0.5 * car_length) & (u <= cu + 0.5 * car_length)
                if int(local.sum()) < 6:
                    continue
                c2 = np.cov(np.column_stack([u[local], v[local]]).T)
                if not np.all(np.isfinite(c2)):
                    continue
                w2, v2 = np.linalg.eigh(c2)
                if w2[int(np.argmax(w2))] <= 0:
                    continue
                ax_local = v2[:, int(np.argmax(w2))]
                # angle of per-window elongation relative to the row axis
                angles.append(float(np.degrees(
                    np.arctan2(abs(ax_local[1]), abs(ax_local[0])))))
            cross = len(angles) >= 2 and float(np.median(angles)) > 45.0

        if cross:
            pitch = 1.25 * car_width          # typical side-by-side slot spacing
            theta_box = (theta_deg + 90.0) % 180.0
        else:
            pitch = car_length
            theta_box = theta_deg
        n_slots = int(np.clip(round(extent_l / max(pitch, 1e-3)), 1,
                              max_slots_per_component))
        centers_u = ([0.5 * (u0 + u1)] if n_slots == 1 else
                     np.linspace(u0 + 0.5 * pitch, u1 - 0.5 * pitch, n_slots))
        for cu in centers_u:
            local = (u >= cu - 0.65 * pitch) & (u <= cu + 0.65 * pitch)
            if local.sum() < 4:
                continue
            cv = float(np.median(v[local]))
            center = mu + axis * cu + ortho * cv
            support = candidate_support(mask, origin, cell, center[0], center[1],
                                        car_length, car_width, np.deg2rad(theta_box))
            if support >= support_thresh:
                rows_out.append([center[0], center[1], car_length, car_width, theta_box, 1.0, 1.5])
                scores.append(support)
    return (np.asarray(rows_out, np.float32).reshape(-1, 7),
            np.asarray(scores, np.float32))
