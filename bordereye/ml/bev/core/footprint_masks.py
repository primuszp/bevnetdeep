"""Vehicle footprint mask construction for BEV training supervision.

Pipeline (all sources):
  1. vehicle_point_seg_mask  — rasterise vehicle-class points only (HAG 0.25–4 m)
  2. fp_seg_close            — morphological closing to fill sparse-return gaps
  3. extract_fp_mask_by_boxes — clip result to each OBB region; drops non-OBB blobs

Uses cv2 for elliptic structuring element and contour fill when available;
falls back to scipy otherwise.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

# ── shape filter constants ─────────────────────────────────────────────────────
MIN_AREA_M2 = 1.2   # smallest accepted vehicle footprint (m²)
CLOSE_M_SEG = 0.80  # closing radius to fill sparse-return gaps (metres)


# ── low-level helpers ──────────────────────────────────────────────────────────

def _cv2_close(binary: np.ndarray, close_px: int) -> np.ndarray:
    """Morphological closing with an elliptic structuring element (cv2 preferred)."""
    try:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        return cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, k)
    except ImportError:
        return ndi.binary_closing(binary.astype(bool), iterations=close_px).astype(np.uint8)


def _fill_and_draw(comp: np.ndarray) -> np.ndarray:
    """Fill holes in a boolean component mask and smooth with contour rasterisation."""
    filled = ndi.binary_fill_holes(comp).astype(np.uint8)
    try:
        import cv2
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(filled)
        cv2.drawContours(out, cnts, -1, 1, thickness=cv2.FILLED)
        return out
    except ImportError:
        return filled


# ── public API ─────────────────────────────────────────────────────────────────

def fp_seg_close(seg_mask: np.ndarray, cell: float) -> np.ndarray:
    """Seg-close mode for airborne LiDAR without ground shadow (ECLAIR, YUTO, ISPRS).

    Returns a uint8 mask with morphologically-completed vehicle footprints.
    """
    binary = (seg_mask > 0).astype(np.uint8)
    if not binary.any():
        return binary

    close_px = max(1, round(CLOSE_M_SEG / cell))
    closed = _cv2_close(binary, close_px)

    # No upper area limit here — merged blobs (parking rows) are split later
    # by split_fp_mask_by_boxes.  Only remove tiny noise fragments.
    labeled, _ = ndi.label(closed)
    out = np.zeros_like(binary, np.uint8)

    component_sizes = np.bincount(labeled.ravel())
    component_slices = ndi.find_objects(labeled)
    for lbl in np.flatnonzero(component_sizes[1:] * cell ** 2 >= MIN_AREA_M2) + 1:
        region = component_slices[int(lbl) - 1]
        if region is None:
            continue
        comp = labeled[region] == lbl
        out[region] |= _fill_and_draw(comp)

    return out


def fp_mask_from_boxes(
    seg_mask: np.ndarray,
    boxes: np.ndarray,
    origin: tuple[float, float],
    cell: float,
    margin_in: float = 0.3,
    keep_unexplained: bool = False,
    sparse_box_fill: bool = False,
    sparse_min_pts: int = 4,
    solid_fill: bool = False,
) -> np.ndarray:
    """Build fp_mask with per-OBB independent closing.

    For each OBB:
      1. Clip `seg_mask` to OBB + margin_in to pick up edge vehicle points.
      2. Morphologically close within that clip (fills sparse-return gaps).
      3. Clip result back to the STRICT OBB boundary (no margin).
      4. Erode by 1 pixel so adjacent cars never share boundary pixels in `out`.

    Steps 3-4 ensure that two neighbouring OBBs always produce a ≥1-pixel gap
    in the output mask, preventing `ndi.label` from merging their contributions
    into a single connected component.

    With ``keep_unexplained=True`` (weak-label sources), seg components that no
    box touches at all are seg-closed and added back instead of being dropped:
    on semantic sources these are almost always real vehicles the pseudo-OBB
    teacher missed, and silently deleting them from the footprint target would
    train the segmentation head to ignore them. GT-box sources should keep the
    default False, where non-box blobs are noise by construction.

    ``sparse_box_fill=True`` handles sparse airborne evidence (~10 vehicle
    pixels per car): when the close+erode result of a box falls below
    MIN_AREA_M2 but at least ``sparse_min_pts`` seg pixels lie inside the
    strict OBB, the box footprint itself (eroded 1 px) becomes the target —
    the same box-derived footprint policy GT sources use via seg_box_mask.
    Without this, whole sparse tiles end up with an empty fp_mask.

    ``solid_fill=True`` generalises that policy to EVERY box: whenever at
    least ``sparse_min_pts`` seg pixels support the box, the target is the
    solid box footprint (eroded 1 px), not the closed evidence. Sparse
    airborne scan lines (ECLAIR/YUTO) leave striped evidence that no
    reasonable closing radius can bridge, and a striped segmentation target
    teaches the seg head to reproduce the scan pattern instead of contiguous
    vehicle blobs.
    """
    boxes = np.asarray(boxes, np.float32).reshape(-1, 7)
    if not seg_mask.any() or len(boxes) == 0:
        if keep_unexplained and seg_mask.any():
            return fp_seg_close(seg_mask, cell)
        return np.zeros_like(seg_mask, np.uint8)

    H, W = seg_mask.shape
    ox, oy = float(origin[0]), float(origin[1])

    cols = np.arange(W, dtype=np.float32)
    rows = np.arange(H, dtype=np.float32)
    cc, rr = np.meshgrid(cols, rows)
    px = (ox + (cc + 0.5) * cell).astype(np.float32)
    py = (oy + (H - rr - 0.5) * cell).astype(np.float32)

    close_px = max(1, round(CLOSE_M_SEG / cell))
    try:
        import cv2 as _cv2
        erode_k = np.ones((3, 3), np.uint8)
        def _erode1(m): return _cv2.erode(m, erode_k, iterations=1)
    except ImportError:
        def _erode1(m): return ndi.binary_erosion(m).astype(np.uint8)

    out = np.zeros((H, W), np.uint8)

    for b in boxes:
        cx, cy  = float(b[0]), float(b[1])
        L, W_box, theta = float(b[2]), float(b[3]), float(b[4])
        th = np.deg2rad(theta)
        cos_t, sin_t = float(np.cos(th)), float(np.sin(th))
        dx = px - cx
        dy = py - cy
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy

        obb_in  = (np.abs(lx) <= L / 2 + margin_in) & (np.abs(ly) <= W_box / 2 + margin_in)
        obb_strict = (np.abs(lx) <= L / 2) & (np.abs(ly) <= W_box / 2)

        sub = (seg_mask > 0) & obb_in
        if not sub.any():
            continue

        if solid_fill:
            if int(((seg_mask > 0) & obb_strict).sum()) >= int(sparse_min_pts):
                out |= _erode1(obb_strict.astype(np.uint8))
            continue

        closed = _cv2_close(sub.astype(np.uint8), close_px)
        closed = (closed > 0) & obb_strict          # clip to strict OBB
        closed = _erode1(closed.astype(np.uint8))   # 1-px gap vs neighbours
        # No fill_and_draw here — it would undo the erosion-based separation.
        if int(closed.sum()) * cell ** 2 >= MIN_AREA_M2:
            out |= closed
        elif (sparse_box_fill
              and int(((seg_mask > 0) & obb_strict).sum()) >= int(sparse_min_pts)):
            # Too few returns to shape a footprint; the fitted box is the best
            # available estimate. Erode 1 px to keep neighbours separable.
            out |= _erode1(obb_strict.astype(np.uint8))

    if keep_unexplained:
        # Recover seg components no box touches (dilate `out` 2 px so slivers
        # at box borders do not count as separate unexplained components).
        closed_seg = fp_seg_close(seg_mask, cell)
        if closed_seg.any():
            reach = (ndi.binary_dilation(out.astype(bool), iterations=2)
                     if out.any() else np.zeros_like(out, bool))
            labels, _ = ndi.label(closed_seg > 0)
            component_sizes = np.bincount(labels.ravel())
            component_slices = ndi.find_objects(labels)
            kept_labels = np.flatnonzero(
                component_sizes[1:] * cell ** 2 >= MIN_AREA_M2
            ) + 1
            for lbl in kept_labels:
                region = component_slices[int(lbl) - 1]
                if region is None:
                    continue
                comp = labels[region] == lbl
                if (comp & reach[region]).any():
                    continue
                out[region] |= comp.astype(np.uint8)

    return out
