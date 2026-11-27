from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import spatial


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2-D points, vertices in CCW order.

    Uses scipy's Qhull (C implementation); falls back to a pure-Python
    monotone chain when the input is degenerate (collinear points).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")

    # Qhull already ignores duplicate points, so the expensive row-wise
    # np.unique is deferred to the degenerate path it is actually needed for.
    if len(pts) > 2:
        try:
            return pts[spatial.ConvexHull(pts).vertices]
        except spatial.QhullError:
            pass
    pts = np.unique(pts, axis=0)
    if len(pts) <= 2:
        return pts
    return _convex_hull_2d_monotone(pts)


def _convex_hull_2d_monotone(pts: np.ndarray) -> np.ndarray:
    """Monotone-chain fallback; *pts* must be unique Nx2 with N > 2."""
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def polygon_area(vertices: np.ndarray) -> float:
    """Area of a simple polygon given as an (N, 2) vertex array (shoelace)."""
    v = np.asarray(vertices, dtype=np.float64)
    if len(v) < 3:
        return 0.0
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def convex_polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Intersect two convex polygons, independent of vertex winding.

    This dependency-free Sutherland-Hodgman path is tuned for the small
    polygons used by box fitting and non-maximum suppression.
    """
    output = [p for p in np.asarray(subject, dtype=np.float64)]
    boundary = np.asarray(clip, dtype=np.float64)
    if len(output) < 3 or len(boundary) < 3:
        return np.empty((0, 2), dtype=np.float64)
    # The clipping half-plane test below expects counter-clockwise edges.
    # Normalising once is cheaper and safer than branching for every point.
    signed_area = np.dot(boundary[:, 0], np.roll(boundary[:, 1], -1)) - np.dot(
        boundary[:, 1], np.roll(boundary[:, 0], -1)
    )
    if signed_area < 0:
        boundary = boundary[::-1]

    def inside(point, a, b):
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0

    def intersection(start, end, a, b):
        dc, dp = a - b, start - end
        n1 = a[0] * b[1] - a[1] * b[0]
        n2 = start[0] * end[1] - start[1] * end[0]
        denominator = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denominator) < 1e-12:
            return (start + end) * 0.5
        return np.array(((n1 * dp[0] - n2 * dc[0]) / denominator,
                         (n1 * dp[1] - n2 * dc[1]) / denominator))

    for i, a in enumerate(boundary):
        if not output:
            break
        b = boundary[(i + 1) % len(boundary)]
        source, output = output, []
        for j, current in enumerate(source):
            previous = source[j - 1]
            current_inside, previous_inside = inside(current, a, b), inside(previous, a, b)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current, a, b))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current, a, b))
    return np.asarray(output) if output else np.empty((0, 2), dtype=np.float64)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two convex polygons."""
    intersection = convex_polygon_clip(a, b)
    if len(intersection) < 3:
        return 0.0
    intersection_area = polygon_area(intersection)
    union_area = polygon_area(a) + polygon_area(b) - intersection_area
    return float(intersection_area / union_area) if union_area > 1e-12 else 0.0


def hull_area_2d(points: np.ndarray) -> float:
    """Convex-hull area of an Nx2 point set (0.0 for degenerate input)."""
    return polygon_area(convex_hull_2d(points))


def rect_corners_2d(
    cx: float,
    cy: float,
    length: float,
    width: float,
    theta: float,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    """Four CCW world-XY corners of a rotated rectangle.

    *theta* is the long-axis direction in radians. Corner order is
    (-l,-w), (+l,-w), (+l,+w), (-l,+w) in the box frame, matching
    :attr:`OrientedBoundingBox2D.vertices`.
    """
    ca, sa = np.cos(theta), np.sin(theta)
    hl, hw = length / 2.0, width / 2.0
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=dtype)
    rot = np.array([[ca, -sa], [sa, ca]], dtype=dtype)
    return local @ rot.T + np.array([cx, cy], dtype=dtype)


def to_box_frame_2d(
    xy: np.ndarray,
    cx: float,
    cy: float,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate world XY into the box frame centred at ``(cx, cy)``.

    Returns the along-length and along-width coordinates, so a point is inside
    the box when ``|along_length| <= length/2`` and ``|along_width| <= width/2``.
    """
    ca, sa = np.cos(theta), np.sin(theta)
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    return dx * ca + dy * sa, -dx * sa + dy * ca


@dataclass(frozen=True)
class OrientedBoundingBox2D:
    """Minimum-area rectangle with a canonical long-axis orientation.

    Vertices are counter-clockwise and edge ``0 -> 1`` is the length axis.
    ``rotation_deg`` is always in ``[0, 180)``.
    """

    vertices: np.ndarray
    center: tuple[float, float]
    length: float
    width: float
    rotation_deg: float
    area: float

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """Transform XY points into the box's length/width coordinate frame."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            raise ValueError("points must be an Nx2+ array")
        along_length, along_width = to_box_frame_2d(
            pts, self.center[0], self.center[1], np.deg2rad(self.rotation_deg)
        )
        return np.column_stack((along_length, along_width))

    def contains(self, points: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        """Boolean mask for points inside the rectangle."""
        local = self.to_local(points)
        eps = np.finfo(np.float64).eps * max(self.length, self.width, 1.0) * 8.0
        return (
            (np.abs(local[:, 0]) <= self.length / 2.0 + margin + eps)
            & (np.abs(local[:, 1]) <= self.width / 2.0 + margin + eps)
        )


def _make_obb_2d(
    center: np.ndarray,
    length: float,
    width: float,
    angle: float,
) -> OrientedBoundingBox2D:
    """Construct a canonical CCW box; *angle* is the long-axis direction."""
    angle %= np.pi
    c, s = np.cos(angle), np.sin(angle)
    axes = np.array([[c, s], [-s, c]], dtype=np.float64)
    local = np.array(
        [
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
        ]
    )
    vertices = local @ axes + center
    return OrientedBoundingBox2D(
        vertices=vertices,
        center=(float(center[0]), float(center[1])),
        length=float(length),
        width=float(width),
        rotation_deg=float(np.degrees(angle)),
        area=float(length * width),
    )


def oriented_bounding_box_2d(points: np.ndarray) -> OrientedBoundingBox2D:
    """Exact minimum-area rectangle around a finite 2-D point set."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite coordinates")

    return _oriented_bounding_box_2d_from_hull(convex_hull_2d(pts))


def _oriented_bounding_box_2d_from_hull(hull: np.ndarray) -> OrientedBoundingBox2D:
    """Fit an OBB to an already computed CCW convex hull."""
    if len(hull) == 1:
        return _make_obb_2d(hull[0], 0.0, 0.0, 0.0)
    if len(hull) == 2:
        direction = hull[1] - hull[0]
        return _make_obb_2d(
            (hull[0] + hull[1]) / 2.0,
            float(np.linalg.norm(direction)),
            0.0,
            float(np.arctan2(direction[1], direction[0])),
        )

    # Subtracting a local reference avoids precision loss for projected
    # cadastral/geodetic coordinates with large absolute offsets.
    reference = hull.mean(axis=0)
    centered = hull - reference
    edges = np.diff(np.vstack((hull, hull[0])), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    # Bound the projection matrix for hulls with many boundary samples.
    batch_size = max(1, 2_000_000 // len(centered))
    best_area = float("inf")
    best_values: tuple[float, float, float, float, float] | None = None
    for start in range(0, len(angles), batch_size):
        batch_angles = angles[start:start + batch_size]
        c, s = np.cos(batch_angles), np.sin(batch_angles)
        first = centered[:, 0, None] * c + centered[:, 1, None] * s
        second = -centered[:, 0, None] * s + centered[:, 1, None] * c
        lo1, hi1 = first.min(axis=0), first.max(axis=0)
        lo2, hi2 = second.min(axis=0), second.max(axis=0)
        areas = (hi1 - lo1) * (hi2 - lo2)
        local_best = int(np.argmin(areas))
        if areas[local_best] < best_area:
            best_area = float(areas[local_best])
            best_values = (
                float(batch_angles[local_best]),
                float(lo1[local_best]),
                float(hi1[local_best]),
                float(lo2[local_best]),
                float(hi2[local_best]),
            )
    assert best_values is not None
    angle, lo1, hi1, lo2, hi2 = best_values
    side1, side2 = hi1 - lo1, hi2 - lo2
    center_local = np.array([(lo1 + hi1) / 2, (lo2 + hi2) / 2])
    cb, sb = np.cos(angle), np.sin(angle)
    center = center_local @ np.array([[cb, sb], [-sb, cb]]) + reference
    if side1 >= side2:
        return _make_obb_2d(center, side1, side2, angle)
    return _make_obb_2d(center, side2, side1, angle + np.pi / 2)


@dataclass(frozen=True)
class HullFit2D:
    """Everything derivable from one convex-hull pass over a 2-D point set.

    Callers that need more than one of these values must reuse a single fit:
    recomputing the hull per value is the dominant cost in shape-based vehicle
    fitting.
    """

    hull: np.ndarray
    hull_area: float
    obb: OrientedBoundingBox2D
    rectangularity: float


def fit_hull_obb_2d(points: np.ndarray) -> HullFit2D:
    """Convex hull, its area, the minimum-area rectangle, and their ratio."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite coordinates")
    hull = convex_hull_2d(pts)
    obb = _oriented_bounding_box_2d_from_hull(hull)
    hull_area = polygon_area(hull)
    rectangularity = 1.0 if obb.area <= 1e-9 else min(1.0, hull_area / obb.area)
    return HullFit2D(hull=hull, hull_area=hull_area, obb=obb, rectangularity=rectangularity)


def rectangularity_2d(points: np.ndarray) -> tuple[float, OrientedBoundingBox2D]:
    """Return convex-hull/OBB area ratio and the fitted box in one hull pass."""
    fit = fit_hull_obb_2d(points)
    return fit.rectangularity, fit.obb


@dataclass(frozen=True)
class OrientedBoundingBox3D:
    """Gravity-aligned 3D oriented bounding box (yaw rotation about Z only).

    Corners 0-3 are the bottom face (CCW), 4-7 the top face.
    """

    center: tuple[float, float, float]
    length: float
    width: float
    height: float
    rotation_deg: float
    vertices: np.ndarray  # (8, 3)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def footprint(self) -> OrientedBoundingBox2D:
        """The 2D footprint of the box."""
        return OrientedBoundingBox2D(
            vertices=self.vertices[:4, :2].copy(),
            center=(self.center[0], self.center[1]),
            length=self.length,
            width=self.width,
            rotation_deg=self.rotation_deg,
            area=self.length * self.width,
        )

    def contains(self, points: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        """Boolean mask of the points inside the box (optionally grown by *margin*)."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be an Nx3+ array")
        xo, yo = to_box_frame_2d(
            pts, self.center[0], self.center[1], np.deg2rad(self.rotation_deg)
        )
        dz = pts[:, 2] - self.center[2]
        return (
            (np.abs(xo) <= self.length / 2.0 + margin)
            & (np.abs(yo) <= self.width / 2.0 + margin)
            & (np.abs(dz) <= self.height / 2.0 + margin)
        )


def oriented_bounding_box_3d(points: np.ndarray) -> OrientedBoundingBox3D:
    """Fit a gravity-aligned 3D OBB: minimal-area 2D OBB in XY + Z extent.

    For upright objects (vehicles, poles, furniture) this is the natural box;
    use :func:`calculate_mvbb` when a fully free orientation is required.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    if not np.isfinite(pts[:, :3]).all():
        raise ValueError("points must contain only finite coordinates")

    obb2 = oriented_bounding_box_2d(pts[:, :2])
    z_min = float(pts[:, 2].min())
    z_max = float(pts[:, 2].max())
    height = z_max - z_min
    bottom = np.column_stack((obb2.vertices, np.full(4, z_min)))
    top = np.column_stack((obb2.vertices, np.full(4, z_max)))
    return OrientedBoundingBox3D(
        center=(obb2.center[0], obb2.center[1], (z_min + z_max) / 2.0),
        length=obb2.length,
        width=obb2.width,
        height=height,
        rotation_deg=obb2.rotation_deg,
        vertices=np.vstack((bottom, top)),
    )


def random_downsample(points: np.ndarray, percentage: float, *, seed: int | None = None) -> np.ndarray:
    if not (0 < percentage <= 100):
        raise ValueError("percentage must be in (0, 100]")
    pts = np.asarray(points)
    if len(pts) == 0:
        return pts.copy()
    rng = np.random.default_rng(seed)
    n = max(1, int(len(pts) * percentage / 100.0))
    return pts[rng.choice(len(pts), size=n, replace=False)]


def calculate_obb_corners(
    width: float,
    length: float,
    height: float,
    rotation_angle: float,
    *,
    center: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Return the 8 corners of a 3-D oriented bounding box.

    By default the box sits with its bottom face at z=0 (ground convention).
    Pass *center* to override: the box will then be centred at that point.
    Corners 0-3 are the bottom face, 4-7 are the top face (consistent winding).

    Args:
        width: Extent along the Y axis before rotation (metres).
        length: Extent along the X axis before rotation (metres).
        height: Extent along the Z axis (metres).
        rotation_angle: Yaw rotation about Z in degrees.
        center: World-space centre of the box. When None (default) the bottom
            face lies on z=0, i.e. center = (0, 0, height/2).
    """
    dimensions = np.asarray((width, length, height), dtype=np.float64)
    if not np.isfinite(dimensions).all() or np.any(dimensions < 0):
        raise ValueError("width, length, and height must be finite and >= 0")
    if not np.isfinite(rotation_angle):
        raise ValueError("rotation_angle must be finite")
    hw, hl, hh = width / 2.0, length / 2.0, height / 2.0
    local = np.array(
        [
            [-hl, -hw, -hh],
            [ hl, -hw, -hh],
            [ hl,  hw, -hh],
            [-hl,  hw, -hh],
            [-hl, -hw,  hh],
            [ hl, -hw,  hh],
            [ hl,  hw,  hh],
            [-hl,  hw,  hh],
        ],
        dtype=np.float64,
    )
    angle = np.deg2rad(rotation_angle)
    cosine, sine = np.cos(angle), np.sin(angle)
    rot = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotated = local @ rot.T
    if center is None:
        # Legacy ground-plane convention: bottom face at z=0
        return rotated + np.array([0.0, 0.0, hh])
    center_array = np.asarray(center, dtype=np.float64)
    if center_array.shape != (3,) or not np.isfinite(center_array).all():
        raise ValueError("center must contain three finite coordinates")
    return rotated + center_array


# One chunk of the facet x point plane-distance matrix. 4M float64 = 32 MB.
_MAX_HULL_TEST_ENTRIES = 4_000_000


def points_in_convex_hull(points: np.ndarray, hull: spatial.ConvexHull) -> np.ndarray:
    """Boolean mask of the points inside a scipy convex hull.

    The facet-distance matrix is (n_facets, n_points), so it is evaluated in
    row blocks: materialising it whole costs gigabytes on large clouds.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must be an NxD array")
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1:]
    inside = np.ones(len(pts), dtype=bool)
    if len(pts) == 0:
        return inside
    block = max(1, _MAX_HULL_TEST_ENTRIES // max(len(normals), 1))
    for start in range(0, len(pts), block):
        chunk = pts[start:start + block]
        inside[start:start + block] = np.all(
            normals @ chunk.T + offsets <= 1e-10, axis=0
        )
    return inside


def query_point_cloud_obb(
    points: np.ndarray,
    center: tuple[float, float, float],
    width: float,
    length: float,
    height: float,
    rotation_angle: float,
    *,
    index=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select points in a ground-based 3-D OBB.

    Pass a reusable :class:`bordereye.SpatialIndex` when querying several
    boxes against the same cloud.
    """
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")
    if not np.isfinite(cloud[:, :3]).all():
        raise ValueError("points must contain only finite coordinates")
    corners_local = calculate_obb_corners(width, length, height, rotation_angle)
    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (3,) or not np.isfinite(center_arr).all():
        raise ValueError("center must contain three finite coordinates")
    if index is not None:
        if not index.indexes(points):
            raise ValueError("index must have been built from the supplied points")
        selected = index.oriented_box(
            center_arr[:2],
            length,
            width,
            rotation_angle,
            z_bounds=(float(center_arr[2]), float(center_arr[2] + height)),
        )
        return cloud[selected], corners_local + center_arr
    cloud_local = cloud[:, :3] - center_arr

    min_c = corners_local.min(axis=0)
    max_c = corners_local.max(axis=0)
    pre_mask = (
        (cloud_local[:, 0] >= min_c[0])
        & (cloud_local[:, 0] <= max_c[0])
        & (cloud_local[:, 1] >= min_c[1])
        & (cloud_local[:, 1] <= max_c[1])
        & (cloud_local[:, 2] >= min_c[2])
        & (cloud_local[:, 2] <= max_c[2])
    )
    pre_indices = np.flatnonzero(pre_mask)
    candidates = cloud_local[pre_indices]
    angle = np.deg2rad(rotation_angle)
    c, s = np.cos(angle), np.sin(angle)
    local_x = c * candidates[:, 0] + s * candidates[:, 1]
    local_y = -s * candidates[:, 0] + c * candidates[:, 1]
    eps = 1e-10
    inside = (
        (np.abs(local_x) <= length / 2.0 + eps)
        & (np.abs(local_y) <= width / 2.0 + eps)
        & (candidates[:, 2] >= -eps)
        & (candidates[:, 2] <= height + eps)
    )
    # Index once with the composed selection; cloud[pre_mask][inside] would
    # copy every pre-filtered row only to discard most of them again.
    return cloud[pre_indices[inside]], corners_local + center_arr


def calculate_mvbb(
    points: np.ndarray,
    *,
    max_face_orientations: int | None = 32,
) -> tuple[np.ndarray, float]:
    """Compute a tight minimum-volume-oriented 3-D bounding box.

    For every unique convex-hull face orientation, this solves the remaining
    in-plane rotation with an exact 2-D minimum-area rectangle. PCA axes are
    included as additional candidates. Collinear, coplanar, and repeated-point
    inputs have defined zero-volume results. By default the 32 largest hull
    facets are evaluated; pass ``max_face_orientations=None`` for exhaustive
    face evaluation.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must be an Nx3+ array")
    if len(pts) == 0:
        raise ValueError("points must not be empty")
    if max_face_orientations is not None and max_face_orientations <= 0:
        raise ValueError("max_face_orientations must be > 0 or None")
    xyz = np.unique(pts[:, :3], axis=0)
    if not np.isfinite(xyz).all():
        raise ValueError("points must contain only finite coordinates")

    reference = xyz.mean(axis=0)
    centered = xyz - reference
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        np.finfo(np.float64).eps * max(centered.shape) * singular[0]
        if singular[0]
        else 0.0
    )
    rank = int(np.count_nonzero(singular > tolerance))

    def fit_normal(normal: np.ndarray, cloud: np.ndarray) -> tuple[np.ndarray, float]:
        normal = normal / np.linalg.norm(normal)
        helper = np.eye(3)[int(np.argmin(np.abs(normal)))]
        axis_u = np.cross(helper, normal)
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        projected = np.column_stack((cloud @ axis_u, cloud @ axis_v))
        rectangle = oriented_bounding_box_2d(projected)
        normal_coordinate = cloud @ normal
        low = float(normal_coordinate.min())
        high = float(normal_coordinate.max())
        bottom = (
            rectangle.vertices[:, 0, None] * axis_u
            + rectangle.vertices[:, 1, None] * axis_v
            + low * normal
            + reference
        )
        top = bottom + (high - low) * normal
        return np.vstack((bottom, top)), rectangle.area * (high - low)

    if rank == 0:
        return np.repeat(reference[None, :], 8, axis=0), 0.0
    if rank < 3:
        return fit_normal(axes[-1], centered)

    hull = spatial.ConvexHull(xyz)
    hull_centered = xyz[hull.vertices] - reference
    candidate_normals = [*axes]
    seen: set[tuple[float, ...]] = set()
    simplices = xyz[hull.simplices]
    facet_area = np.linalg.norm(
        np.cross(simplices[:, 1] - simplices[:, 0], simplices[:, 2] - simplices[:, 0]),
        axis=1,
    )
    face_order = np.argsort(facet_area)[::-1]
    if max_face_orientations is not None:
        face_order = face_order[:max_face_orientations]
    for face_index in face_order:
        raw_normal = hull.equations[face_index, :3]
        normal = raw_normal / np.linalg.norm(raw_normal)
        first = np.flatnonzero(np.abs(normal) > 1e-12)
        if len(first) and normal[first[0]] < 0:
            normal = -normal
        key = tuple(np.round(normal, 7))
        if key not in seen:
            seen.add(key)
            candidate_normals.append(normal)

    best_corners: np.ndarray | None = None
    best_volume = float("inf")
    for normal in candidate_normals:
        corners, volume = fit_normal(normal, hull_centered)
        if volume < best_volume:
            best_corners, best_volume = corners, volume
    assert best_corners is not None
    return best_corners, float(best_volume)


@dataclass(frozen=True)
class MinimumVolumeBox3D:
    """Free-orientation 3-D box with orthonormal world-space axes."""

    vertices: np.ndarray
    center: tuple[float, float, float]
    axes: np.ndarray
    length: float
    width: float
    height: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def to_local(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points must be an Nx3+ array")
        return (pts[:, :3] - self.center) @ self.axes.T

    def contains(self, points: np.ndarray, *, margin: float = 0.0) -> np.ndarray:
        local = self.to_local(points)
        half = np.array([self.length, self.width, self.height]) / 2.0 + margin
        eps = np.finfo(np.float64).eps * max(self.length, self.width, self.height, 1.0) * 16
        return np.all(np.abs(local) <= half + eps, axis=1)


def minimum_volume_box_3d(
    points: np.ndarray,
    *,
    max_face_orientations: int | None = 32,
) -> MinimumVolumeBox3D:
    """Return the object-oriented result of :func:`calculate_mvbb`."""
    vertices, _ = calculate_mvbb(
        points,
        max_face_orientations=max_face_orientations,
    )
    edges = np.array(
        [vertices[1] - vertices[0], vertices[2] - vertices[1], vertices[4] - vertices[0]]
    )
    extents = np.linalg.norm(edges, axis=1)
    axes = np.zeros((3, 3), dtype=np.float64)
    nonzero = extents > np.finfo(np.float64).eps
    axes[nonzero] = edges[nonzero] / extents[nonzero, None]
    if not nonzero.all():
        centered = np.asarray(points, dtype=np.float64)[:, :3] - vertices.mean(axis=0)
        _, _, fallback = np.linalg.svd(centered, full_matrices=True)
        for axis_index in np.flatnonzero(~nonzero):
            candidate = fallback[axis_index]
            for previous in range(axis_index):
                candidate -= np.dot(candidate, axes[previous]) * axes[previous]
            norm = np.linalg.norm(candidate)
            if norm > np.finfo(np.float64).eps:
                axes[axis_index] = candidate / norm
        if np.linalg.norm(axes[2]) <= np.finfo(np.float64).eps:
            axes[2] = np.cross(axes[0], axes[1])
    return MinimumVolumeBox3D(
        vertices=vertices,
        center=tuple(vertices.mean(axis=0)),
        axes=axes,
        length=float(extents[0]),
        width=float(extents[1]),
        height=float(extents[2]),
    )
