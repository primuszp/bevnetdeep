"""Point cloud clustering algorithms.

Provides Fast Euclidean Clustering (FEC) as a faster alternative to DBSCAN
for point clouds with roughly uniform density, and a shape-guided cluster
splitter (refine_cluster_labels) that corrects the chaining artefacts
inherent to Euclidean clustering.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .geometry import rectangularity_2d, to_box_frame_2d

# Full kNN arrays are fastest because SciPy can find graph components in C.
# Above this edge budget we retain the bounded-memory streaming union path.
_MAX_SPARSE_FEC_EDGES = 16_000_000
_MAX_FEC_QUERY_ENTRIES = 2_000_000


def _relabel_by_size(
    tags: np.ndarray,
    min_cluster_size: int,
    max_cluster_size: int | None,
) -> np.ndarray:
    """Drop out-of-size components and renumber the survivors to 1..K.

    *tags* are 1-indexed component ids with 0 reserved for noise.
    """
    counts = np.bincount(tags)
    valid_cluster = counts >= min_cluster_size
    valid_cluster[0] = False
    if max_cluster_size is not None:
        valid_cluster &= counts <= max_cluster_size
    mapping = np.zeros(len(counts), dtype=np.int32)
    mapping[valid_cluster] = np.arange(1, int(valid_cluster.sum()) + 1, dtype=np.int32)
    return mapping[tags]


def fec_cluster(
    points: np.ndarray,
    radius: float,
    *,
    min_cluster_size: int = 1,
    max_cluster_size: int | None = None,
    use_3d: bool = True,
    max_neighbors: int = 50,
) -> np.ndarray:
    """Fast Euclidean Clustering via KDTree region growing.

    Each seed point that has not yet been assigned expands outward using
    radius-ball queries. This is an order of magnitude faster than DBSCAN
    for large, uniformly-spaced point clouds because it avoids the quadratic
    neighbourhood comparison.

    Args:
        points: Nx3+ point array. Only the first 2 or 3 columns are used.
        radius: Maximum Euclidean distance between neighbouring points (metres).
        min_cluster_size: Clusters with fewer points are relabelled 0 (noise).
        max_cluster_size: Clusters larger than this are labelled as noise.
            None means unlimited.
        use_3d: Use XYZ distance (True) or XY-only distance (False).
        max_neighbors: Maximum nearest neighbours considered per point. This
            bound is what keeps FEC fast and memory-efficient on dense clouds.

    Returns:
        Integer label array of length N. Labels are 1-indexed; 0 means noise.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < (3 if use_3d else 2):
        raise ValueError(f"points must be Nx{'3' if use_3d else '2'}+ array")
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be > 0")

    search_pts = pts[:, :3] if use_3d else pts[:, :2]
    n = len(search_pts)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    k = min(max_neighbors + 1, n)
    # Sliding-midpoint construction and 32-point leaves won the mixed dense /
    # sparse benchmark. The radius bound avoids finding distant neighbours
    # that will immediately be discarded (about 5x faster on sparse clouds).
    tree = cKDTree(search_pts, leafsize=32, balanced_tree=False)

    if n * k <= _MAX_SPARSE_FEC_EDGES:
        # Build CSR directly from the capped kNN matrix.  Avoiding COO row
        # indices saves one n*k integer allocation and conversion pass.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        # Build CSR in bounded query batches. Preallocation avoids retaining
        # float distances or duplicating the final integer edge array.
        indices = np.empty(n * k, dtype=np.int32 if n <= np.iinfo(np.int32).max else np.int64)
        indptr = np.empty(n + 1, dtype=np.int64)
        indptr[0] = 0
        edge_count = 0
        query_rows = max(1, _MAX_FEC_QUERY_ENTRIES // k)
        for start in range(0, n, query_rows):
            stop = min(start + query_rows, n)
            distances, neighbours = tree.query(
                search_pts[start:stop],
                k=k,
                workers=-1,
                distance_upper_bound=radius,
            )
            if k == 1:
                distances = distances[:, None]
                neighbours = neighbours[:, None]
            within = np.isfinite(distances)
            degree = within.sum(axis=1, dtype=np.int64)
            count = int(degree.sum())
            indices[edge_count:edge_count + count] = neighbours[within]
            np.cumsum(degree, out=indptr[start + 1:stop + 1])
            indptr[start + 1:stop + 1] += edge_count
            edge_count += count
        indices = indices[:edge_count]
        graph = csr_matrix(
            (np.ones(len(indices), dtype=np.uint8), indices, indptr),
            shape=(n, n),
        )
        _, component = connected_components(graph, directed=False, return_labels=True)
        tags = component.astype(np.int32, copy=False) + 1
        return _relabel_by_size(tags, min_cluster_size, max_cluster_size)

    # Vectorised union-find over radius edges. Each kNN chunk is merged to a
    # fixed point with pointer jumping, so no per-edge Python loop runs while
    # peak memory still only holds one chunk of edges plus the parent array.
    parent = np.arange(n, dtype=np.int64)

    def _flatten(parent: np.ndarray) -> np.ndarray:
        """Point every entry directly at its component root."""
        while True:
            grandparent = parent[parent]
            if np.array_equal(grandparent, parent):
                return parent
            parent = grandparent

    # Query the kNN in chunks to bound peak memory (a single query on a
    # multi-million-point cloud allocates n*k float64 + int64 at once).
    chunk = max(1, min(n, _MAX_FEC_QUERY_ENTRIES // k))
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        distances, neighbours = tree.query(
            search_pts[start:stop],
            k=k,
            workers=-1,
            distance_upper_bound=radius,
        )
        if k == 1:
            distances = distances[:, None]
            neighbours = neighbours[:, None]
        within = np.isfinite(distances)
        left = np.repeat(
            np.arange(start, stop, dtype=np.int64), within.sum(axis=1, dtype=np.int64)
        )
        right = neighbours[within].astype(np.int64, copy=False)
        # Hook the higher root onto the lower one and repeat until this chunk's
        # edges connect nothing new. Unions only ever coarsen, so edges settled
        # by an earlier chunk stay settled.
        while True:
            root_left, root_right = parent[left], parent[right]
            active = root_left != root_right
            if not active.any():
                break
            high = np.maximum(root_left[active], root_right[active])
            low = np.minimum(root_left[active], root_right[active])
            np.minimum.at(parent, high, low)
            parent = _flatten(parent)

    _, tags = np.unique(parent, return_inverse=True)
    tags = tags.astype(np.int32) + 1  # 1-indexed, 0 reserved for noise

    return _relabel_by_size(tags, min_cluster_size, max_cluster_size)


def fec_cluster_labels(
    points: np.ndarray,
    radius: float,
    *,
    min_cluster_size: int = 1,
    max_cluster_size: int | None = None,
    use_3d: bool = True,
    max_neighbors: int = 50,
) -> tuple[np.ndarray, int]:
    """Like fec_cluster but also returns the number of clusters found.

    Returns:
        (labels, n_clusters) where labels are 1-indexed (0=noise).
    """
    labels = fec_cluster(
        points,
        radius,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        use_3d=use_3d,
        max_neighbors=max_neighbors,
    )
    n_clusters = int(labels.max()) if len(labels) > 0 else 0
    return labels, n_clusters


def refine_cluster_labels(
    points: np.ndarray,
    labels: np.ndarray,
    *,
    min_rectangularity: float = 0.75,
    max_length: float | None = None,
    max_width: float | None = None,
    min_split_points: int = 40,
    min_split_fraction: float = 0.1,
    model_penalty: float = 0.08,
) -> np.ndarray:
    """Split loosely packed (chained) clusters using a convex-hull shape test.

    Euclidean clustering (FEC, DBSCAN) merges objects connected by any bridge
    of points within the clustering radius. Such merged clusters do not fill
    their oriented bounding box tightly: the convex-hull area to OBB area
    ratio (*rectangularity*) drops well below 1. This function recursively
    cuts every suspicious cluster along its OBB principal axes (largest gap
    plus balanced quantile cuts) and keeps a cut only when the point-weighted
    rectangularity of the two parts beats the merged cluster's by more than
    *model_penalty* — so compact clusters are never touched.

    Args:
        points: Nx2+ array; only XY is used for the shape test.
        labels: integer labels (0 = noise), e.g. from :func:`fec_cluster`.
        min_rectangularity: clusters at or above this hull/OBB ratio are kept
            as-is (unless they violate a size gate below).
        max_length: optional hard gate — clusters with a longer OBB are
            always candidates for splitting (e.g. vehicle max length).
        max_width: optional hard gate on OBB width; catches side-by-side
            objects that merge into a wide but still rectangular cluster.
        min_split_points: minimum points on each side of an accepted cut.
        min_split_fraction: minimum fraction of the cluster on each side.
        model_penalty: quality margin a split must exceed (MDL cost of the
            extra cluster); raise it to split less eagerly.

    Returns:
        New label array (0 stays noise). Original ids are preserved for
        untouched clusters; split parts get fresh ids after the current max.
    """
    pts = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be an Nx2+ array")
    if labels.shape != (len(pts),):
        raise ValueError("labels must be a 1D array with one label per point")
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("labels must contain non-negative integers")
    if not np.isfinite(pts[:, :2]).all():
        raise ValueError("points must contain only finite XY coordinates")

    # int64 prevents fresh split IDs from wrapping narrow/unsigned input dtypes.
    out = labels.astype(np.int64, copy=True)
    next_id = int(out.max()) + 1 if len(out) else 1
    positive_index = np.flatnonzero(labels > 0)
    order = np.argsort(labels[positive_index])
    sorted_index = positive_index[order]
    sorted_labels = labels[sorted_index]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    stops = np.r_[starts[1:], len(sorted_index)]
    stack = [sorted_index[start:stop] for start, stop in zip(starts, stops, strict=True)]
    rng = np.random.default_rng(0)
    max_eval_points = 4000  # shape decisions run on a subsample; cuts apply to all

    while stack:
        idx = stack.pop()
        if len(idx) < 2 * min_split_points:
            continue
        xy = pts[idx, :2]
        # Decisions (hull ratios, cut search) run on a bounded subsample so a
        # 100k-point cluster costs the same as a 4k one; the accepted cut is a
        # threshold along an OBB axis and is applied to every point.
        if len(idx) > max_eval_points:
            sub = xy[rng.choice(len(idx), size=max_eval_points, replace=False)]
        else:
            sub = xy
        rect, obb = rectangularity_2d(sub)
        oversized = ((max_length is not None and obb.length > max_length)
                     or (max_width is not None and obb.width > max_width))
        if rect >= min_rectangularity and not oversized:
            continue

        # Rotate into the OBB frame; cuts are tried along both principal axes.
        angle = np.deg2rad(obb.rotation_deg)
        axes_sub = to_box_frame_2d(sub, obb.center[0], obb.center[1], angle)
        axes_full = to_box_frame_2d(xy, obb.center[0], obb.center[1], angle)

        def _valid_cut(
            axis_i: int,
            threshold: float,
            full_axes: tuple[np.ndarray, np.ndarray] = axes_full,
            point_count: int = len(idx),
        ) -> bool:
            n_left = int((full_axes[axis_i] <= threshold).sum())
            small = min(n_left, point_count - n_left)
            return small >= min_split_points and small >= min_split_fraction * point_count

        chosen: tuple[int, float] | None = None  # (axis index, threshold)

        if rect < min_rectangularity:
            # Chained / L-shaped cluster: accept the cut whose parts fill their
            # own boxes demonstrably better than the merged cluster does.
            best: tuple[float, int, float] | None = None
            for axis_i, values in enumerate(axes_sub):
                sv = np.sort(values)
                # Candidate thresholds: a balanced quantile sweep plus the
                # largest spatial gap of the FULL cluster (real gaps between
                # objects vanish in a subsample).
                thresholds = {float(np.quantile(sv, q))
                              for q in np.arange(0.15, 0.86, 0.05)}
                fv = np.sort(axes_full[axis_i])
                fgaps = np.diff(fv)
                if len(fgaps):
                    g = int(np.argmax(fgaps))
                    thresholds.add(float((fv[g] + fv[g + 1]) / 2.0))
                for threshold in thresholds:
                    if not _valid_cut(axis_i, threshold):
                        continue
                    left = values <= threshold
                    n_left = int(left.sum())
                    if n_left == 0 or n_left == len(values):
                        continue
                    ra, _ = rectangularity_2d(sub[left])
                    rb, _ = rectangularity_2d(sub[~left])
                    quality = (n_left * ra + (len(values) - n_left) * rb) / len(values)
                    if best is None or quality > best[0]:
                        best = (quality, axis_i, threshold)
            if best is not None and best[0] > rect + model_penalty:
                chosen = (best[1], best[2])

        if chosen is None and oversized:
            # Rectangular but physically too large (e.g. side-by-side objects):
            # the shape test cannot justify a cut, so cut only at a genuine
            # spatial gap — this keeps legitimately large solid objects whole.
            # Gaps are measured on the FULL cluster (a subsample widens gaps).
            spacing = float(np.sqrt(obb.area / max(len(idx), 1)))  # ~mean pt spacing
            best_gap = 2.5 * spacing  # a cut must open a genuine gap
            for axis_i, values in enumerate(axes_full):
                sv = np.sort(values)
                gaps = np.diff(sv)
                if not len(gaps):
                    continue
                g = int(np.argmax(gaps))
                threshold = float((sv[g] + sv[g + 1]) / 2.0)
                if gaps[g] >= best_gap and _valid_cut(axis_i, threshold):
                    best_gap = float(gaps[g])
                    chosen = (axis_i, threshold)

        if chosen is None:
            continue  # two clusters are not demonstrably better -> keep merged

        left_mask = axes_full[chosen[0]] <= chosen[1]
        part_a, part_b = idx[left_mask], idx[~left_mask]
        out[part_a] = next_id
        out[part_b] = next_id + 1
        next_id += 2
        stack.extend((part_a, part_b))  # parts may need further splitting

    # Compact ids to 1..K, keeping noise at 0.
    # Compress only labels that are present. A dense lookup by max(label)
    # could allocate gigabytes for perfectly valid sparse external IDs.
    result = np.zeros(len(out), dtype=np.int32)
    positive = out > 0
    if positive.any():
        _, inverse = np.unique(out[positive], return_inverse=True)
        result[positive] = inverse.astype(np.int32, copy=False) + 1
    return result
