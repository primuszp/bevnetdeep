from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MAX_DENSE_COMPONENT_CELLS = 10_000_000
_MAX_DENSE_TO_OCCUPIED_RATIO = 16


@dataclass(frozen=True)
class GridComponent:
    id: int
    coords: np.ndarray

    @property
    def size(self) -> int:
        return len(self.coords)

    @property
    def min_coord(self) -> np.ndarray:
        return self.coords.min(axis=0)

    @property
    def max_coord(self) -> np.ndarray:
        return self.coords.max(axis=0)


def connected_components_sparse(
    coords: np.ndarray,
    *,
    diagonal: bool = False,
    min_size: int = 1,
) -> list[GridComponent]:
    """Connected components on sparse integer grid coordinates.

    Compact inputs use fast dense labeling; wide sparse inputs switch to a
    neighbour graph so memory scales with occupied cells rather than extent.
    Supports 2D (raster) and 3D (voxel) coordinate arrays.

    Args:
        coords: NxD integer grid coordinates (D=2 for raster, D=3 for voxel).
        diagonal: False gives face connectivity (4-neighbour in 2D, 6 in 3D).
            True gives full connectivity (8-neighbour in 2D, 26 in 3D).
        min_size: components smaller than this are omitted.
    """
    import scipy.ndimage as ndi

    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2:
        raise ValueError("coords must be an NxD array")
    if min_size <= 0:
        raise ValueError("min_size must be > 0")
    if len(coords) == 0:
        return []

    ndim = coords.shape[1]
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D coordinate arrays are supported")

    # Shift coords to zero-origin to keep the dense array as small as possible
    origin = coords.min(axis=0)
    shifted = coords - origin
    grid_shape = tuple((shifted.max(axis=0) + 1).tolist())  # (X,) or (X,Y) or (X,Y,Z)

    # Python integers avoid int64 overflow for extremely wide sparse extents.
    dense_cells = math.prod(grid_shape)
    use_dense = (
        dense_cells <= _MAX_DENSE_COMPONENT_CELLS
        and dense_cells <= _MAX_DENSE_TO_OCCUPIED_RATIO * len(coords)
    )
    if use_dense:
        # Dense labeling is fastest for compact components.
        dense_shape = tuple(reversed(grid_shape))
        dense = np.zeros(dense_shape, dtype=bool)
        dense_index = tuple(shifted[:, dim] for dim in range(ndim - 1, -1, -1))
        dense[dense_index] = True
        structure = ndi.generate_binary_structure(ndim, ndim if diagonal else 1)
        labeled, _ = ndi.label(dense, structure=structure)
        occupied_labels = labeled[dense_index]
    else:
        # A KD-tree discovers only nearby occupied cells, so memory scales with
        # occupied coordinates rather than a potentially enormous empty extent.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        pairs = cKDTree(coords.astype(np.float64, copy=False)).query_pairs(
            np.sqrt(ndim),
            output_type="ndarray",
        )
        if len(pairs):
            delta = np.abs(coords[pairs[:, 0]] - coords[pairs[:, 1]])
            adjacent = (
                np.max(delta, axis=1) <= 1
                if diagonal
                else np.sum(delta, axis=1) <= 1
            )
            pairs = pairs[adjacent]
        graph = csr_matrix(
            (np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
            shape=(len(coords), len(coords)),
        )
        _, zero_based = connected_components(graph, directed=False, return_labels=True)
        occupied_labels = zero_based + 1

    counts = np.bincount(occupied_labels)
    keep = counts[occupied_labels] >= min_size
    kept_labels = occupied_labels[keep]
    kept_coords = coords[keep]
    if len(kept_coords) == 0:
        return []

    order = np.argsort(kept_labels, kind="stable")
    sorted_labels = kept_labels[order]
    sorted_coords = kept_coords[order]
    starts = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    groups = np.split(sorted_coords, starts[1:])
    return [GridComponent(id=i, coords=group) for i, group in enumerate(groups, start=1)]
