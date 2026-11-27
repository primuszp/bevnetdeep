"""bordereye — LiDAR point cloud processing: raster/voxel grids, terrain,
vehicle detection, and visualisation.

Submodules are imported lazily (PEP 562): accessing ``bordereye.RasterGrid``
only imports :mod:`bordereye.raster` and its dependencies, keeping
``import bordereye`` fast for scripts that use a small part of the library.
"""
from __future__ import annotations

from ._lazy import lazy_accessors

# Symbol -> defining submodule. Kept explicit (not introspected) so a typo
# fails loudly and "where does this come from?" stays greppable.
_LAZY: dict[str, str] = {
    # simple API
    "rasterize_las": "streaming",
    "voxelize_las": "streaming",
    # reusable spatial queries
    "SpatialIndex": "spatial",
    "points_in_polygon_2d": "spatial",
    "spatial_index": "spatial",
    # neighbourhood denoising and local geometry
    # candidates
    "GridCandidate": "candidates",
    "VehicleGeometryLimits": "candidates",
    "candidate_from_raster_component": "candidates",
    "candidate_from_voxel_component": "candidates",
    "extract_raster_candidates": "candidates",
    "extract_voxel_candidates": "candidates",
    "filter_vehicle_candidates": "candidates",
    "is_vehicle_like": "candidates",
    # lshape
    "LShapeResult": "lshape",
    "fit_lshape": "lshape",
    "fit_lshape_batch": "lshape",
    # detection
    "DetectedVehicle": "detection",
    "VehicleDetectionConfig": "detection",
    "detect_vehicles": "detection",
    # classify
    "VehicleClass": "classify",
    "VEHICLE_CLASSES": "classify",
    "classify_vehicle": "classify",
    "classify_vehicle_name": "classify",
    # clustering
    "fec_cluster": "clustering",
    "fec_cluster_labels": "clustering",
    "refine_cluster_labels": "clustering",
    # components
    "GridComponent": "components",
    "connected_components_sparse": "components",
    # ermapper
    "ERMapper": "ermapper",
    # export
    "save_candidates_dxf": "export",
    "save_candidates_geojson": "export",
    "save_candidates_shapefile": "export",
    "save_holes_shapefile": "export",
    "save_hulls_dxf": "export",
    "save_raster_points": "export",
    # features
    "CandidateFeatures": "features",
    "extract_features": "features",
    # source-independent point object features
    # geometry
    "OrientedBoundingBox2D": "geometry",
    "OrientedBoundingBox3D": "geometry",
    "MinimumVolumeBox3D": "geometry",
    "oriented_bounding_box_3d": "geometry",
    "calculate_mvbb": "geometry",
    "minimum_volume_box_3d": "geometry",
    "calculate_obb_corners": "geometry",
    "convex_hull_2d": "geometry",
    "hull_area_2d": "geometry",
    "polygon_area": "geometry",
    "convex_polygon_clip": "geometry",
    "polygon_iou": "geometry",
    "rectangularity_2d": "geometry",
    "oriented_bounding_box_2d": "geometry",
    "points_in_convex_hull": "geometry",
    "query_point_cloud_obb": "geometry",
    "random_downsample": "geometry",
    # holes
    "RasterHole": "holes",
    "find_raster_holes": "holes",
    # instances
    "DbscanConfig": "instances",
    "FecInstanceConfig": "instances",
    "InstanceBBoxLimits": "instances",
    "InstanceSegmentationStats": "instances",
    "InstancePostprocessConfig": "instances",
    "attach_instance_id_to_las": "instances",
    "collect_instance_hulls": "instances",
    "filter_instances_by_bbox": "instances",
    "labels_to_instance_id": "instances",
    "reindex_instance_id": "instances",
    "reindex_instance_id_by_centroid": "instances",
    "segment_instances_dbscan": "instances",
    "segment_instances_fec": "instances",
    "postprocess_instances": "instances",
    # labeling
    # io
    "DALES_DTYPE": "io",
    "read_dales_ply": "io",
    "read_las": "io",
    "read_las_data": "io",
    "read_las_full": "io",
    "iter_las_chunks": "io",
    "write_las": "io",
    # generic polygon SHP I/O
    "PolygonFeature": "vector_io",
    "read_shapefile": "vector_io",
    "write_shapefile": "vector_io",
    # streaming LAS/LAZ merge
    "MergeReport": "las_merge",
    "merge_las": "las_merge",
    "merge_pointclouds": "las_merge",
    # raster
    "RasterGrid": "raster",
    # rect_search
    "DEFAULT_TEMPLATES": "rect_search",
    "RectDetection": "rect_search",
    "RectSearchConfig": "rect_search",
    "VehicleTemplate": "rect_search",
    "detect_vehicles_rect": "rect_search",
    "instance_ids_from_rect": "rect_search",
    "nms_rectangles": "rect_search",
    "search_component": "rect_search",
    "search_points": "rect_search",
    # raster_ops
    "crop_to_bounds": "raster_ops",
    "fill_nodata": "raster_ops",
    "fill_nodata_nn": "raster_ops",
    "merge_rasters": "raster_ops",
    "pit_filter": "raster_ops",
    # terrain
    "build_dtm": "terrain",
    "csf_ground_mask": "terrain",
    "filter_by_height_above_ground": "terrain",
    "ground_mask": "terrain",
    "height_above_ground": "terrain",
    "progressive_ground_dtm": "terrain",
    "segment_ground": "terrain",
    # watershed
    "watershed_blobs": "watershed",
    # tiles
    "TileInfo": "tiles",
    "merge_tiles": "tiles",
    "list_las_files": "tiles",
    "read_tile_info": "tiles",
    "scan_tiles": "tiles",
    "summarize_tiles": "tiles",
    "export_tile_layout_csv": "tiles",
    "plot_tile_layout": "tiles",
    "tile_point_cloud": "tiles",
    "tiles_covering": "tiles",
    # vehicles
    # voxel
    "VoxelGrid": "voxel",
    "voxel_downsample": "voxel",
    "voxelize_points": "voxel",
    # hierarchical sparse voxels
    # viz
    # vehicle_pipeline
    "CandidateConfig": "vehicle_pipeline",
    "CandidateDiag": "vehicle_pipeline",
    "CandidateRecord": "vehicle_pipeline",
    "ExtractedObject": "vehicle_pipeline",
    "MatchConfig": "vehicle_pipeline",
    "TileResult": "vehicle_pipeline",
    "VehicleCandidate": "vehicle_pipeline",
    "VehicleTile": "vehicle_pipeline",
    "build_tile_dataset": "vehicle_pipeline",
    "compute_tile_hag": "vehicle_pipeline",
    "detect_candidates": "vehicle_pipeline",
    "detect_candidates_rect": "vehicle_pipeline",
    "diagnose_candidates": "vehicle_pipeline",
    "load_tile_las": "vehicle_pipeline",
    "load_tile_ply": "vehicle_pipeline",
    "ls_to_corners": "vehicle_pipeline",
    "match_and_extract": "vehicle_pipeline",
    "obb_point_indices": "vehicle_pipeline",
    "points_in_obb": "vehicle_pipeline",
    "to_obb_frame": "vehicle_pipeline",
}

__all__ = sorted(_LAZY)


__getattr__, __dir__ = lazy_accessors(__name__, _LAZY, globals())
