import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from humanoid.config import ClusterConfig
from humanoid.types import ChangeRegion, LabeledDiff

FLOOR_DISTANCE_M = 0.02


def cluster_changes(diff: LabeledDiff, cfg: ClusterConfig) -> list[ChangeRegion]:
    changed = diff.label == "changed"
    floor = _floor_plane(diff.points[diff.label == "unchanged"])
    regions: list[ChangeRegion] = []

    for scene_id, change_type in (("v1", "removed"), ("v2", "added")):
        points = diff.points[changed & (diff.source_scene == scene_id)]
        if len(points) == 0:
            continue
        labels = DBSCAN(
            eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples
        ).fit_predict(points)
        for label in sorted(set(labels) - {-1}):
            cluster = points[labels == label]
            region = _region(len(regions), scene_id, change_type, cluster)
            if region is None or not _keep(region, floor, cfg):
                continue
            regions.append(region)
    return regions


def unknown_regions(diff: LabeledDiff, cfg: ClusterConfig) -> list[ChangeRegion]:
    unknown = diff.label == "unknown"
    regions: list[ChangeRegion] = []

    for scene_id in ("v1", "v2"):
        points = diff.points[unknown & (diff.source_scene == scene_id)]
        if len(points) == 0:
            continue
        labels = DBSCAN(
            eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples
        ).fit_predict(points)
        for label in sorted(set(labels) - {-1}):
            cluster = points[labels == label]
            region = _region(len(regions), scene_id, "unknown", cluster)
            if region is None or not _keep(region, None, cfg):
                continue
            regions.append(region)
    return regions


def _floor_plane(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 3:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    plane, _ = cloud.segment_plane(
        distance_threshold=FLOOR_DISTANCE_M,
        ransac_n=3,
        num_iterations=1000,
    )
    return np.asarray(plane, dtype=float)


def _region(
    region_id: int, scene_id: str, change_type: str, points: np.ndarray
) -> ChangeRegion | None:
    if len(points) < 4:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    box = cloud.get_oriented_bounding_box()
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id=scene_id,
        points=points,
        centroid=np.asarray(box.center, dtype=float),
        extent=np.asarray(box.extent, dtype=float),
        obb_rotation=np.asarray(box.R, dtype=float),
    )


def _keep(region: ChangeRegion, floor: np.ndarray | None, cfg: ClusterConfig) -> bool:
    longest = float(np.max(region.extent))
    if (
        len(region.points) < cfg.min_points
        or longest < cfg.min_extent_m
        or longest > cfg.max_extent_m
    ):
        return False
    if floor is None:
        return True
    normal = floor[:3]
    height = abs(float(normal @ region.centroid + floor[3])) / np.linalg.norm(normal)
    return height >= cfg.min_height_above_floor_m
