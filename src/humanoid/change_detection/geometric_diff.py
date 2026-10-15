import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from humanoid.config import DiffConfig
from humanoid.types import GeometricDiff


def compute_geometric_diff(
    cloud_v1: o3d.geometry.PointCloud,
    cloud_v2: o3d.geometry.PointCloud,
    cfg: DiffConfig,
) -> GeometricDiff:
    points_v1 = _xyz(cloud_v1)
    points_v2 = _xyz(cloud_v2)
    distance_v1 = _nn_distance(points_v1, points_v2)
    distance_v2 = _nn_distance(points_v2, points_v1)

    points = np.vstack([points_v1, points_v2])
    source_scene = np.array(["v1"] * len(points_v1) + ["v2"] * len(points_v2))
    nn_distance = np.concatenate([distance_v1, distance_v2])
    return GeometricDiff(
        points=points,
        source_scene=source_scene,
        nn_distance=nn_distance,
        candidate=nn_distance > cfg.change_distance_m,
    )


def _xyz(cloud) -> np.ndarray:
    points = cloud if isinstance(cloud, np.ndarray) else np.asarray(cloud.points)
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3))
    return points.reshape(-1, 3)


def _nn_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(query) == 0:
        return np.zeros(0)
    if len(reference) == 0:
        return np.full(len(query), np.inf)
    return cKDTree(reference).query(query, k=1)[0]
