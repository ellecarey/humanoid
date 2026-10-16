import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from humanoid.types import SceneRecon
from humanoid.utils.geometry import depth_margin, project, sample_nearest
from humanoid.utils.logging import get_logger

log = get_logger(__name__)

MIN_FITNESS = 0.3
MAX_CORRECTION_M = 0.5
MAX_CORRECTION_DEG = 10.0


def align_clouds(
    source: np.ndarray, target: np.ndarray, voxels_m: list[float]
) -> tuple[np.ndarray, dict[str, float]]:
    """Robust coarse-to-fine point-to-plane ICP taking `source` onto `target`."""
    transform = np.eye(4)
    if len(source) < 100 or len(target) < 100 or not voxels_m:
        return transform, {"icp_fitness": 0.0}
    registration = o3d.pipelines.registration
    result = None
    for voxel in voxels_m:
        moving = _downsample(source, voxel)
        fixed = _downsample(target, voxel)
        fixed.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30)
        )
        estimation = registration.TransformationEstimationPointToPlane(
            registration.TukeyLoss(k=voxel * 2.0)
        )
        result = registration.registration_icp(
            moving,
            fixed,
            voxel * 3.0,
            transform,
            estimation,
            registration.ICPConvergenceCriteria(max_iteration=50),
        )
        transform = np.asarray(result.transformation)

    stats = {
        "icp_fitness": float(result.fitness),
        "icp_rmse_m": float(result.inlier_rmse),
        "icp_translation_m": float(np.linalg.norm(transform[:3, 3])),
        "icp_rotation_deg": _rotation_deg(transform[:3, :3]),
    }
    if (
        stats["icp_fitness"] < MIN_FITNESS
        or stats["icp_translation_m"] > MAX_CORRECTION_M
        or stats["icp_rotation_deg"] > MAX_CORRECTION_DEG
    ):
        log.warning("Rejecting ICP correction %s", stats)
        stats["icp_applied"] = 0.0
        return np.eye(4), stats
    stats["icp_applied"] = 1.0
    return transform, stats


def transform_scene(recon: SceneRecon, scene_id: str, transform: np.ndarray) -> None:
    """Move every camera of `scene_id` by the world transform `transform`."""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    for pose in recon.poses.values():
        if pose.scene_id != scene_id:
            continue
        new_rotation = pose.R @ rotation.T
        pose.t = pose.t - new_rotation @ translation
        pose.R = new_rotation


def self_consistent(
    points: np.ndarray,
    recon: SceneRecon,
    scene_id: str,
    margin_m: float,
    margin_rel: float,
    min_views: int,
) -> np.ndarray:
    """True where at least `min_views` frames of the point's own video see it."""
    if min_views <= 1 or len(points) == 0:
        return np.ones(len(points), dtype=bool)
    counts = np.zeros(len(points), dtype=int)
    for image_name, pose in recon.poses.items():
        if pose.scene_id != scene_id:
            continue
        depth_map = recon.depths[scene_id][image_name]
        pixels, z, keep = project(points, pose.R, pose.t, pose.K, depth_map.shape)
        observed = sample_nearest(depth_map, pixels)
        margin = depth_margin(z, margin_m, margin_rel)
        agree = (observed > 0) & (np.abs(observed - z) <= margin)
        counts[np.flatnonzero(keep)[agree]] += 1
    return counts >= min_views


def overlap_fraction(
    source: np.ndarray, target: np.ndarray, radius_m: float = 0.05, sample: int = 50000
) -> float:
    if len(source) == 0 or len(target) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    if len(source) > sample:
        source = source[rng.choice(len(source), sample, replace=False)]
    distance, _ = cKDTree(target).query(source, k=1)
    return float(np.mean(distance <= radius_m))


def _downsample(points: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return cloud.voxel_down_sample(voxel)


def _rotation_deg(rotation: np.ndarray) -> float:
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
