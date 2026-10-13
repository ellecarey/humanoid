import numpy as np
import open3d as o3d

VOXEL_SIZE = 0.05  # metres; sparse SfM points, not the 1.5 cm fusion voxels


def _cloud(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    cloud = cloud.voxel_down_sample(voxel_size)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    return cloud


def register_clouds(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    voxel_size = VOXEL_SIZE
    source_cloud = _cloud(source, voxel_size)
    target_cloud = _cloud(target, voxel_size)
    if len(source_cloud.points) < 10 or len(target_cloud.points) < 10:
        raise RuntimeError("Too few points to register the two scenes")

    radius = voxel_size * 5.0
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_cloud, search)
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_cloud, search)

    distance = voxel_size * 1.5
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud,
        target_cloud,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
    )
    icp = o3d.pipelines.registration.registration_icp(
        source_cloud,
        target_cloud,
        voxel_size * 0.4,
        ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return np.asarray(icp.transformation, dtype=float)