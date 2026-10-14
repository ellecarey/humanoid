import numpy as np
import open3d as o3d

from humanoid.change_detection import apply_visibility, compute_geometric_diff
from humanoid.config import DiffConfig, VisibilityConfig
from humanoid.types import CameraPose, GeometricDiff, SceneRecon


def _cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return cloud


def _pose(scene_id: str, name: str, translation: list[float]) -> CameraPose:
    return CameraPose(
        scene_id=scene_id,
        image_name=name,
        R=np.eye(3),
        t=np.asarray(translation, dtype=float),
        K=np.array([
            [4.0, 0.0, 4.0],
            [0.0, 4.0, 4.0],
            [0.0, 0.0, 1.0],
        ]),
    )


def _recon(depth_m: float) -> tuple[SceneRecon, dict]:
    names = ("v2/a.jpg", "v2/b.jpg")
    translations = ([0.0, 0.0, 0.0], [-0.1, 0.0, 0.0])
    poses = {
        name: _pose("v2", name, translation)
        for name, translation in zip(names, translations)
    }
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[4, 4] = depth_m
    recon = SceneRecon(
        poses=poses,
        points_xyz=np.zeros((0, 3)),
        point_image_ids=np.zeros(0, dtype=int),
        observations={},
    )
    return recon, {"v2": {name: depth.copy() for name in names}}


def _candidate(point: list[float]) -> GeometricDiff:
    return GeometricDiff(
        points=np.array([point], dtype=float),
        source_scene=np.array(["v1"]),
        nn_distance=np.array([0.1]),
        candidate=np.array([True]),
    )


def test_distance_above_threshold_is_a_candidate():
    shared = np.array([[0.0, 0.0, 0.0]])
    shifted = np.array([[1.0, 0.0, 0.1]])
    diff = compute_geometric_diff(
        _cloud(np.vstack([shared, [[1.0, 0.0, 0.0]]])),
        _cloud(np.vstack([shared, shifted])),
        DiffConfig(change_distance_m=0.03),
    )

    by_source = {
        scene: diff.candidate[diff.source_scene == scene]
        for scene in ("v1", "v2")
    }
    assert by_source["v1"].tolist() == [False, True]
    assert by_source["v2"].tolist() == [False, True]


def test_free_space_in_two_views_is_changed():
    recon, depths = _recon(depth_m=3.0)
    labeled = apply_visibility(
        _candidate([0.0, 0.0, 2.0]),
        recon,
        depths,
        VisibilityConfig(depth_margin_m=0.05, min_confirming_views=2),
    )
    assert labeled.label.tolist() == ["changed"]


def test_occluder_in_front_stays_unknown():
    recon, depths = _recon(depth_m=1.0)
    labeled = apply_visibility(
        _candidate([0.0, 0.0, 2.0]),
        recon,
        depths,
        VisibilityConfig(depth_margin_m=0.05, min_confirming_views=2),
    )
    assert labeled.label.tolist() == ["unknown"]


def test_point_behind_the_cameras_stays_unknown():
    recon, depths = _recon(depth_m=3.0)
    labeled = apply_visibility(
        _candidate([0.0, 0.0, -1.0]),
        recon,
        depths,
        VisibilityConfig(depth_margin_m=0.05, min_confirming_views=2),
    )
    assert labeled.label.tolist() == ["unknown"]