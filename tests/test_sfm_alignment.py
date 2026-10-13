import numpy as np

from humanoid.reconstruction.sfm import _apply_transform, _merge
from humanoid.types import CameraPose, SceneRecon
from humanoid.utils.geometry import transform_points


def _pose(scene_id: str, name: str) -> CameraPose:
    return CameraPose(
        scene_id=scene_id,
        image_name=name,
        R=np.eye(3),
        t=np.zeros(3),
        K=np.eye(3),
    )


def _scene(name: str, point: list[float]) -> SceneRecon:
    return SceneRecon(
        poses={name: _pose(name.split("/")[0], name)},
        points_xyz=np.array([point], dtype=float),
        point_image_ids=np.array([0], dtype=int),
        observations={name: np.array([[0.0, 10.0, 20.0]])},
    )


def test_apply_transform_keeps_points_in_front_of_the_same_camera():
    scene = _scene("v2/frame_000000.jpg", [1.0, 2.0, 3.0])
    pose = scene.poses["v2/frame_000000.jpg"]
    before = transform_points(scene.points_xyz, pose.R, pose.t)

    transform = np.array([
        [0.0, -1.0, 0.0, 0.5],
        [1.0, 0.0, 0.0, -0.25],
        [0.0, 0.0, 1.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    moved = _apply_transform(scene, transform)
    moved_pose = moved.poses["v2/frame_000000.jpg"]
    after = transform_points(moved.points_xyz, moved_pose.R, moved_pose.t)

    np.testing.assert_allclose(after, before, atol=1e-8)
    np.testing.assert_allclose(moved.points_xyz, [[-1.5, 0.75, 4.5]])


def test_merge_shifts_source_point_indices_and_pose_ids():
    target = _scene("v1/frame_000000.jpg", [0.0, 0.0, 1.0])
    source = _scene("v2/frame_000000.jpg", [1.0, 0.0, 1.0])

    merged = _merge(target, source)

    assert list(merged.poses) == ["v1/frame_000000.jpg", "v2/frame_000000.jpg"]
    np.testing.assert_allclose(merged.points_xyz, [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    np.testing.assert_array_equal(merged.point_image_ids, [0, 1])
    np.testing.assert_allclose(
        merged.observations["v1/frame_000000.jpg"],
        [[0.0, 10.0, 20.0]],
    )
    np.testing.assert_allclose(
        merged.observations["v2/frame_000000.jpg"],
        [[1.0, 10.0, 20.0]],
    )