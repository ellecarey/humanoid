import numpy as np

from humanoid.reconstruction.feedforward import _cover_crop_map
from humanoid.reconstruction.refine import (
    align_clouds,
    self_consistent,
    transform_scene,
)
from humanoid.types import CameraPose, SceneRecon
from humanoid.utils.geometry import backproject, project, to_frame_pixels


def _room(rng) -> np.ndarray:
    """Floor, two walls and a box: enough structure to pin all six degrees of freedom."""
    floor = np.c_[rng.uniform(0, 3, 4000), rng.uniform(0, 3, 4000), np.zeros(4000)]
    wall_x = np.c_[np.zeros(3000), rng.uniform(0, 3, 3000), rng.uniform(0, 2, 3000)]
    wall_y = np.c_[rng.uniform(0, 3, 3000), np.zeros(3000), rng.uniform(0, 2, 3000)]
    box = np.c_[
        rng.uniform(1, 1.5, 1500), rng.uniform(1, 1.5, 1500), np.full(1500, 0.5)
    ]
    return np.vstack([floor, wall_x, wall_y, box])


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_icp_recovers_a_small_offset():
    rng = np.random.default_rng(0)
    target = _room(rng)
    rotation = _rotation_z(2.0)
    offset = np.array([0.06, -0.04, 0.03])
    source = (target - 1.5) @ rotation.T + 1.5 + offset

    transform, stats = align_clouds(source, target, [0.1, 0.05, 0.02])

    aligned = source @ transform[:3, :3].T + transform[:3, 3]
    assert stats["icp_applied"] == 1.0
    assert np.abs(aligned - target).max() < 0.01


def test_transform_scene_keeps_projections_consistent():
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    pose = CameraPose("v2", "v2/a.jpg", _rotation_z(10.0), np.array([0.1, 0.2, 3.0]), K)
    recon = SceneRecon(poses={"v2/a.jpg": pose}, depths={}, confidence={})
    point = np.array([[0.2, -0.1, 0.5]])
    before, _, _ = project(point, pose.R, pose.t, K)
    transform = np.eye(4)
    transform[:3, :3] = _rotation_z(5.0)
    transform[:3, 3] = [0.3, -0.2, 0.1]

    transform_scene(recon, "v2", transform)

    moved = point @ transform[:3, :3].T + transform[:3, 3]
    after, _, _ = project(moved, pose.R, pose.t, K)
    np.testing.assert_allclose(after, before, atol=1e-9)


def test_self_consistency_drops_points_seen_by_one_frame():
    K = np.array([[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]])
    names = ("v1/a.jpg", "v1/b.jpg")
    poses = {name: CameraPose("v1", name, np.eye(3), np.zeros(3), K) for name in names}
    surface = np.full((8, 8), 2.0, dtype=np.float32)
    floater = surface.copy()
    floater[4, 4] = 1.0
    recon = SceneRecon(
        poses=poses,
        depths={"v1": {"v1/a.jpg": surface, "v1/b.jpg": floater}},
        confidence={},
    )
    points = np.vstack(
        [
            backproject([4], [4], [2.0], K, np.eye(3), np.zeros(3)),
            backproject([4], [4], [1.0], K, np.eye(3), np.zeros(3)),
        ]
    )

    keep = self_consistent(points, recon, "v1", 0.04, 0.0, 2)

    assert keep.tolist() == [False, False]
    floater[4, 4] = 2.0
    keep = self_consistent(points, recon, "v1", 0.04, 0.0, 2)
    assert keep.tolist() == [True, False]


def test_cover_crop_map_returns_full_resolution_pixels():
    frame_map = _cover_crop_map(1280, 720, 518, 294)
    scale = frame_map[0]

    corners = to_frame_pixels(np.array([[0.0, 0.0], [517.0, 293.0]]), frame_map)

    assert scale == frame_map[1]
    assert abs(corners[0, 0] - frame_map[2] / scale) < 1e-6
    assert corners[1, 0] < 1280 and corners[1, 1] < 720
    assert abs(corners[1, 1] - 293.0 / scale) < 1e-6
