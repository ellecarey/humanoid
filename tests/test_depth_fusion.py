from pathlib import Path

import cv2
import numpy as np

from humanoid.config import FusionConfig
from humanoid.reconstruction.depth import _align
from humanoid.reconstruction.fusion import fuse_tsdf
from humanoid.types import CameraPose, Frame, SceneRecon


def _pose(width: int, height: int) -> CameraPose:
    return CameraPose(
        scene_id="v1",
        image_name="v1/frame_000000.jpg",
        R=np.eye(3),
        t=np.zeros(3),
        K=np.array([
            [50.0, 0.0, width / 2],
            [0.0, 50.0, height / 2],
            [0.0, 0.0, 1.0],
        ]),
    )


def test_align_recovers_scale_and_shift():
    prediction = np.zeros((6, 6), dtype=np.float32)
    prediction[3, 3] = 1.0
    prediction[3, 4] = 2.0
    observations = np.array([
        [0.0, 3.0, 3.0],
        [1.0, 3.5, 3.0],
    ])
    points = np.array([
        [0.0, 0.0, 2.0],
        [0.2, 0.0, 4.0],
    ])

    aligned, reason = _align(prediction, observations, points, _pose(6, 6), min_points=2)

    assert reason is None
    np.testing.assert_allclose(aligned[3, 3], 2.0, atol=1e-5)
    np.testing.assert_allclose(aligned[3, 4], 4.0, atol=1e-5)


def test_align_reports_too_few_points():
    prediction = np.ones((6, 6), dtype=np.float32)
    observations = np.array([[0.0, 3.0, 3.0]])
    points = np.array([[0.0, 0.0, 2.0]])

    aligned, reason = _align(prediction, observations, points, _pose(6, 6), min_points=2)

    assert aligned is None
    assert reason == "fewer than 2 SfM depths"


def test_align_reports_non_positive_scale():
    prediction = np.zeros((6, 6), dtype=np.float32)
    prediction[3, 3] = 4.0
    prediction[3, 4] = 2.0
    observations = np.array([
        [0.0, 3.0, 3.0],
        [1.0, 3.5, 3.0],
    ])
    points = np.array([
        [0.0, 0.0, 2.0],
        [0.2, 0.0, 4.0],
    ])

    aligned, reason = _align(prediction, observations, points, _pose(6, 6), min_points=2)

    assert aligned is None
    assert reason.startswith("non-positive depth scale")


def test_fuse_tsdf_reconstructs_a_plane_at_one_metre(tmp_path: Path):
    height, width = 48, 64
    image_path = tmp_path / "frame_000000.jpg"
    cv2.imwrite(str(image_path), np.full((height, width, 3), 255, dtype=np.uint8))
    name = "v1/frame_000000.jpg"
    pose = _pose(width, height)
    frame = Frame("v1", image_path, 0, width, height)
    recon = SceneRecon(
        poses={name: pose},
        points_xyz=np.zeros((0, 3)),
        point_image_ids=np.zeros(0, dtype=int),
        observations={name: np.zeros((0, 3))},
    )
    depths = {"v1": {name: np.full((height, width), 1.0, dtype=np.float32)}}

    clouds = fuse_tsdf(
        {"v1": [frame]},
        recon,
        depths,
        FusionConfig(voxel_size=0.015, sdf_trunc=0.04),
    )

    points = np.asarray(clouds["v1"].points)
    assert len(points) > 0
    np.testing.assert_allclose(np.median(points[:, 2]), 1.0, atol=0.03)