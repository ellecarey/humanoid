import numpy as np

from humanoid.utils.geometry import backproject, project


def test_project_roundtrip_and_rejects_points_behind_camera():
    rotation = np.eye(3)
    translation = np.zeros(3)
    intrinsics = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ])

    world = np.array([[0.2, -0.1, 2.0]])
    pixels, depth = project(world, rotation, translation, intrinsics)

    np.testing.assert_allclose(pixels, [[370.0, 215.0]])
    np.testing.assert_allclose(depth, [2.0])

    recovered = backproject(pixels[:, 0], pixels[:, 1], depth, intrinsics, rotation, translation)
    np.testing.assert_allclose(recovered, world, atol=1e-6)

    behind = np.array([[0.0, 0.0, -1.0]])
    pixels_behind, depth_behind = project(behind, rotation, translation, intrinsics)
    assert len(pixels_behind) == 0
    assert len(depth_behind) == 0