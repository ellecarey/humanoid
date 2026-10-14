import cv2
import numpy as np

from humanoid.semantics import select_views
from humanoid.types import CameraPose, ChangeRegion, Frame, SceneRecon


def test_select_views_keeps_the_unoccluded_frame(tmp_path):
    image = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    clear = tmp_path / "clear.jpg"
    blocked = tmp_path / "blocked.jpg"
    cv2.imwrite(str(clear), image)
    cv2.imwrite(str(blocked), image)

    intrinsics = np.array([
        [4.0, 0.0, 4.0],
        [0.0, 4.0, 4.0],
        [0.0, 0.0, 1.0],
    ])

    def pose(name: str) -> CameraPose:
        return CameraPose("v1", name, np.eye(3), np.zeros(3), intrinsics)

    names = ("v1/clear.jpg", "v1/blocked.jpg")
    recon = SceneRecon(
        poses={name: pose(name) for name in names},
        points_xyz=np.zeros((0, 3)),
        point_image_ids=np.zeros(0, dtype=int),
        observations={},
    )
    depths = {
        "v1": {
            "v1/clear.jpg": np.full((8, 8), 2.0, np.float32),
            "v1/blocked.jpg": np.full((8, 8), 0.5, np.float32),
        }
    }
    points = np.array([
        [0.0, 0.0, 2.0],
        [0.25, 0.0, 2.0],
        [-0.25, 0.0, 2.0],
        [0.0, 0.25, 2.0],
        [0.0, -0.25, 2.0],
    ])
    region = ChangeRegion(
        id=1,
        change_type="removed",
        scene_id="v1",
        points=points,
        centroid=points.mean(axis=0),
        extent=np.array([0.5, 0.5, 0.1]),
        obb_rotation=np.eye(3),
    )
    frames = {
        "v1": [
            Frame("v1", clear, 0, 8, 8),
            Frame("v1", blocked, 1, 8, 8),
        ]
    }

    views = select_views([region], frames, recon, depths)

    assert [view.image_name for view in views[1]] == ["v1/clear.jpg"]