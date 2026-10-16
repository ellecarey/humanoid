import numpy as np
import open3d as o3d

from humanoid.change_detection import apply_visibility, compute_geometric_diff
from humanoid.config import DiffConfig, VisibilityConfig
from humanoid.types import CameraPose, GeometricDiff, SceneRecon

NAMES = ("v2/a.jpg", "v2/b.jpg", "v2/c.jpg", "v2/d.jpg")


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
        K=np.array(
            [
                [4.0, 0.0, 4.0],
                [0.0, 4.0, 4.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def _recon(*depth_m: float) -> tuple[SceneRecon, dict]:
    """One v2 camera per depth, all looking at the point (0, 0, 2) down +Z."""
    names = NAMES[: len(depth_m)]
    poses = {name: _pose("v2", name, [0.0, 0.0, 0.0]) for name in names}
    depths = {}
    for name, value in zip(names, depth_m):
        depth = np.zeros((8, 8), dtype=np.float32)
        depth[4, 4] = value
        depths[name] = depth
    recon = SceneRecon(poses=poses, depths={}, confidence={})
    return recon, {"v2": depths}


def _candidate(point: list[float], candidate: bool = True) -> GeometricDiff:
    return GeometricDiff(
        points=np.array([point], dtype=float),
        source_scene=np.array(["v1"]),
        nn_distance=np.array([0.1 if candidate else 0.0]),
        candidate=np.array([candidate]),
    )


def _cfg(**update) -> VisibilityConfig:
    cfg = VisibilityConfig(
        depth_margin_m=0.05,
        min_free_views=2,
        free_ratio=0.6,
        min_agree_views=2,
        appearance_cos=0.5,
    )
    return cfg.model_copy(update=update)


def test_distance_above_threshold_is_a_candidate():
    shared = np.array([[0.0, 0.0, 0.0]])
    shifted = np.array([[1.0, 0.0, 0.1]])
    diff = compute_geometric_diff(
        _cloud(np.vstack([shared, [[1.0, 0.0, 0.0]]])),
        _cloud(np.vstack([shared, shifted])),
        DiffConfig(change_distance_m=0.03),
    )

    by_source = {
        scene: diff.candidate[diff.source_scene == scene] for scene in ("v1", "v2")
    }
    assert by_source["v1"].tolist() == [False, True]
    assert by_source["v2"].tolist() == [False, True]


def test_free_space_in_two_views_is_changed():
    recon, depths = _recon(3.0, 3.0)
    labeled = apply_visibility(_candidate([0.0, 0.0, 2.0]), recon, depths, _cfg())
    assert labeled.label.tolist() == ["changed"]
    assert labeled.cue.tolist() == ["geometry"]
    assert labeled.free_views.tolist() == [2]


def test_split_vote_stays_unknown_and_agreement_keeps_the_point():
    recon, depths = _recon(3.0, 3.0, 2.0, 2.0)
    labeled = apply_visibility(_candidate([0.0, 0.0, 2.0]), recon, depths, _cfg())
    assert labeled.label.tolist() == ["unknown"]

    recon, depths = _recon(3.0, 2.0, 2.0)
    labeled = apply_visibility(_candidate([0.0, 0.0, 2.0]), recon, depths, _cfg())
    assert labeled.label.tolist() == ["unchanged"]


def test_relative_margin_absorbs_far_depth_noise():
    recon, depths = _recon(4.15, 4.15)
    point = _candidate([0.0, 0.0, 4.0])
    assert apply_visibility(point, recon, depths, _cfg()).label.tolist() == ["changed"]
    labeled = apply_visibility(point, recon, depths, _cfg(depth_margin_rel=0.05))
    assert labeled.label.tolist() == ["unchanged"]


def test_low_confidence_depth_stays_unknown():
    recon, depths = _recon(3.0, 3.0)
    confidence = np.zeros((8, 8), dtype=np.float32)
    recon.confidence = {"v2": {name: confidence.copy() for name in NAMES[:2]}}
    labeled = apply_visibility(
        _candidate([0.0, 0.0, 2.0]), recon, depths, _cfg(min_confidence=1.5)
    )
    assert labeled.label.tolist() == ["unknown"]


def test_occluder_in_front_stays_unknown():
    recon, depths = _recon(1.0, 1.0)
    labeled = apply_visibility(_candidate([0.0, 0.0, 2.0]), recon, depths, _cfg())
    assert labeled.label.tolist() == ["unknown"]


def test_point_behind_the_cameras_stays_unknown():
    recon, depths = _recon(3.0, 3.0)
    labeled = apply_visibility(_candidate([0.0, 0.0, -1.0]), recon, depths, _cfg())
    assert labeled.label.tolist() == ["unknown"]


def _with_v1_view(recon: SceneRecon, depths: dict) -> None:
    recon.poses["v1/a.jpg"] = _pose("v1", "v1/a.jpg", [0.0, 0.0, 0.0])
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[4, 4] = 2.0
    depths["v1"] = {"v1/a.jpg": depth}


def _features(own: list[float], other: list[float]) -> dict:
    def grid(vector):
        return np.tile(np.asarray(vector, dtype=np.float16), (8, 8, 1))

    return {
        "v1": {"v1/a.jpg": grid(own)},
        "v2": {name: grid(other) for name in NAMES[:2]},
    }


def test_same_depth_with_different_appearance_is_changed():
    recon, depths = _recon(2.0, 2.0)
    _with_v1_view(recon, depths)
    labeled = apply_visibility(
        _candidate([0.0, 0.0, 2.0], candidate=False),
        recon,
        depths,
        _cfg(),
        _features([1.0, 0.0], [0.0, 1.0]),
    )
    assert labeled.label.tolist() == ["changed"]
    assert labeled.cue.tolist() == ["appearance"]
    assert labeled.appearance[0] < 0.5


def test_same_depth_and_appearance_is_unchanged():
    recon, depths = _recon(2.0, 2.0)
    _with_v1_view(recon, depths)
    labeled = apply_visibility(
        _candidate([0.0, 0.0, 2.0], candidate=False),
        recon,
        depths,
        _cfg(),
        _features([1.0, 0.0], [1.0, 0.1]),
    )
    assert labeled.label.tolist() == ["unchanged"]
    assert labeled.appearance[0] > 0.9
