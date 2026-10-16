import numpy as np

from humanoid.change_detection.clustering import _floor_plane, cluster_changes
from humanoid.config import ClusterConfig
from humanoid.types import LabeledDiff


def _cfg() -> ClusterConfig:
    return ClusterConfig(
        dbscan_eps_m=0.05,
        dbscan_min_samples=5,
        min_points=20,
        min_extent_m=0.05,
        max_extent_m=1.5,
        min_height_above_floor_m=0.05,
    )


def _grid(x, y, z) -> np.ndarray:
    xs, ys, zs = np.meshgrid(x, y, z, indexing="ij")
    return np.c_[xs.ravel(), ys.ravel(), zs.ravel()]


def test_floor_is_the_lowest_horizontal_plane_not_the_biggest_wall():
    floor = _grid(np.linspace(0, 2, 30), np.linspace(0, 2, 30), [0.0])
    wall = _grid([0.0], np.linspace(0, 4, 60), np.linspace(0, 3, 60))

    plane = _floor_plane(np.vstack([wall, floor]), np.array([0.0, 0.0, 1.0]), _cfg())

    normal = plane[:3] / np.linalg.norm(plane[:3])
    assert abs(normal[2]) > 0.99
    assert abs(plane[3] / np.linalg.norm(plane[:3])) < 0.02


def test_regions_carry_height_above_floor_and_vote_score():
    floor = _grid(np.linspace(-1, 2, 40), np.linspace(-1, 2, 40), [0.0])
    box = _grid(
        np.linspace(0.4, 0.6, 6), np.linspace(0.4, 0.6, 6), np.linspace(0.0, 0.5, 13)
    )
    points = np.vstack([floor, box])
    count = len(points)
    diff = LabeledDiff(
        points=points,
        source_scene=np.array(["v1"] * count),
        nn_distance=np.zeros(count),
        label=np.array(
            ["unchanged"] * len(floor) + ["changed"] * len(box), dtype=object
        ),
        cue=np.array([""] * len(floor) + ["geometry"] * len(box), dtype=object),
        free_views=np.r_[np.zeros(len(floor)), np.full(len(box), 3)].astype(int),
        agree_views=np.r_[np.zeros(len(floor)), np.full(len(box), 1)].astype(int),
    )

    regions = cluster_changes(diff, _cfg(), up=np.array([0.0, 0.0, 1.0]))

    assert len(regions) == 1
    assert regions[0].base_m < 0.05
    assert abs(regions[0].height_m - 0.25) < 0.05
    assert abs(regions[0].score - 0.75) < 1e-6


def test_flat_appearance_change_keeps_only_the_protruding_side():
    floor = _grid(np.linspace(-1, 2, 40), np.linspace(-1, 2, 40), [0.0])
    door = _grid([1.0], np.linspace(0.2, 0.3, 6), np.linspace(0.5, 1.2, 30))
    umbrella = door - [0.04, 0.0, 0.0]
    points = np.vstack([floor, door, umbrella])
    source = np.array(["v1"] * (len(floor) + len(door)) + ["v2"] * len(umbrella))
    label = np.array(["unchanged"] * len(floor) + ["changed"] * (len(door) * 2))
    cue = np.array([""] * len(floor) + ["appearance"] * (len(door) * 2))
    residual = np.r_[
        np.zeros(len(floor)), np.full(len(door), -0.04), np.full(len(door), 0.04)
    ]
    diff = LabeledDiff(
        points=points,
        source_scene=source,
        nn_distance=np.zeros(len(points)),
        label=label.astype(object),
        cue=cue.astype(object),
        residual=residual,
    )

    regions = cluster_changes(diff, _cfg(), up=np.array([0.0, 0.0, 1.0]))

    assert [(region.change_type, region.cue) for region in regions] == [
        ("added", "appearance")
    ]
    assert regions[0].centroid[0] < 1.0
