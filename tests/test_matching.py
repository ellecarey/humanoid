import numpy as np
import pytest

from humanoid.change_detection import match_moved
from humanoid.config import MatchConfig
from humanoid.types import ChangeRegion, RegionLabel


def _cfg() -> MatchConfig:
    return MatchConfig(
        max_cost=0.45,
        weight_label=0.4,
        weight_embed=0.4,
        weight_size=0.2,
    )


def _region(region_id: int, change_type: str, centroid: list[float], extent: list[float] | None = None):
    point = np.asarray(centroid, dtype=float)
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id="v1" if change_type == "removed" else "v2",
        points=point.reshape(1, 3),
        centroid=point,
        extent=np.ones(3) if extent is None else np.asarray(extent, dtype=float),
        obb_rotation=np.eye(3),
    )


def test_matching_pairs_similar_removed_and_added():
    removed = _region(1, "removed", [0.0, 0.8, 0.0])
    added = _region(2, "added", [1.0, 0.8, 0.0])
    labels = {
        1: RegionLabel(1, "red mug", {"color": "red"}),
        2: RegionLabel(2, "red mug", {"color": "red"}),
    }
    embeds = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0])}

    changes = match_moved([removed, added], labels, embeds, _cfg())

    assert len(changes) == 1
    change = changes[0]
    assert change.id == 1
    assert change.change_type == "moved"
    assert change.label == "red mug"
    assert change.region_ids == (1, 2)
    assert change.displacement_m == pytest.approx(1.0)
    assert change.confidence == pytest.approx(1.0)
    np.testing.assert_allclose(change.from_centroid, [0.0, 0.8, 0.0])
    np.testing.assert_allclose(change.to_centroid, [1.0, 0.8, 0.0])


def test_matching_leaves_dissimilar_regions_unpaired():
    removed = _region(1, "removed", [0.0, 0.8, 0.0], [0.1, 0.1, 0.1])
    added = _region(2, "added", [1.0, 0.8, 0.0], [1.0, 1.0, 1.0])
    labels = {
        1: RegionLabel(1, "red mug", {}),
        2: RegionLabel(2, "chair", {}),
    }
    embeds = {1: np.array([1.0, 0.0]), 2: np.array([-1.0, 0.0])}

    changes = match_moved([removed, added], labels, embeds, _cfg())

    assert [change.change_type for change in changes] == ["removed", "added"]
    assert [change.region_ids for change in changes] == [(1,), (2,)]


def test_matching_keeps_the_extra_added_region():
    removed = _region(1, "removed", [0.0, 0.0, 0.0])
    close = _region(2, "added", [0.2, 0.0, 0.0])
    other = _region(3, "added", [3.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    labels = {
        1: RegionLabel(1, "bin", {}),
        2: RegionLabel(2, "bin", {}),
        3: RegionLabel(3, "rat", {}),
    }
    embeds = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([-1.0, 0.0]),
    }

    changes = match_moved([removed, close, other], labels, embeds, _cfg())

    moved = next(change for change in changes if change.change_type == "moved")
    added = next(change for change in changes if change.change_type == "added")
    assert moved.region_ids == (1, 2)
    assert added.region_ids == (3,)