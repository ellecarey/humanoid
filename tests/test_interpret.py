import numpy as np

from humanoid.change_detection import build_changes
from humanoid.config import VlmConfig
from humanoid.semantics.vlm import EXTENT_TOLERANCE_M, accept_pair, interpret_regions
from humanoid.types import ChangeRegion


def _region(
    region_id: int, change_type: str, centroid: list[float], extent: list[float]
) -> ChangeRegion:
    point = np.asarray(centroid, dtype=float)
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id="v1" if change_type == "removed" else "v2",
        points=point.reshape(1, 3),
        centroid=point,
        extent=np.asarray(extent, dtype=float),
        obb_rotation=np.eye(3),
    )


def _cfg() -> VlmConfig:
    return VlmConfig(provider="gemini", model="gemini-2.5-flash", cache=False)


def test_unset_key_pairs_permuted_extents(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    regions = [
        _region(1, "removed", [0.0, 0.0, 0.0], [0.4, 0.3, 0.5]),
        _region(2, "added", [1.0, 0.0, 0.0], [0.5, 0.4, 0.3]),
    ]

    changes = interpret_regions(regions, {}, _cfg())

    assert len(changes) == 1
    assert changes[0].id == 1
    assert changes[0].change_type == "moved"
    assert changes[0].label == "object"
    assert changes[0].region_ids == (1, 2)
    assert changes[0].displacement_m == 1.0
    assert changes[0].description == "The object was moved 1.00 m, mainly +X."


def test_extents_outside_tolerance_stay_removed_and_added(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    regions = [
        _region(1, "removed", [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
        _region(2, "added", [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    ]

    changes = interpret_regions(regions, {}, _cfg())

    assert [change.change_type for change in changes] == ["removed", "added"]
    assert [change.id for change in changes] == [1, 2]
    assert changes[0].displacement_m is None


def test_accept_pair_drops_a_proposed_pair_outside_tolerance():
    changes = build_changes(
        [
            _region(1, "removed", [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            _region(2, "added", [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ]
    )

    assert accept_pair(changes, 0, 1, EXTENT_TOLERANCE_M) is False
