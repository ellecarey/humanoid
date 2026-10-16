import numpy as np

from humanoid.change_detection import build_changes
from humanoid.config import VlmConfig
from humanoid.semantics import vlm
from humanoid.semantics.vlm import (
    EXTENT_ABS_TOL_M,
    EXTENT_REL_TOL,
    MovedPair,
    RegionVerdict,
    Verdict,
    accept_pair,
    interpret_regions,
)
from humanoid.types import ChangeRegion


def _region(
    region_id: int,
    change_type: str,
    centroid: list[float],
    extent: list[float],
    cue: str = "geometry",
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
        cue=cue,
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
    assert changes[0].description == "The object was moved 1.00 m."


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
    assert changes[0].description == "The object was removed."


def test_accept_pair_scales_the_tolerance_with_object_size():
    small = build_changes(
        [
            _region(1, "removed", [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]),
            _region(2, "added", [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ]
    )
    chair = build_changes(
        [
            _region(1, "removed", [0.0, 0.0, 0.0], [0.5, 0.6, 1.0]),
            _region(2, "added", [1.0, 0.0, 0.0], [0.6, 0.7, 1.25]),
        ]
    )

    assert accept_pair(small, 0, 1, EXTENT_ABS_TOL_M, EXTENT_REL_TOL) is False
    assert accept_pair(chair, 0, 1, EXTENT_ABS_TOL_M, EXTENT_REL_TOL) is True


def test_gemini_verdict_rejects_flips_and_pairs(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    regions = [
        _region(0, "removed", [0.0, 0.0, 0.0], [0.5, 0.6, 1.0]),
        _region(1, "removed", [3.0, 0.0, 0.0], [0.45, 0.45, 0.15]),
        _region(2, "removed", [2.0, 1.0, 0.0], [0.08, 0.1, 0.8], cue="appearance"),
        _region(3, "added", [1.0, 0.0, 0.0], [0.6, 0.6, 1.1]),
        _region(4, "added", [5.0, 0.0, 0.0], [0.3, 0.3, 0.3]),
    ]
    verdict = Verdict(
        regions=[
            RegionVerdict(
                index=0, real_change=True, change_type="removed", label="green chair"
            ),
            RegionVerdict(
                index=1,
                real_change=True,
                change_type="removed",
                label="flag cushion",
                description="A cushion was on the sofa.",
            ),
            RegionVerdict(
                index=2, real_change=True, change_type="added", label="umbrella"
            ),
            RegionVerdict(
                index=3, real_change=True, change_type="added", label="office chair"
            ),
            RegionVerdict(
                index=4, real_change=False, change_type="none", label="white wall"
            ),
        ],
        moved_pairs=[MovedPair(removed_index=1, added_index=3)],
    )
    monkeypatch.setattr(vlm, "_ask_gemini", lambda *args: verdict)

    changes = interpret_regions(regions, {}, _cfg())

    summary = [(change.change_type, change.label) for change in changes]
    assert summary == [
        ("moved", "green chair"),
        ("removed", "flag cushion"),
        ("added", "umbrella"),
    ]
    assert changes[0].displacement_m == 1.0
    assert changes[1].description == "A cushion was on the sofa."
    assert changes[2].to_centroid.tolist() == [2.0, 1.0, 0.0]
    assert changes[2].from_centroid is None
