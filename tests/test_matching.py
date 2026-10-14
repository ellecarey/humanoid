import numpy as np

from humanoid.change_detection import build_changes
from humanoid.types import ChangeRegion


def _region(region_id: int, change_type: str, centroid: list[float]) -> ChangeRegion:
    point = np.asarray(centroid, dtype=float)
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id="v1" if change_type == "removed" else "v2",
        points=point.reshape(1, 3),
        centroid=point,
        extent=np.ones(3),
        obb_rotation=np.eye(3),
    )


def test_build_changes_keeps_removed_and_added_unpaired():
    regions = [
        _region(2, "added", [1.0, 0.8, 0.0]),
        _region(1, "removed", [0.0, 0.8, 0.0]),
    ]
    changes = build_changes(regions)
    assert [change.id for change in changes] == [1, 2]
    assert [change.change_type for change in changes] == ["removed", "added"]
    assert changes[0].region_ids == (1,)
    assert changes[1].label == "object"
    assert changes[0].displacement_m is None
