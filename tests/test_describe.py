import numpy as np

from humanoid.semantics import describe_changes
from humanoid.types import DetectedChange


def _change(change_type: str, label: str, heights: dict, **extra) -> DetectedChange:
    fields = {
        "from_centroid": np.zeros(3) if change_type != "added" else None,
        "to_centroid": np.ones(3) if change_type != "removed" else None,
        "displacement_m": None,
    }
    fields.update(extra)
    return DetectedChange(
        id=1,
        change_type=change_type,
        label=label,
        attributes={},
        confidence=0.82,
        region_ids=(1,),
        extent=np.array([0.09, 0.12, 0.09]),
        description="",
        heights=heights,
        **fields,
    )


def test_template_describes_a_move_across_the_floor():
    change = _change(
        "moved",
        "desk chair",
        {"from": (0.02, 0.5), "to": (0.03, 0.52)},
        displacement_m=0.96,
    )

    report = describe_changes([change])

    entry = report["changes"][0]
    assert entry["type"] == "moved"
    assert entry["displacement_m"] == 0.96
    assert entry["from"]["height_above_floor_m"] == 0.5
    assert entry["description"] == "The desk chair was moved 0.96 m across the floor."
    assert report["markdown"].startswith("# Scene changes\n")


def test_template_reports_a_change_of_level():
    change = _change(
        "moved", "mug", {"from": (0.7, 0.75), "to": (0.0, 0.05)}, displacement_m=1.2
    )
    assert describe_changes([change])["changes"][0]["description"] == (
        "The mug was moved 1.20 m and now sits 0.7 m lower."
    )


def test_template_places_removed_and_added_objects():
    removed = _change("removed", "flag pillow", {"from": (0.45, 0.62)})
    hung = _change("added", "umbrella", {"to": (0.3, 0.9)})
    on_floor = _change("added", "box", {"to": (0.02, 0.2)})
    unplaced = _change("added", "orange", {})

    report = describe_changes([removed, hung, on_floor, unplaced])

    assert [entry["description"] for entry in report["changes"]] == [
        "The flag pillow was removed from about 0.6 m above the floor.",
        "An umbrella was added about 0.9 m above the floor.",
        "A box was added on the floor.",
        "An orange was added.",
    ]
