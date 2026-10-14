import numpy as np
import open3d as o3d

from humanoid.config import DescribeConfig
from humanoid.semantics import describe_changes
from humanoid.types import DetectedChange


def _cfg() -> DescribeConfig:
    return DescribeConfig(use_llm=False, provider="openai", model="gpt-4o")


def _cloud(*points: list[float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    return cloud


def test_template_describes_a_move():
    change = DetectedChange(
        id=3,
        change_type="moved",
        label="red mug",
        attributes={"color": "red"},
        confidence=0.82,
        region_ids=(1, 2),
        from_centroid=np.array([0.0, 0.8, 0.0]),
        to_centroid=np.array([0.96, 0.8, 0.0]),
        extent=np.array([0.09, 0.12, 0.09]),
        displacement_m=0.96,
        description="",
    )

    report = describe_changes([change], {"v1": _cloud([0.0, 0.0, 0.0])}, _cfg())

    entry = report["changes"][0]
    assert entry["type"] == "moved"
    assert entry["displacement_m"] == 0.96
    assert entry["from"]["centroid"] == [0.0, 0.8, 0.0]
    assert entry["from"]["bbox_extent"] == [0.09, 0.12, 0.09]
    assert entry["to"]["centroid"] == [0.96, 0.8, 0.0]
    assert entry["description"] == "The red mug was moved 0.96 m, mainly +X."
    assert report["markdown"].startswith("# Scene changes\n")
    assert "The red mug was moved 0.96 m, mainly +X." in report["markdown"]


def test_template_places_removed_and_added_objects():
    removed = DetectedChange(
        id=1,
        change_type="removed",
        label="gold elephant",
        attributes={},
        confidence=1.0,
        region_ids=(1,),
        from_centroid=np.array([2.0, 0.0, 0.0]),
        to_centroid=None,
        extent=np.ones(3),
        displacement_m=None,
        description="",
    )
    added = DetectedChange(
        id=2,
        change_type="added",
        label="orange",
        attributes={},
        confidence=1.0,
        region_ids=(2,),
        from_centroid=None,
        to_centroid=np.array([0.1, 0.0, 0.0]),
        extent=np.ones(3),
        displacement_m=None,
        description="",
    )
    clouds = {"v1": _cloud([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])}

    report = describe_changes([removed, added], clouds, _cfg())

    assert report["changes"][0]["description"] == (
        "The gold elephant was removed on the +X side of the room."
    )
    assert report["changes"][1]["description"] == (
        "An orange was added near the centre of the room."
    )