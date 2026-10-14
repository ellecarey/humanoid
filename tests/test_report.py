import json

import numpy as np

from humanoid.types import DetectedChange
from humanoid.visualisation.report import write_report


def test_report_writes_html_and_json(tmp_path):
    removed = DetectedChange(
        id=1,
        change_type="removed",
        label="gold elephant",
        attributes={},
        confidence=1.0,
        region_ids=(1,),
        from_centroid=np.array([1.0, 0.0, 0.0]),
        to_centroid=None,
        extent=np.ones(3),
        displacement_m=None,
        description="The gold elephant was removed.",
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
        description="An orange was added near the centre of the room.",
    )
    points = {
        "static": np.array([[0.0, 0.0, 0.0]]),
        "removed": np.array([[1.0, 0.0, 0.0]]),
    }

    write_report(tmp_path, [removed, added], points)

    html = (tmp_path / "report.html").read_text()
    assert "The gold elephant was removed." in html
    parsed = json.loads((tmp_path / "changes.json").read_text())
    assert parsed[0]["description"] == "The gold elephant was removed."
    assert parsed[1]["type"] == "added"
