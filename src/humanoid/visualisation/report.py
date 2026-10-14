import base64
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from humanoid.semantics.describe import _record
from humanoid.types import DetectedChange, RegionView

_COLOURS = {
    "static": ("#9e9e9e", "static"),
    "removed": ("#e53935", "removed"),
    "added": ("#43a047", "added"),
    "unknown": ("rgba(255, 235, 59, 0.45)", "unobserved"),
}
_MOVED = "#1565c0"


def write_report(
    directory: Path,
    changes: list[DetectedChange],
    points: dict[str, np.ndarray],
    views: dict[int, list[RegionView]] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    records = [_record(change) for change in changes]
    (directory / "changes.json").write_text(json.dumps(records, indent=2) + "\n")
    (directory / "report.html").write_text(_page(changes, points, views))
    _write_ply(directory / "scene.ply", points)


def _page(changes, points, views) -> str:
    figure = _figure(changes, points).to_html(include_plotlyjs=True, full_html=False)
    cards = "\n".join(_card(change, views) for change in changes)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Scene changes</title>
<style>
  body {{ font-family: sans-serif; margin: 1.5rem; color: #222; }}
  .cards {{ display: grid; gap: 1rem; margin-top: 1.5rem; }}
  .card {{ border: 1px solid #ddd; padding: 0.75rem 1rem; }}
  .crops {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
  figure {{ margin: 0; }}
  img {{ max-width: 100%; height: auto; }}
  .empty {{ color: #888; }}
</style>
</head>
<body>
{figure}
<section class="cards">
{cards}
</section>
</body>
</html>
"""


def _figure(changes: list[DetectedChange], points: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for key, (colour, name) in _COLOURS.items():
        cloud = _xyz(points.get(key))
        if len(cloud) == 0:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=cloud[:, 0],
                y=cloud[:, 1],
                z=cloud[:, 2],
                mode="markers",
                name=name,
                marker={"size": 2, "color": colour},
            )
        )
    segments = []
    for change in changes:
        if change.change_type != "moved":
            continue
        if change.from_centroid is None or change.to_centroid is None:
            continue
        segments.extend(_box_edges(change.from_centroid, change.extent))
        segments.extend(_box_edges(change.to_centroid, change.extent))
        segments.append(
            (
                np.asarray(change.from_centroid, dtype=float),
                np.asarray(change.to_centroid, dtype=float),
            )
        )
    if segments:
        xs, ys, zs = _broken_line(segments)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name="moved",
                line={"color": _MOVED, "width": 4},
            )
        )
    fig.update_layout(
        title="Scene changes",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"orientation": "h"},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
        },
    )
    return fig


def _card(change: DetectedChange, views) -> str:
    before, after = _before_after(change, views)
    return f"""<article class="card">
  <div class="crops">
    <figure><figcaption>Before</figcaption>{_image(before)}</figure>
    <figure><figcaption>After</figcaption>{_image(after)}</figure>
  </div>
  <p>{escape(change.description)}</p>
</article>"""


def _before_after(change: DetectedChange, views):
    ids = change.region_ids
    if change.change_type == "removed":
        return _crop(views, ids[0] if ids else None), None
    if change.change_type == "added":
        return None, _crop(views, ids[0] if ids else None)
    before = _crop(views, ids[0]) if len(ids) > 0 else None
    after = _crop(views, ids[1]) if len(ids) > 1 else None
    return before, after


def _crop(views, region_id: int | None) -> np.ndarray | None:
    if views is None or region_id is None:
        return None
    crops = views.get(region_id) or []
    return crops[0].crop if crops else None


def _image(crop: np.ndarray | None) -> str:
    if crop is None:
        return '<p class="empty">No crop</p>'
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        return '<p class="empty">No crop</p>'
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f'<img alt="" src="data:image/jpeg;base64,{payload}">'


def _xyz(values) -> np.ndarray:
    if values is None:
        return np.zeros((0, 3))
    points = np.asarray(values, dtype=float)
    if points.size == 0:
        return np.zeros((0, 3))
    return points.reshape(-1, 3)


def _box_edges(center, extent) -> list[tuple[np.ndarray, np.ndarray]]:
    center = np.asarray(center, dtype=float)
    half = np.asarray(extent, dtype=float) / 2.0
    corners = [
        center + np.array([x, y, z])
        for x in (-half[0], half[0])
        for y in (-half[1], half[1])
        for z in (-half[2], half[2])
    ]
    pairs = (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    return [(corners[a], corners[b]) for a, b in pairs]


def _broken_line(segments) -> tuple[list, list, list]:
    xs, ys, zs = [], [], []
    for start, end in segments:
        xs.extend([float(start[0]), float(end[0]), None])
        ys.extend([float(start[1]), float(end[1]), None])
        zs.extend([float(start[2]), float(end[2]), None])
    return xs, ys, zs


def _write_ply(path: Path, points: dict[str, np.ndarray]) -> None:
    chunks = []
    colours = []
    rgb = {
        "static": (0.62, 0.62, 0.62),
        "removed": (0.90, 0.22, 0.21),
        "added": (0.26, 0.63, 0.28),
        "unknown": (1.0, 0.92, 0.23),
    }
    for key, colour in rgb.items():
        cloud = _xyz(points.get(key))
        if len(cloud) == 0:
            continue
        chunks.append(cloud)
        colours.append(np.tile(colour, (len(cloud), 1)))
    if not chunks:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(chunks))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack(colours))
    o3d.io.write_point_cloud(str(path), cloud)
