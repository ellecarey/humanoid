import base64
import json
from html import escape
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from humanoid.semantics.describe import _record
from humanoid.semantics.views import best_view
from humanoid.types import DetectedChange, RegionView

_COLOURS = {
    "static": ("#9e9e9e", "static"),
    "removed": ("#e53935", "removed"),
    "added": ("#43a047", "added"),
    "unknown": ("rgba(255, 235, 59, 0.45)", "unobserved"),
}
_BOX = {"removed": "#e53935", "added": "#43a047", "moved": "#1565c0"}
MAX_POINTS = {"static": 150_000, "unknown": 40_000, "removed": 40_000, "added": 40_000}


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
    if not cards:
        cards = '<p class="empty">No changes detected.</p>'
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Scene changes</title>
<style>
  body {{ font-family: sans-serif; margin: 1.5rem; color: #222; }}
  .cards {{ display: grid; gap: 1rem; margin-top: 1.5rem; }}
  .card {{ border: 1px solid #ddd; padding: 0.75rem 1rem; }}
  .card h3 {{ margin: 0 0 0.5rem; text-transform: capitalize; }}
  .crops {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }}
  figure {{ margin: 0; }}
  img {{ max-width: 100%; height: auto; }}
  .empty {{ color: #888; }}
  .meta {{ color: #666; font-size: 0.85rem; margin: 0; }}
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
        colours = points.get(f"{key}_colors")
        cloud, colours = _thin(cloud, colours, MAX_POINTS[key])
        marker = {"size": 2, "color": colour}
        if colours is not None:
            marker["color"] = [
                f"rgb({r},{g},{b})" for r, g, b in (colours * 255).astype(int)
            ]
        fig.add_trace(
            go.Scatter3d(
                x=cloud[:, 0],
                y=cloud[:, 1],
                z=cloud[:, 2],
                mode="markers",
                name=name,
                marker=marker,
            )
        )
    for change_type, colour in _BOX.items():
        segments = []
        for change in changes:
            if change.change_type != change_type:
                continue
            for centre in (change.from_centroid, change.to_centroid):
                if centre is not None:
                    segments.extend(_box_edges(centre, change.extent))
            if change.from_centroid is not None and change.to_centroid is not None:
                segments.append(
                    (
                        np.asarray(change.from_centroid, dtype=float),
                        np.asarray(change.to_centroid, dtype=float),
                    )
                )
        if not segments:
            continue
        xs, ys, zs = _broken_line(segments)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                name=f"{change_type} box",
                line={"color": colour, "width": 5},
            )
        )
    fig.update_layout(
        title="Scene changes",
        height=720,
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


def _thin(cloud: np.ndarray, colours, limit: int):
    if colours is not None:
        colours = np.asarray(colours, dtype=float).reshape(-1, 3)
        if len(colours) != len(cloud):
            colours = None
    if len(cloud) <= limit:
        return cloud, colours
    pick = np.random.default_rng(0).choice(len(cloud), limit, replace=False)
    return cloud[pick], None if colours is None else colours[pick]


def _card(change: DetectedChange, views) -> str:
    before, after = _before_after(change, views)
    title = f"{change.id}. {change.change_type}: {change.label}"
    return f"""<article class="card">
  <h3>{escape(title)}</h3>
  <div class="crops">
    <figure><figcaption>Before (video 1)</figcaption>{_image(before)}</figure>
    <figure><figcaption>After (video 2)</figcaption>{_image(after)}</figure>
  </div>
  <p>{escape(change.description)}</p>
  <p class="meta">Confidence {change.confidence:.2f} · cue: {escape(change.attributes.get("cue", "geometry"))}</p>
</article>"""


def _before_after(change: DetectedChange, views):
    ids = change.region_ids
    if not ids:
        return None, None
    before = best_view(views, ids[0], "v1")
    after = best_view(views, ids[-1], "v2")
    if before is not None and before.scene_id == "" and change.change_type == "added":
        before = None
    if after is not None and after.scene_id == "" and change.change_type == "removed":
        after = None
    return (
        before.crop if before is not None else None,
        after.crop if after is not None else None,
    )


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
        real = points.get(f"{key}_colors")
        if key == "static" and real is not None and len(real) == len(cloud):
            colours.append(np.asarray(real, dtype=float))
        else:
            colours.append(np.tile(colour, (len(cloud), 1)))
    if not chunks:
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(chunks))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack(colours))
    o3d.io.write_point_cloud(str(path), cloud)
