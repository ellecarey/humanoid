import json
import re
from pathlib import Path

import cv2
import numpy as np

from humanoid.types import ChangeRegion, DetectedChange, Frame, LabeledDiff

THUMB_W, THUMB_H = 240, 135
SHEET_COLUMNS = 10


def frame_sheets(frames: dict[str, list[Frame]], directory: Path) -> None:
    """One contact sheet per video so coverage of the room can be checked by eye."""
    directory.mkdir(parents=True, exist_ok=True)
    for scene_id, scene_frames in frames.items():
        thumbs = []
        for frame in scene_frames:
            image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            thumb = cv2.resize(image, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb, f"{frame.time_s:.1f}s", (6, 22), 0, 0.6, (0, 255, 255), 2
            )
            thumbs.append(thumb)
        if not thumbs:
            continue
        blank = np.zeros_like(thumbs[0])
        thumbs += [blank] * (-len(thumbs) % SHEET_COLUMNS)
        rows = [
            np.hstack(thumbs[start : start + SHEET_COLUMNS])
            for start in range(0, len(thumbs), SHEET_COLUMNS)
        ]
        cv2.imwrite(str(directory / f"frames_{scene_id}.jpg"), np.vstack(rows))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_plain) + "\n")


def detection_summary(
    labeled: LabeledDiff, regions: list[ChangeRegion], directory: Path
) -> dict:
    """Label counts, regions, and score histograms for threshold tuning."""
    directory.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    cue = labeled.cue if labeled.cue is not None else np.full(len(labeled.label), "")
    for scene_id in ("v1", "v2"):
        mask = labeled.source_scene == scene_id
        counts[scene_id] = {
            "points": int(mask.sum()),
            "unchanged": int((mask & (labeled.label == "unchanged")).sum()),
            "unknown": int((mask & (labeled.label == "unknown")).sum()),
            "changed_geometry": int((mask & (cue == "geometry")).sum()),
            "changed_appearance": int((mask & (cue == "appearance")).sum()),
        }
    summary = {
        "counts": counts,
        "regions": [
            {
                "id": region.id,
                "type": region.change_type,
                "cue": region.cue,
                "points": len(region.points),
                "centroid": np.round(region.centroid, 3).tolist(),
                "sorted_extent": np.round(np.sort(region.extent), 3).tolist(),
            }
            for region in regions
        ],
    }
    write_json(directory / "detect.json", summary)
    _score_histograms(labeled, directory / "scores.html")
    return summary


def _score_histograms(labeled: LabeledDiff, path: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if labeled.free_views is None or labeled.agree_views is None:
        return
    observed = labeled.free_views + labeled.agree_views
    ratio = labeled.free_views[observed > 0] / observed[observed > 0]
    cosine = labeled.appearance
    cosine = cosine[np.isfinite(cosine)] if cosine is not None else np.zeros(0)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "free / (free + agree) per point",
            "appearance cosine to the other video",
        ),
    )
    fig.add_trace(go.Histogram(x=_sample(ratio), nbinsx=50, name="free ratio"), 1, 1)
    fig.add_trace(go.Histogram(x=_sample(cosine), nbinsx=50, name="cosine"), 1, 2)
    fig.update_yaxes(type="log")
    fig.write_html(str(path), include_plotlyjs="cdn")


def check_ground_truth(changes: list[DetectedChange], path: Path | None) -> list[dict]:
    """Match each ground-truth change to a detected change of the same type and label."""
    if path is None or not Path(path).is_file():
        return []
    truth = json.loads(Path(path).read_text())
    used: set[int] = set()
    rows = []
    for item in truth:
        keywords = [word.lower() for word in item.get("keywords", [item["label"]])]
        match, type_only = None, None
        for change in changes:
            if change.id in used or change.change_type != item["type"]:
                continue
            words = set(re.findall(r"[a-z]+", change.label.lower()))
            if any(
                keyword in words or keyword in change.label.lower()
                for keyword in keywords
            ):
                match = change
                break
            type_only = type_only or change
        found = match or type_only
        if found is not None:
            used.add(found.id)
        rows.append(
            {
                "expected": f"{item['type']} {item['label']}",
                "detected": f"{found.change_type} {found.label}" if found else "",
                "result": "match" if match else ("type only" if found else "missed"),
            }
        )
    return rows


def _sample(values: np.ndarray, limit: int = 200_000) -> np.ndarray:
    if len(values) <= limit:
        return values
    return np.random.default_rng(0).choice(values, limit, replace=False)


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
