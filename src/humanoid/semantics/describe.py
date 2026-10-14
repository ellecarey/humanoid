from dataclasses import replace

import numpy as np

from humanoid.types import DetectedChange

NEAR_CENTRE_M = 0.3


def describe_changes(changes: list[DetectedChange], clouds: dict) -> dict:
    centroid = _scene_centroid(clouds)
    described = [
        replace(change, description=_template(change, centroid)) for change in changes
    ]
    return {
        "changes": [_record(change) for change in described],
        "markdown": _markdown(described),
    }


def _template(change: DetectedChange, centroid: np.ndarray | None) -> str:
    if change.change_type == "removed":
        return (
            f"The {change.label} was removed{_place(change.from_centroid, centroid)}."
        )
    if change.change_type == "added":
        return f"{_article(change.label)} {change.label} was added{_place(change.to_centroid, centroid)}."
    return (
        f"The {change.label} was moved {change.displacement_m:.2f} m"
        f"{_axis_phrase(change.from_centroid, change.to_centroid)}."
    )


def _article(label: str) -> str:
    return "An" if label[:1].lower() in "aeiou" else "A"


def _place(point, centroid) -> str:
    if point is None or centroid is None:
        return ""
    delta = np.asarray(point, dtype=float) - np.asarray(centroid, dtype=float)
    if float(np.linalg.norm(delta)) < NEAR_CENTRE_M:
        return " near the centre of the room"
    return f" on the {_side(delta)} side of the room"


def _axis_phrase(origin, target) -> str:
    if origin is None or target is None:
        return ""
    delta = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
    return f", mainly {_side(delta)}"


def _side(delta: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(delta)))
    positive = float(delta[axis]) >= 0
    names = (("+X", "-X"), ("+Y", "-Y"), ("+Z", "-Z"))
    return names[axis][0 if positive else 1]


def _scene_centroid(clouds: dict) -> np.ndarray | None:
    chunks = []
    for cloud in clouds.values():
        points = np.asarray(cloud.points, dtype=float)
        if points.size:
            chunks.append(points.reshape(-1, 3))
    if not chunks:
        return None
    return np.mean(np.vstack(chunks), axis=0)


def _record(change: DetectedChange) -> dict:
    record = {
        "id": change.id,
        "type": change.change_type,
        "label": change.label,
        "confidence": round(float(change.confidence), 2),
        "description": change.description,
    }
    if change.from_centroid is not None:
        record["from"] = {
            "centroid": _xyz(change.from_centroid),
            "bbox_extent": _xyz(change.extent),
        }
    if change.to_centroid is not None:
        destination = {"centroid": _xyz(change.to_centroid)}
        if change.from_centroid is None:
            destination["bbox_extent"] = _xyz(change.extent)
        record["to"] = destination
    if change.displacement_m is not None:
        record["displacement_m"] = round(float(change.displacement_m), 2)
    return record


def _xyz(values) -> list[float]:
    return [round(float(value), 3) for value in values]


def _markdown(changes: list[DetectedChange]) -> str:
    lines = ["# Scene changes", ""]
    if not changes:
        lines.append("No changes detected.")
        return "\n".join(lines) + "\n"
    lines.extend(f"- {change.description}" for change in changes)
    lines.extend(
        ["", "## Summary", "", " ".join(change.description for change in changes)]
    )
    return "\n".join(lines) + "\n"
