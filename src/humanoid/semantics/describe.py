from dataclasses import replace

from humanoid.types import DetectedChange

FLOOR_CONTACT_M = 0.1
LEVEL_CHANGE_M = 0.15


def describe_changes(changes: list[DetectedChange]) -> dict:
    described = [replace(change, description=_template(change)) for change in changes]
    return {
        "changes": [_record(change) for change in described],
        "markdown": _markdown(described),
    }


def _template(change: DetectedChange) -> str:
    label = change.label
    if change.change_type == "removed":
        return f"The {label} was removed{_where(change.heights.get('from'), True)}."
    if change.change_type == "added":
        where = _where(change.heights.get("to"), False)
        return f"{_article(label)} {label} was added{where}."
    return f"The {label} was moved {change.displacement_m:.2f} m{_level(change)}."


def _article(label: str) -> str:
    return "An" if label[:1].lower() in "aeiou" else "A"


def _where(heights: tuple[float, float] | None, removed: bool) -> str:
    if heights is None:
        return ""
    base, centre = heights
    if base < FLOOR_CONTACT_M:
        return " from the floor" if removed else " on the floor"
    prefix = " from about" if removed else " about"
    return f"{prefix} {centre:.1f} m above the floor"


def _level(change: DetectedChange) -> str:
    before, after = change.heights.get("from"), change.heights.get("to")
    if before is None or after is None:
        return ""
    rise = after[1] - before[1]
    if abs(rise) >= LEVEL_CHANGE_M:
        return f" and now sits {abs(rise):.1f} m {'higher' if rise > 0 else 'lower'}"
    if before[0] < FLOOR_CONTACT_M and after[0] < FLOOR_CONTACT_M:
        return " across the floor"
    return ""


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
    for key in ("from", "to"):
        if key in record and key in change.heights:
            record[key]["height_above_floor_m"] = round(
                float(change.heights[key][1]), 2
            )
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
