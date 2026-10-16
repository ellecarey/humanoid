import json
import os
import re
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel

from humanoid.change_detection.matching import build_changes
from humanoid.config import VlmConfig
from humanoid.semantics.describe import _template
from humanoid.semantics.views import best_view
from humanoid.types import ChangeRegion, DetectedChange, RegionView
from humanoid.utils.logging import get_logger

log = get_logger(__name__)

EXTENT_ABS_TOL_M = 0.15
EXTENT_REL_TOL = 0.3
MAX_REGIONS = 40
MAX_IMAGE_SIDE = 768
_STOPWORDS = {"a", "an", "the", "of", "with", "and", "on", "in", "small", "large"}

_PROMPT = """You are checking candidate changes between two visits to the same room.
Each numbered region has a BEFORE image (first visit) and an AFTER image (second visit).
The green box marks the same 3D location in both images; the camera position differs.
The geometry JSON gives each region its geometric guess (removed: something was there
before and is gone; added: something is there now that was not before), the cue that
found it (geometry: a surface appeared or disappeared; appearance: the surface looks
different at the same depth), and its centroid and sorted extent in metres.

For every region decide:
- real_change: true only if an object inside the box was really added, removed or moved.
  Use false for differences caused only by viewpoint, lighting, exposure, blur,
  reflections or reconstruction noise, or when the box holds only walls, floor or
  fixed furniture that look the same in both images.
- change_type: "removed" if the object is in BEFORE but not AFTER, "added" if it is in
  AFTER but not BEFORE, "none" when real_change is false.
- label: a short noun phrase for the object, for example "union jack cushion" or
  "green office chair". For a rejected region say what the box holds.
- description: one sentence on where the object was or is, relative to nearby
  furniture. Do not give distances.
moved_pairs: [removed_index, added_index] for regions that show the same object at two
different places. Only pair regions you marked real.
summary: one sentence covering every real change.
Reply with JSON only."""


class RegionVerdict(BaseModel):
    index: int
    real_change: bool
    change_type: Literal["added", "removed", "none"]
    label: str
    description: str = ""


class MovedPair(BaseModel):
    removed_index: int
    added_index: int
    description: str = ""


class Verdict(BaseModel):
    regions: list[RegionVerdict]
    moved_pairs: list[MovedPair] = []
    summary: str = ""


def interpret_regions(
    regions: list[ChangeRegion],
    views: dict[int, list[RegionView]],
    cfg: VlmConfig,
    debug_dir: Path | None = None,
) -> list[DetectedChange]:
    changes = build_changes(regions)
    if not os.environ.get("GEMINI_API_KEY"):
        return _collapse(changes, _offline_pairs(changes, cfg), {}, {}, set())

    sizes = {region.id: len(region.points) for region in regions}
    order = sorted(
        range(len(changes)),
        key=lambda index: sizes.get(changes[index].region_ids[0], 0),
        reverse=True,
    )
    asked = sorted(order[:MAX_REGIONS])
    verdict = _ask_gemini(changes, asked, views, cfg, debug_dir)
    rejected = set(order[MAX_REGIONS:])
    rejected |= _apply_verdict(changes, verdict)
    texts = {item.index: item.description for item in verdict.regions}

    pairs, pair_texts = [], {}
    for item in verdict.moved_pairs:
        i, j = item.removed_index, item.added_index
        if i in rejected or j in rejected:
            continue
        if accept_pair(changes, i, j, cfg.pair_abs_tol_m, cfg.pair_rel_tol):
            pairs.append((i, j))
            pair_texts[(i, j)] = item.description
    pairs += _offline_pairs(
        changes,
        cfg,
        exclude=rejected | {k for pair in pairs for k in pair},
        by_label=True,
    )
    return _collapse(changes, pairs, pair_texts, texts, rejected)


def accept_pair(
    changes: list[DetectedChange],
    i: int,
    j: int,
    abs_tol: float = EXTENT_ABS_TOL_M,
    rel_tol: float = EXTENT_REL_TOL,
) -> bool:
    if i == j or not (0 <= i < len(changes) and 0 <= j < len(changes)):
        return False
    left, right = changes[i], changes[j]
    if {left.change_type, right.change_type} != {"removed", "added"}:
        return False
    tolerance = max(
        abs_tol, rel_tol * float(max(np.max(left.extent), np.max(right.extent)))
    )
    return _extent_cost(left.extent, right.extent) <= tolerance


def _extent_cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.sort(a) - np.sort(b))))


def _offline_pairs(
    changes: list[DetectedChange],
    cfg: VlmConfig,
    exclude: set[int] | None = None,
    by_label: bool = False,
) -> list[tuple[int, int]]:
    used: set[int] = set(exclude or ())
    added = [
        j
        for j, change in enumerate(changes)
        if change.change_type == "added" and j not in used
    ]
    pairs: list[tuple[int, int]] = []
    for i, change in enumerate(changes):
        if change.change_type != "removed" or i in used:
            continue
        choices = []
        for j in added:
            if j in used or not accept_pair(
                changes, i, j, cfg.pair_abs_tol_m, cfg.pair_rel_tol
            ):
                continue
            if by_label and not _same_label(change.label, changes[j].label):
                continue
            distance = float(
                np.linalg.norm(changes[j].to_centroid - change.from_centroid)
            )
            choices.append(
                (distance, _extent_cost(change.extent, changes[j].extent), j)
            )
        if not choices:
            continue
        j = min(choices)[2]
        used.add(j)
        pairs.append((i, j))
    return pairs


def _same_label(left: str, right: str) -> bool:
    def words(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+", text.lower())) - _STOPWORDS - {"object"}

    return bool(words(left) & words(right))


def _apply_verdict(changes: list[DetectedChange], verdict: Verdict) -> set[int]:
    rejected: set[int] = set()
    for item in verdict.regions:
        if not 0 <= item.index < len(changes):
            continue
        change = changes[item.index]
        if not item.real_change or item.change_type == "none":
            rejected.add(item.index)
            change.attributes["rejected_as"] = item.label
            continue
        change.label = item.label or change.label
        if (
            item.change_type != change.change_type
            and change.attributes.get("cue") == "appearance"
        ):
            _flip(change)
    return rejected


def _flip(change: DetectedChange) -> None:
    if change.change_type == "removed":
        change.change_type = "added"
        change.to_centroid, change.from_centroid = change.from_centroid, None
    else:
        change.change_type = "removed"
        change.from_centroid, change.to_centroid = change.to_centroid, None


def _ask_gemini(
    changes: list[DetectedChange],
    indexes: list[int],
    views: dict[int, list[RegionView]],
    cfg: VlmConfig,
    debug_dir: Path | None,
) -> Verdict:
    from google import genai
    from google.genai import types

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    geometry = []
    contents: list = [_PROMPT]
    for index in indexes:
        change = changes[index]
        point = (
            change.from_centroid
            if change.from_centroid is not None
            else change.to_centroid
        )
        geometry.append(
            {
                "index": index,
                "geometric_type": change.change_type,
                "cue": change.attributes.get("cue", "geometry"),
                "centroid": [round(float(v), 3) for v in point],
                "sorted_extent": [round(float(v), 3) for v in np.sort(change.extent)],
            }
        )
        region_id = change.region_ids[0]
        for when, scene_id in (("BEFORE", "v1"), ("AFTER", "v2")):
            contents.append(f"Region {index} {when}:")
            view = best_view(views, region_id, scene_id)
            if view is None:
                contents.append("(no view of this location in this visit)")
                continue
            data = _jpeg(view.crop)
            contents.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
            if debug_dir is not None:
                (debug_dir / f"region_{index:02d}_{when.lower()}.jpg").write_bytes(data)
    contents.append("Geometry: " + json.dumps(geometry))

    response = genai.Client().models.generate_content(
        model=cfg.model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Verdict,
            temperature=0.0,
        ),
    )
    text = response.text or "{}"
    if debug_dir is not None:
        (debug_dir / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
        (debug_dir / "response.json").write_text(text + "\n")
    try:
        return Verdict.model_validate_json(text)
    except ValueError:
        log.warning("Gemini reply did not match the schema: %s", text[:500])
        return Verdict(regions=[])


def _collapse(
    changes: list[DetectedChange],
    pairs: list[tuple[int, int]],
    pair_texts: dict[tuple[int, int], str],
    texts: dict[int, str],
    rejected: set[int],
) -> list[DetectedChange]:
    drop: set[int] = set(rejected)
    moved_at: dict[int, DetectedChange] = {}
    for i, j in pairs:
        removed, added = changes[i], changes[j]
        if removed.change_type != "removed":
            removed, added = added, removed
            i, j = j, i
        displacement = float(np.linalg.norm(added.to_centroid - removed.from_centroid))
        label = removed.label if removed.label != "object" else added.label
        moved = DetectedChange(
            id=0,
            change_type="moved",
            label=label,
            attributes={"cue": removed.attributes.get("cue", "geometry")},
            confidence=1.0,
            region_ids=removed.region_ids + added.region_ids,
            from_centroid=removed.from_centroid,
            to_centroid=added.to_centroid,
            extent=removed.extent,
            displacement_m=displacement,
            description="",
        )
        text = pair_texts.get((i, j), "")
        moved.description = (
            _with_distance(text, displacement) if text else _template(moved, None)
        )
        drop.add(max(i, j))
        moved_at[min(i, j)] = moved

    kept = []
    for index, change in enumerate(changes):
        if index in drop:
            continue
        change = moved_at.get(index, change)
        if change.change_type != "moved":
            change.description = texts.get(index) or _template(change, None)
        kept.append(change)
    for number, change in enumerate(kept, start=1):
        change.id = number
    return kept


def _with_distance(text: str, metres: float) -> str:
    rendered = f"{metres:.1f} m"
    updated, count = re.subn(r"\d+(?:\.\d+)?\s*m\b", rendered, text, count=1)
    return updated if count else f"{text.rstrip('.')} ({rendered})."


def _jpeg(crop: np.ndarray) -> bytes:
    image = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    longest = max(image.shape[:2])
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Could not encode crop as JPEG")
    return encoded.tobytes()
