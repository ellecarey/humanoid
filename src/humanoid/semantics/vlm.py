import json
import os
import re

import cv2
import numpy as np

from humanoid.change_detection.matching import build_changes
from humanoid.config import VlmConfig
from humanoid.semantics.describe import _template
from humanoid.types import ChangeRegion, DetectedChange, RegionView

EXTENT_TOLERANCE_M = 0.05

_PROMPT = """These crops are regions that differ between two visits to the same room.
Each image is preceded by its index. The JSON list gives every index a centroid and a sorted_extent, in metres.
Reply with JSON only:
{"regions": {"0": "<short noun phrase for index 0>"},
 "moved_pairs": [[0, 1]],
 "descriptions": {"0-1": "<where the object was, where it is now, and how it changed>"},
 "summary": "<one sentence>"}
regions must include every index.
moved_pairs lists [removed_index, added_index] only when both crops are the same object that moved.
descriptions has one sentence per pair, keyed "removed-added". Do not invent a distance; the measured displacement is added later.
summary covers every region, including ones that were only removed or only added."""


def interpret_regions(
    regions: list[ChangeRegion],
    views: dict[int, list[RegionView]],
    cfg: VlmConfig,
) -> list[DetectedChange]:
    changes = build_changes(regions)
    if not os.environ.get("GEMINI_API_KEY"):
        return _collapse(changes, _offline_pairs(changes), descriptions={})
    pairs, descriptions = _from_gemini(changes, views, cfg)
    return _collapse(changes, pairs, descriptions)


def accept_pair(
    changes: list[DetectedChange],
    i: int,
    j: int,
    tolerance: float = EXTENT_TOLERANCE_M,
) -> bool:
    if i == j or not (0 <= i < len(changes) and 0 <= j < len(changes)):
        return False
    left, right = changes[i], changes[j]
    kinds = {left.change_type, right.change_type}
    if kinds != {"removed", "added"}:
        return False
    return _extent_cost(left.extent, right.extent) <= tolerance


def _extent_cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.sort(a) - np.sort(b))))


def _offline_pairs(changes: list[DetectedChange]) -> list[tuple[int, int]]:
    added = [j for j, change in enumerate(changes) if change.change_type == "added"]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, change in enumerate(changes):
        if change.change_type != "removed":
            continue
        choices = []
        for j in added:
            if j in used or not accept_pair(changes, i, j):
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


def _from_gemini(
    changes: list[DetectedChange],
    views: dict[int, list[RegionView]],
    cfg: VlmConfig,
) -> tuple[list[tuple[int, int]], dict]:
    from google import genai
    from google.genai import types

    geometry = []
    contents: list = [_PROMPT]
    for index, change in enumerate(changes):
        point = (
            change.from_centroid
            if change.from_centroid is not None
            else change.to_centroid
        )
        geometry.append(
            {
                "index": index,
                "centroid": [round(float(v), 3) for v in point],
                "sorted_extent": [round(float(v), 3) for v in np.sort(change.extent)],
            }
        )
        crop = _crop(views, change.region_ids[0])
        if crop is not None:
            contents.append(f"index {index}")
            contents.append(
                types.Part.from_bytes(data=_jpeg(crop), mime_type="image/jpeg")
            )
    contents.append(json.dumps(geometry))
    response = genai.Client().models.generate_content(
        model=cfg.model,
        contents=contents,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    payload = json.loads(response.text or "{}")
    for key, label in (payload.get("regions") or {}).items():
        index = int(key)
        if 0 <= index < len(changes):
            changes[index].label = str(label)
    pairs = []
    for pair in payload.get("moved_pairs") or []:
        i, j = int(pair[0]), int(pair[1])
        if accept_pair(changes, i, j):
            pairs.append((i, j))
    return pairs, payload.get("descriptions") or {}


def _collapse(
    changes: list[DetectedChange],
    pairs: list[tuple[int, int]],
    descriptions: dict | None = None,
) -> list[DetectedChange]:
    descriptions = descriptions or {}
    drop: set[int] = set()
    moved_at: dict[int, DetectedChange] = {}
    for i, j in pairs:
        removed, added = changes[i], changes[j]
        if removed.change_type != "removed":
            removed, added = added, removed
            i, j = j, i
        displacement = float(np.linalg.norm(added.to_centroid - removed.from_centroid))
        text = descriptions.get(f"{i}-{j}") or descriptions.get(f"{j}-{i}") or ""
        moved = DetectedChange(
            id=0,
            change_type="moved",
            label=removed.label,
            attributes={},
            confidence=1.0,
            region_ids=removed.region_ids + added.region_ids,
            from_centroid=removed.from_centroid,
            to_centroid=added.to_centroid,
            extent=removed.extent,
            displacement_m=displacement,
            description="",
        )
        moved.description = (
            _with_distance(text, displacement) if text else _template(moved, None)
        )
        lo, hi = min(i, j), max(i, j)
        drop.add(hi)
        moved_at[lo] = moved
    kept = [
        moved_at.get(index, change)
        for index, change in enumerate(changes)
        if index not in drop
    ]
    for number, change in enumerate(kept, start=1):
        change.id = number
    return kept


def _with_distance(text: str, metres: float) -> str:
    rendered = f"{metres:.1f} m"
    updated, count = re.subn(r"\d+(?:\.\d+)?\s*m", rendered, text, count=1)
    return updated if count else f"{text.rstrip('.')} ({rendered})."


def _crop(views: dict[int, list[RegionView]], region_id: int) -> np.ndarray | None:
    crops = views.get(region_id) or []
    return crops[0].crop if crops else None


def _jpeg(crop: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode crop as JPEG")
    return encoded.tobytes()
