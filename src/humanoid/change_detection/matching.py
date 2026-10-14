import numpy as np
from scipy.optimize import linear_sum_assignment

from humanoid.config import MatchConfig
from humanoid.types import ChangeRegion, DetectedChange, RegionLabel


def match_moved(
    regions: list[ChangeRegion],
    labels: dict[int, RegionLabel],
    embeds: dict[int, np.ndarray],
    cfg: MatchConfig,
) -> list[DetectedChange]:
    removed = [region for region in regions if region.change_type == "removed"]
    added = [region for region in regions if region.change_type == "added"]
    pairs = _assign(removed, added, labels, embeds, cfg)
    used = {region_id for left, right, _cost in pairs for region_id in (left.id, right.id)}

    changes = [_moved(left, right, labels, cost) for left, right, cost in pairs]
    changes.extend(_single(region, labels) for region in regions if region.id not in used)
    changes.sort(key=lambda change: change.region_ids[0])
    for index, change in enumerate(changes, start=1):
        change.id = index
    return changes


def _assign(removed, added, labels, embeds, cfg):
    if not removed or not added:
        return []
    cost = np.array([
        [_pair_cost(left, right, labels, embeds, cfg) for right in added]
        for left in removed
    ])
    rows, cols = linear_sum_assignment(cost)
    pairs = []
    for row, col in zip(rows, cols):
        value = float(cost[row, col])
        if value <= cfg.max_cost:
            pairs.append((removed[row], added[col], value))
    return pairs


def _pair_cost(left, right, labels, embeds, cfg) -> float:
    return (
        cfg.weight_label * _label_cost(labels, left.id, right.id)
        + cfg.weight_embed * _embed_cost(embeds, left.id, right.id)
        + cfg.weight_size * _size_cost(left.extent, right.extent)
    )


def _label_cost(labels, left_id, right_id) -> float:
    left = _name(labels, left_id)
    right = _name(labels, right_id)
    if left == right:
        return 0.0
    if left == "unknown object" or right == "unknown object":
        return 0.5
    return 1.0


def _name(labels, region_id) -> str:
    return _raw(labels, region_id).casefold()


def _raw(labels, region_id) -> str:
    item = labels.get(region_id)
    if item is None or not item.label.strip():
        return "unknown object"
    return item.label.strip()


def _embed_cost(embeds, left_id, right_id) -> float:
    left = embeds.get(left_id)
    right = embeds.get(right_id)
    if left is None or right is None:
        return 0.5
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return 1.0
    cosine = float(np.dot(left, right) / denom)
    return float(np.clip((1.0 - cosine) / 2.0, 0.0, 1.0))


def _size_cost(left, right) -> float:
    scale = np.maximum(np.maximum(left, right), 1e-6)
    return float(np.clip(np.mean(np.abs(left - right) / scale), 0.0, 1.0))


def _display_label(labels, *region_ids) -> str:
    names = [_raw(labels, region_id) for region_id in region_ids]
    for name in names:
        if name.casefold() != "unknown object":
            return name
    return names[0]


def _attributes(labels, region_id) -> dict[str, str]:
    item = labels.get(region_id)
    return dict(item.attributes) if item is not None else {}


def _moved(left, right, labels, cost) -> DetectedChange:
    return DetectedChange(
        id=0,
        change_type="moved",
        label=_display_label(labels, left.id, right.id),
        attributes=_attributes(labels, left.id),
        confidence=float(np.clip(1.0 - cost, 0.0, 1.0)),
        region_ids=(left.id, right.id),
        from_centroid=np.asarray(left.centroid, dtype=float),
        to_centroid=np.asarray(right.centroid, dtype=float),
        extent=np.asarray(left.extent, dtype=float),
        displacement_m=float(np.linalg.norm(right.centroid - left.centroid)),
        description="",
    )


def _single(region, labels) -> DetectedChange:
    centroid = np.asarray(region.centroid, dtype=float)
    removed = region.change_type == "removed"
    return DetectedChange(
        id=0,
        change_type=region.change_type,
        label=_display_label(labels, region.id),
        attributes=_attributes(labels, region.id),
        confidence=1.0,
        region_ids=(region.id,),
        from_centroid=centroid if removed else None,
        to_centroid=None if removed else centroid,
        extent=np.asarray(region.extent, dtype=float),
        displacement_m=None,
        description="",
    )