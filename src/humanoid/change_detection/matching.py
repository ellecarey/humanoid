import numpy as np

from humanoid.types import ChangeRegion, DetectedChange


def build_changes(regions: list[ChangeRegion]) -> list[DetectedChange]:
    changes = [
        _single(region) for region in sorted(regions, key=lambda region: region.id)
    ]
    for index, change in enumerate(changes, start=1):
        change.id = index
    return changes


def flip_direction(change: DetectedChange) -> None:
    """Swap removed and added, for a flat change whose object turned out to be on the other side."""
    if change.change_type == "removed":
        change.change_type = "added"
        change.to_centroid, change.from_centroid = change.from_centroid, None
    else:
        change.change_type = "removed"
        change.from_centroid, change.to_centroid = change.to_centroid, None
    swap = {"from": "to", "to": "from"}
    change.heights = {swap[key]: value for key, value in change.heights.items()}


def _single(region: ChangeRegion) -> DetectedChange:
    centroid = np.asarray(region.centroid, dtype=float)
    removed = region.change_type == "removed"
    heights = {}
    if region.base_m is not None and region.height_m is not None:
        heights["from" if removed else "to"] = (region.base_m, region.height_m)
    return DetectedChange(
        id=0,
        change_type=region.change_type,
        label="object",
        attributes={"cue": region.cue},
        confidence=float(region.score),
        region_ids=(region.id,),
        from_centroid=centroid if removed else None,
        to_centroid=None if removed else centroid,
        extent=np.asarray(region.extent, dtype=float),
        displacement_m=None,
        description="",
        heights=heights,
    )
