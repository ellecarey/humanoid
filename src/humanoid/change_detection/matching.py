import numpy as np

from humanoid.types import ChangeRegion, DetectedChange


def build_changes(regions: list[ChangeRegion]) -> list[DetectedChange]:
    changes = [
        _single(region) for region in sorted(regions, key=lambda region: region.id)
    ]
    for index, change in enumerate(changes, start=1):
        change.id = index
    return changes


def _single(region: ChangeRegion) -> DetectedChange:
    centroid = np.asarray(region.centroid, dtype=float)
    removed = region.change_type == "removed"
    return DetectedChange(
        id=0,
        change_type=region.change_type,
        label="object",
        attributes={},
        confidence=1.0,
        region_ids=(region.id,),
        from_centroid=centroid if removed else None,
        to_centroid=None if removed else centroid,
        extent=np.asarray(region.extent, dtype=float),
        displacement_m=None,
        description="",
    )
