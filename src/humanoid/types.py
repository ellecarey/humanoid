from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    scene_id: str
    path: Path
    index: int
    width: int
    height: int
    time_s: float = 0.0


@dataclass
class CameraPose:
    scene_id: str
    image_name: str
    R: np.ndarray  # 3x3 world-to-camera rotation
    t: np.ndarray  # shape (3,) world-to-camera translation
    K: np.ndarray  # 3x3 intrinsics at the model image resolution


@dataclass
class SceneRecon:
    poses: dict[str, CameraPose]  # key: "v1/frame_000006.jpg"
    depths: dict[str, dict[str, np.ndarray]]  # scene_id -> image name -> (H, W) metres
    confidence: dict[str, dict[str, np.ndarray]]  # scene_id -> image name -> (H, W)
    images: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # RGB uint8
    # image name -> (scale_x, scale_y, offset_x, offset_y); frame = (model + offset) / scale
    frame_maps: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )
    alignment: dict[str, float] = field(default_factory=dict)


@dataclass
class ColoredCloud:
    points: np.ndarray  # (N, 3)
    colors: np.ndarray  # (N, 3) float in [0, 1]


@dataclass
class ChangeRegion:
    id: int
    change_type: str  # "removed" or "added"
    scene_id: str
    points: np.ndarray  # (M, 3)
    centroid: np.ndarray  # (3,)
    extent: np.ndarray  # (3,) oriented bbox size in metres
    obb_rotation: np.ndarray  # 3x3
    cue: str = "geometry"  # "geometry" or "appearance"
    score: float = 1.0  # mean vote strength of the member points, 0..1
    base_m: float | None = None  # lowest part above the floor
    height_m: float | None = None  # centroid above the floor


@dataclass
class GeometricDiff:
    points: np.ndarray  # (N, 3) world coordinates from both clouds
    source_scene: np.ndarray  # (N,) "v1" or "v2"
    nn_distance: np.ndarray  # (N,) metres to the nearest point in the other cloud
    candidate: np.ndarray  # (N,) True when nn_distance exceeds the threshold
    colors: np.ndarray | None = None  # (N, 3) float in [0, 1]


@dataclass
class LabeledDiff:
    points: np.ndarray
    source_scene: np.ndarray
    nn_distance: np.ndarray
    label: np.ndarray  # (N,) "changed", "unchanged", or "unknown"
    colors: np.ndarray | None = None
    cue: np.ndarray | None = None  # (N,) "geometry", "appearance", or ""
    free_views: np.ndarray | None = None  # other-video views that see through the point
    agree_views: np.ndarray | None = None  # other-video views that see the same surface
    residual: np.ndarray | None = None  # mean (other depth - point depth), metres
    appearance: np.ndarray | None = (
        None  # feature cosine to the other video, nan if unseen
    )


@dataclass
class RegionView:
    region_id: int
    image_name: str
    crop: np.ndarray  # RGB uint8, region boxed
    score: float
    scene_id: str = ""
    plain: np.ndarray | None = None  # same crop without the box drawn
    box: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1 inside the crop


@dataclass
class RegionLabel:
    region_id: int
    label: str
    attributes: dict[str, str]


@dataclass
class DetectedChange:
    id: int
    change_type: str  # "removed", "added", or "moved"
    label: str
    attributes: dict[str, str]
    confidence: float
    region_ids: tuple[int, ...]
    from_centroid: np.ndarray | None
    to_centroid: np.ndarray | None
    extent: np.ndarray
    displacement_m: float | None
    description: str
    # "from" / "to" -> (base_m, height_m) above the floor
    heights: dict[str, tuple[float, float]] = field(default_factory=dict)
