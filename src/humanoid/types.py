from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Frame:
    scene_id: str
    path: Path
    index: int
    width: int
    height: int
    
@dataclass
class CameraPose:
    scene_id: str
    image_name: str
    R: np.ndarray  # 3x3 world-to-camera rotation
    t: np.ndarray  # shape (3,) world-to-camera translation
    K: np.ndarray  # 3x3 intrinsics
    
@dataclass
class SceneRecon:
    poses: dict[str, CameraPose]  # key: "v1/frame_000006.jpg"
    depths: dict[str, dict[str, np.ndarray]]  # scene_id -> image name -> (H, W) metres
    confidence: dict[str, dict[str, np.ndarray]]  # scene_id -> image name -> (H, W)

@dataclass
class ChangeRegion:
    id: int
    change_type: str  # "removed" or "added"
    scene_id: str
    points: np.ndarray  # (M, 3)
    centroid: np.ndarray  # (3,)
    extent: np.ndarray  # (3,) oriented bbox size in metres
    obb_rotation: np.ndarray  # 3x3
    
@dataclass
class GeometricDiff:
    points: np.ndarray        # (N, 3) world coordinates from both clouds
    source_scene: np.ndarray  # (N,) "v1" or "v2"
    nn_distance: np.ndarray   # (N,) metres to the nearest point in the other cloud
    candidate: np.ndarray     # (N,) True when nn_distance exceeds the threshold

@dataclass
class LabeledDiff:
    points: np.ndarray
    source_scene: np.ndarray
    nn_distance: np.ndarray
    label: np.ndarray         # (N,) "changed", "unchanged", or "unknown"
    
@dataclass
class RegionView:
    region_id: int
    image_name: str
    crop: np.ndarray  # RGB uint8
    score: float
    
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