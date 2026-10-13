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
    poses: dict[str, CameraPose]
    points_xyz: np.ndarray  # (N, 3) sparse points in the shared world frame
    point_image_ids: np.ndarray  # (N,) image index for each point

@dataclass
class ChangeRegion:
    id: int
    change_type: str  # "removed" or "added"
    scene_id: str
    points: np.ndarray  # (M, 3)
    centroid: np.ndarray  # (3,)
    extent: np.ndarray  # (3,) oriented bbox size in metres
    obb_rotation: np.ndarray  # 3x3