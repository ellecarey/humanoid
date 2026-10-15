from pathlib import Path

import yaml
from pydantic import BaseModel


class FramesConfig(BaseModel):
    target_fps: float
    max_frames: int
    max_width: int
    blur_threshold: float
    duplicate_threshold: float
    min_overlap: float


class ReconConfig(BaseModel):
    model: str
    max_frames: int
    voxel_size: float
    min_confidence: float
    scale_m_per_unit: float


class DiffConfig(BaseModel):
    change_distance_m: float


class VisibilityConfig(BaseModel):
    depth_margin_m: float
    min_confirming_views: int
    min_confidence: float = 0.0


class ClusterConfig(BaseModel):
    dbscan_eps_m: float
    dbscan_min_samples: int
    min_points: int
    min_extent_m: float
    max_extent_m: float
    min_height_above_floor_m: float


class VlmConfig(BaseModel):
    provider: str
    model: str
    cache: bool


class Config(BaseModel):
    run_name: str
    videos: dict[str, Path]
    output_root: Path
    frames: FramesConfig
    recon: ReconConfig
    diff: DiffConfig
    visibility: VisibilityConfig
    cluster: ClusterConfig
    vlm: VlmConfig


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)
