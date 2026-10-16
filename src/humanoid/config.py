from pathlib import Path

import yaml
from pydantic import BaseModel


class FramesConfig(BaseModel):
    target_fps: float
    window_s: float
    max_per_video: int
    max_width: int
    duplicate_threshold: float
    min_relative_sharpness: float = 0.0
    max_gap_s: float = 1.0


class ReconConfig(BaseModel):
    model: str
    voxel_size: float
    conf_percentile: float = 10.0
    min_confidence: float = 0.0
    scale_m_per_unit: float = 1.0  # VGGT fallback only; MapAnything is already metric
    icp: bool = True
    icp_voxels_m: list[float] = [0.05, 0.02, 0.01]
    min_self_views: int = 2
    self_margin_m: float = 0.04
    self_margin_rel: float = 0.025


class FeaturesConfig(BaseModel):
    enabled: bool = True
    model: str = "dinov2_vits14"
    scale: float = 2.0
    pca_dim: int = 64


class DiffConfig(BaseModel):
    change_distance_m: float


class VisibilityConfig(BaseModel):
    depth_margin_m: float
    depth_margin_rel: float = 0.0
    min_free_views: int = 2
    free_ratio: float = 0.5
    min_agree_views: int = 2
    appearance_cos: float = 0.5
    min_confidence: float = 0.0
    center_features: bool = False  # subtract each video's mean feature (exposure shift)


class ClusterConfig(BaseModel):
    dbscan_eps_m: float
    dbscan_min_samples: int
    min_points: int
    min_extent_m: float
    max_extent_m: float
    min_height_above_floor_m: float
    merge_dist_m: float = 0.25
    floor_max_tilt_deg: float = 20.0


class VlmConfig(BaseModel):
    provider: str
    model: str
    cache: bool
    pair_abs_tol_m: float = 0.15
    pair_rel_tol: float = 0.3
    retries: int = 3
    retry_wait_s: float = 10.0


class LabelerConfig(BaseModel):
    enabled: bool = True
    model: str = "openai/clip-vit-large-patch14"
    box_pads: list[float] = [0.15, 0.5]
    min_objectness: float = 0.5
    max_label_ratio: float = 0.7
    max_same_similarity: float = 0.95
    pair_min_similarity: float = 0.75
    object_classes: list[str] | None = None
    background_classes: list[str] | None = None


class Config(BaseModel):
    run_name: str
    videos: dict[str, Path]
    output_root: Path
    ground_truth: Path | None = None
    frames: FramesConfig
    recon: ReconConfig
    features: FeaturesConfig = FeaturesConfig()
    diff: DiffConfig
    visibility: VisibilityConfig
    cluster: ClusterConfig
    vlm: VlmConfig
    labeler: LabelerConfig = LabelerConfig()


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)
