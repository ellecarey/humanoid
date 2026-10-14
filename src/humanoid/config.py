from pathlib import Path

import yaml
from pydantic import BaseModel


class FramesConfig(BaseModel):
    target_fps: float
    max_width: int
    blur_threshold: float
    duplicate_threshold: float


class SfmConfig(BaseModel):
    feature: str
    matcher: str
    retrieval: str
    min_registration_ratio: float


class DepthConfig(BaseModel):
    model: str
    min_sfm_points: int


class FusionConfig(BaseModel):
    voxel_size: float
    sdf_trunc: float


class DiffConfig(BaseModel):
    change_distance_m: float


class VisibilityConfig(BaseModel):
    depth_margin_m: float
    min_confirming_views: int


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


class EmbedConfig(BaseModel):
    model: str


class MatchConfig(BaseModel):
    max_cost: float
    weight_label: float
    weight_embed: float
    weight_size: float


class DescribeConfig(BaseModel):
    use_llm: bool
    provider: str
    model: str


class VizConfig(BaseModel):
    rerun: bool


class EvaluationConfig(BaseModel):
    ground_truth: Path
    match_distance_m: float


class Config(BaseModel):
    run_name: str
    videos: dict[str, Path]
    output_root: Path
    frames: FramesConfig
    sfm: SfmConfig
    depth: DepthConfig
    fusion: FusionConfig
    diff: DiffConfig
    visibility: VisibilityConfig
    cluster: ClusterConfig
    vlm: VlmConfig
    embed: EmbedConfig
    match: MatchConfig
    describe: DescribeConfig
    viz: VizConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> Config:
    data = yaml.safe_load(path.read_text())
    return Config.model_validate(data)