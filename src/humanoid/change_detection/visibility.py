import numpy as np

from humanoid.config import VisibilityConfig
from humanoid.types import GeometricDiff, LabeledDiff, SceneRecon
from humanoid.utils.geometry import project


def apply_visibility(
    diff: GeometricDiff,
    recon: SceneRecon,
    depths: dict[str, dict[str, np.ndarray]],
    cfg: VisibilityConfig,
) -> LabeledDiff:
    confirms = np.zeros(len(diff.points), dtype=int)

    for image_name, pose in recon.poses.items():
        depth_map = depths.get(pose.scene_id, {}).get(image_name)
        if depth_map is None or len(diff.points) == 0:
            continue

        pixels, z, keep = project(diff.points, pose.R, pose.t, pose.K)
        point_index = np.flatnonzero(keep)
        free = _sees_through(pixels, z, depth_map, cfg.depth_margin_m)

        for local_index, index in enumerate(point_index):
            if not free[local_index] or not diff.candidate[index]:
                continue
            if diff.source_scene[index] == pose.scene_id:
                continue
            confirms[index] += 1

    label = np.where(diff.candidate, "unknown", "unchanged").astype(object)
    label[diff.candidate & (confirms >= cfg.min_confirming_views)] = "changed"
    return LabeledDiff(
        points=diff.points,
        source_scene=diff.source_scene,
        nn_distance=diff.nn_distance,
        label=label,
    )


def _sees_through(pixels, z, depth_map, margin: float) -> np.ndarray:
    height, width = depth_map.shape
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    z_map = np.full(len(pixels), np.nan)
    z_map[inside] = depth_map[v[inside], u[inside]]
    return np.isfinite(z_map) & (z_map > z + margin)