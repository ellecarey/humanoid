import numpy as np

from humanoid.config import VisibilityConfig
from humanoid.types import GeometricDiff, LabeledDiff, SceneRecon
from humanoid.utils.geometry import depth_margin, project, sample_nearest

RESIDUAL_CLIP_M = 0.5


def apply_visibility(
    diff: GeometricDiff,
    recon: SceneRecon,
    depths: dict[str, dict[str, np.ndarray]],
    cfg: VisibilityConfig,
    features: dict[str, dict[str, np.ndarray]] | None = None,
) -> LabeledDiff:
    """Vote each point against every view of the other video.

    free: the other video sees past the point, so the surface is gone there.
    agree: the other video sees a surface at the point's depth.
    The appearance path compares feature vectors of agreeing views, which finds
    changes that sit almost flush with the surface behind them.
    """
    count = len(diff.points)
    free = np.zeros(count, dtype=int)
    agree = np.zeros(count, dtype=int)
    residual_sum = np.zeros(count)
    residual_n = np.zeros(count, dtype=int)
    tally = _FeatureTally(count, features, center=cfg.center_features)

    for image_name, pose in recon.poses.items():
        depth_map = depths.get(pose.scene_id, {}).get(image_name)
        if depth_map is None or count == 0:
            continue
        pixels, z, keep = project(diff.points, pose.R, pose.t, pose.K, depth_map.shape)
        index = np.flatnonzero(keep)
        if len(index) == 0:
            continue
        observed = sample_nearest(depth_map, pixels).astype(float)
        valid = np.isfinite(observed) & (observed > 0)
        confidence_map = recon.confidence.get(pose.scene_id, {}).get(image_name)
        if confidence_map is not None and cfg.min_confidence > 0:
            valid &= sample_nearest(confidence_map, pixels) >= cfg.min_confidence
        margin = depth_margin(z, cfg.depth_margin_m, cfg.depth_margin_rel)
        sees_free = valid & (observed > z + margin)
        sees_same = valid & (np.abs(observed - z) <= margin)
        other = diff.source_scene[index] != pose.scene_id

        free[index[other & sees_free]] += 1
        agree[index[other & sees_same]] += 1
        informative = other & (sees_free | sees_same)
        residual_sum[index[informative]] += np.clip(
            observed[informative] - z[informative], -RESIDUAL_CLIP_M, RESIDUAL_CLIP_M
        )
        residual_n[index[informative]] += 1
        tally.add(
            pose.scene_id, image_name, pixels, depth_map.shape, index, sees_same, other
        )

    cosine = tally.cosine()
    observed_views = free + agree
    geometric = (
        diff.candidate
        & (free >= cfg.min_free_views)
        & (free >= cfg.free_ratio * np.maximum(observed_views, 1))
    )
    looks_different = (
        np.isfinite(cosine)
        & (agree >= cfg.min_agree_views)
        & (cosine < cfg.appearance_cos)
        & ~geometric
    )
    surface_kept = (agree >= cfg.min_agree_views) & (agree > free)

    label = np.full(count, "unknown", dtype=object)
    label[~diff.candidate | surface_kept] = "unchanged"
    label[geometric | looks_different] = "changed"
    cue = np.full(count, "", dtype=object)
    cue[geometric] = "geometry"
    cue[looks_different] = "appearance"
    with np.errstate(invalid="ignore", divide="ignore"):
        residual = np.where(residual_n > 0, residual_sum / residual_n, np.nan)
    return LabeledDiff(
        points=diff.points,
        source_scene=diff.source_scene,
        nn_distance=diff.nn_distance,
        label=label,
        colors=diff.colors,
        cue=cue,
        free_views=free,
        agree_views=agree,
        residual=residual,
        appearance=cosine,
    )


class _FeatureTally:
    """Running mean feature of each point in its own video and in the other video.

    With `center`, each video's mean feature is subtracted first. A global
    exposure or white-balance difference between the videos shifts every feature
    the same way, and would otherwise lower the cosine of unchanged surfaces.
    """

    def __init__(self, count: int, features, center: bool = False) -> None:
        self.features = features or {}
        self.means = (
            {
                scene_id: _mean_feature(by_name)
                for scene_id, by_name in self.features.items()
            }
            if center
            else {}
        )
        dim = self._dim()
        self.own = np.zeros((count, dim), dtype=np.float32) if dim else None
        self.other = np.zeros((count, dim), dtype=np.float32) if dim else None
        self.own_n = np.zeros(count, dtype=int)
        self.other_n = np.zeros(count, dtype=int)

    def _dim(self) -> int:
        for by_name in self.features.values():
            for grid in by_name.values():
                return int(grid.shape[-1])
        return 0

    def add(self, scene_id, image_name, pixels, shape, index, sees_same, other):
        if self.own is None:
            return
        grid = self.features.get(scene_id, {}).get(image_name)
        if grid is None:
            return
        sampled = sample_nearest(grid, pixels, shape).astype(np.float32)
        mean = self.means.get(scene_id)
        if mean is not None:
            sampled -= mean
        own = sees_same & ~other
        theirs = sees_same & other
        self.own[index[own]] += sampled[own]
        self.own_n[index[own]] += 1
        self.other[index[theirs]] += sampled[theirs]
        self.other_n[index[theirs]] += 1

    def cosine(self) -> np.ndarray:
        count = len(self.own_n)
        if self.own is None:
            return np.full(count, np.nan)
        result = np.full(count, np.nan)
        seen = (self.own_n > 0) & (self.other_n > 0)
        own = self.own[seen]
        other = self.other[seen]
        own /= np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-6)
        other /= np.maximum(np.linalg.norm(other, axis=1, keepdims=True), 1e-6)
        result[seen] = np.sum(own * other, axis=1)
        return result


def _mean_feature(by_name: dict[str, np.ndarray]) -> np.ndarray | None:
    grids = [
        grid.reshape(-1, grid.shape[-1]).astype(np.float32) for grid in by_name.values()
    ]
    if not grids:
        return None
    return np.concatenate(grids).mean(axis=0)
