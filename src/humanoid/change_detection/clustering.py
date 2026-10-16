import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

from humanoid.config import ClusterConfig
from humanoid.types import ChangeRegion, LabeledDiff

FLOOR_DISTANCE_M = 0.02
FLOOR_ATTEMPTS = 6
FLOOR_SAMPLE = 200_000


def cluster_changes(
    diff: LabeledDiff, cfg: ClusterConfig, up: np.ndarray | None = None
) -> list[ChangeRegion]:
    changed = diff.label == "changed"
    floor = _floor_plane(diff.points[diff.label == "unchanged"], up, cfg)
    count = len(diff.points)
    cue = diff.cue if diff.cue is not None else np.full(count, "geometry")
    residual = diff.residual if diff.residual is not None else np.zeros(count)
    scores = _point_scores(diff, cue)
    regions: list[ChangeRegion] = []
    residuals: list[float] = []

    for scene_id, change_type in (("v1", "removed"), ("v2", "added")):
        mask = changed & (diff.source_scene == scene_id)
        points = diff.points[mask]
        if len(points) == 0:
            continue
        labels = DBSCAN(
            eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples
        ).fit_predict(points)
        point_cue = cue[mask]
        point_residual = residual[mask]
        point_score = scores[mask]
        for label in sorted(set(labels) - {-1}):
            members = labels == label
            region = _region(len(regions), scene_id, change_type, points[members])
            if region is None:
                continue
            heights = _heights(region, floor, up)
            if not _keep(region, heights, cfg):
                continue
            if heights is not None:
                region.base_m, region.height_m = heights
            appearance = np.mean(point_cue[members] == "appearance")
            region.cue = "appearance" if appearance > 0.5 else "geometry"
            region.score = float(np.mean(point_score[members]))
            regions.append(region)
            residuals.append(_finite_mean(point_residual[members]))
    return _merge_appearance(regions, residuals, cfg.merge_dist_m)


def _point_scores(diff: LabeledDiff, cue: np.ndarray) -> np.ndarray:
    """Vote strength per point: the free-space share, or 1 - cosine for appearance."""
    count = len(diff.points)
    if diff.free_views is None or diff.agree_views is None:
        return np.ones(count)
    observed = diff.free_views + diff.agree_views
    scores = diff.free_views / np.maximum(observed, 1)
    if diff.appearance is not None:
        appearance = np.clip(1.0 - np.nan_to_num(diff.appearance, nan=1.0), 0.0, 1.0)
        scores = np.where(cue == "appearance", appearance, scores)
    return scores.astype(float)


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if len(finite) else 0.0


def unknown_regions(diff: LabeledDiff, cfg: ClusterConfig) -> list[ChangeRegion]:
    unknown = diff.label == "unknown"
    regions: list[ChangeRegion] = []

    for scene_id in ("v1", "v2"):
        points = diff.points[unknown & (diff.source_scene == scene_id)]
        if len(points) == 0:
            continue
        labels = DBSCAN(
            eps=cfg.dbscan_eps_m, min_samples=cfg.dbscan_min_samples
        ).fit_predict(points)
        for label in sorted(set(labels) - {-1}):
            cluster = points[labels == label]
            region = _region(len(regions), scene_id, "unknown", cluster)
            if region is None or not _keep(region, None, cfg):
                continue
            regions.append(region)
    return regions


def _merge_appearance(
    regions: list[ChangeRegion], residuals: list[float], merge_dist_m: float
) -> list[ChangeRegion]:
    """Collapse a co-located removed/added pair where appearance drove either side.

    A flat object shows up twice: its own texture in one video and the surface it
    covers in the other. The object side protrudes, so the other video's depth lies
    behind it and its mean residual is the larger one.
    """
    drop: set[int] = set()
    for i, left in enumerate(regions):
        if i in drop or left.change_type != "removed":
            continue
        best, best_distance = None, merge_dist_m
        for j, right in enumerate(regions):
            if j in drop or right.change_type != "added":
                continue
            if "appearance" not in (left.cue, right.cue):
                continue
            distance = float(np.linalg.norm(left.centroid - right.centroid))
            if distance <= best_distance:
                best, best_distance = j, distance
        if best is None:
            continue
        background = best if residuals[i] >= residuals[best] else i
        drop.add(background)
        regions[i if background == best else best].cue = "appearance"
    kept = [region for index, region in enumerate(regions) if index not in drop]
    for number, region in enumerate(kept):
        region.id = number
    return kept


def _floor_plane(
    points: np.ndarray, up: np.ndarray | None, cfg: ClusterConfig
) -> np.ndarray | None:
    """Lowest roughly horizontal plane; the largest plane in a room may be a wall."""
    if len(points) < 3:
        return None
    if len(points) > FLOOR_SAMPLE:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), FLOOR_SAMPLE, replace=False)]
    remaining = points
    best, best_height = None, np.inf
    min_cos = np.cos(np.radians(cfg.floor_max_tilt_deg))
    for _ in range(FLOOR_ATTEMPTS if up is not None else 1):
        if len(remaining) < 3:
            break
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(remaining)
        plane, inliers = cloud.segment_plane(
            distance_threshold=FLOOR_DISTANCE_M, ransac_n=3, num_iterations=1000
        )
        plane = np.asarray(plane, dtype=float)
        if up is None:
            return plane
        normal = plane[:3] / np.linalg.norm(plane[:3])
        if abs(float(normal @ up)) >= min_cos:
            height = float(np.median(remaining[inliers] @ up))
            if height < best_height:
                best, best_height = plane, height
        remaining = np.delete(remaining, inliers, axis=0)
    return best


def _region(
    region_id: int, scene_id: str, change_type: str, points: np.ndarray
) -> ChangeRegion | None:
    if len(points) < 4:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    try:
        box = cloud.get_oriented_bounding_box(robust=True)
    except RuntimeError:
        return None
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id=scene_id,
        points=points,
        centroid=np.asarray(box.center, dtype=float),
        extent=np.asarray(box.extent, dtype=float),
        obb_rotation=np.asarray(box.R, dtype=float),
    )


def _heights(
    region: ChangeRegion, floor: np.ndarray | None, up: np.ndarray | None
) -> tuple[float, float] | None:
    """(lowest part, centroid) above the floor, in metres."""
    if floor is None:
        return None
    normal, offset = floor[:3], float(floor[3])
    scale = float(np.linalg.norm(normal))
    centre = (float(normal @ region.centroid) + offset) / scale
    upward = float(normal @ up) if up is not None else centre
    sign = 1.0 if upward >= 0 else -1.0
    above = sign * (region.points @ normal + offset) / scale
    base = max(0.0, float(np.percentile(above, 2)))
    return base, sign * centre


def _keep(
    region: ChangeRegion, heights: tuple[float, float] | None, cfg: ClusterConfig
) -> bool:
    longest = float(np.max(region.extent))
    if (
        len(region.points) < cfg.min_points
        or longest < cfg.min_extent_m
        or longest > cfg.max_extent_m
    ):
        return False
    if heights is None:
        return True
    return heights[1] >= cfg.min_height_above_floor_m
