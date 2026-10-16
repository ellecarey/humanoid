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
        for label in sorted(set(labels) - {-1}):
            members = labels == label
            region = _region(len(regions), scene_id, change_type, points[members])
            if region is None or not _keep(region, floor, cfg):
                continue
            appearance = np.mean(point_cue[members] == "appearance")
            region.cue = "appearance" if appearance > 0.5 else "geometry"
            regions.append(region)
            residuals.append(_finite_mean(point_residual[members]))
    return _merge_appearance(regions, residuals, cfg.merge_dist_m)


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


def _keep(region: ChangeRegion, floor: np.ndarray | None, cfg: ClusterConfig) -> bool:
    longest = float(np.max(region.extent))
    if (
        len(region.points) < cfg.min_points
        or longest < cfg.min_extent_m
        or longest > cfg.max_extent_m
    ):
        return False
    if floor is None:
        return True
    normal = floor[:3]
    height = abs(float(normal @ region.centroid + floor[3])) / np.linalg.norm(normal)
    return height >= cfg.min_height_above_floor_m
