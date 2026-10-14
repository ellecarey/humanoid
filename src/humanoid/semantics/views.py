from pathlib import Path

import cv2
import numpy as np

from humanoid.preprocessing.frame_selection import blur_score
from humanoid.types import ChangeRegion, Frame, RegionView, SceneRecon
from humanoid.utils.geometry import project

DEPTH_MARGIN_M = 0.05
MIN_VISIBLE_FRACTION = 0.5
MAX_VIEWS = 3
CROP_PAD = 0.1


def select_views(
    regions: list[ChangeRegion],
    frames: dict[str, list[Frame]],
    recon: SceneRecon,
    depths: dict[str, dict[str, np.ndarray]],
) -> dict[int, list[RegionView]]:
    lookup = {
        f"{scene_id}/{frame.path.name}": frame
        for scene_id, scene_frames in frames.items()
        for frame in scene_frames
    }
    return {region.id: _views_for(region, lookup, recon, depths) for region in regions}


def _views_for(region, lookup, recon, depths) -> list[RegionView]:
    ranked: list[RegionView] = []
    for image_name, pose in recon.poses.items():
        if pose.scene_id != region.scene_id:
            continue
        frame = lookup.get(image_name)
        depth_map = depths.get(pose.scene_id, {}).get(image_name)
        if frame is None or depth_map is None or not Path(frame.path).is_file():
            continue
        view = _score_view(region, frame, image_name, pose, depth_map)
        if view is not None:
            ranked.append(view)
    ranked.sort(key=lambda view: view.score, reverse=True)
    return ranked[:MAX_VIEWS]


def _score_view(region, frame, image_name, pose, depth_map) -> RegionView | None:
    pixels, z, _ = project(region.points, pose.R, pose.t, pose.K)
    if len(pixels) < 4:
        return None
    visible = _depth_agrees(pixels, z, depth_map)
    if int(visible.sum()) < 4 or float(visible.mean()) < MIN_VISIBLE_FRACTION:
        return None

    image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    crop = _crop(image, pixels[visible])
    if crop.size == 0:
        return None

    chosen = pixels[visible]
    area = float(
        (chosen[:, 0].max() - chosen[:, 0].min())
        * (chosen[:, 1].max() - chosen[:, 1].min())
    )
    score = area * blur_score(crop) * float(visible.mean())
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return RegionView(region.id, image_name, rgb, score)


def _depth_agrees(pixels: np.ndarray, z: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    height, width = depth_map.shape
    u = np.rint(pixels[:, 0]).astype(int)
    v = np.rint(pixels[:, 1]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    z_map = np.full(len(pixels), np.nan)
    z_map[inside] = depth_map[v[inside], u[inside]]
    return np.isfinite(z_map) & (np.abs(z_map - z) <= DEPTH_MARGIN_M)


def _crop(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = float(pixels[:, 0].min())
    y0 = float(pixels[:, 1].min())
    x1 = float(pixels[:, 0].max())
    y1 = float(pixels[:, 1].max())
    pad_x = CROP_PAD * (x1 - x0)
    pad_y = CROP_PAD * (y1 - y0)
    left = max(0, int(np.floor(x0 - pad_x)))
    top = max(0, int(np.floor(y0 - pad_y)))
    right = min(width, int(np.ceil(x1 + pad_x)) + 1)
    bottom = min(height, int(np.ceil(y1 + pad_y)) + 1)
    return image[top:bottom, left:right]