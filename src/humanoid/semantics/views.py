from pathlib import Path

import cv2
import numpy as np

from humanoid.preprocessing.frame_selection import blur_score
from humanoid.types import ChangeRegion, Frame, RegionView, SceneRecon
from humanoid.utils.geometry import (
    depth_margin,
    project,
    sample_nearest,
    to_frame_pixels,
)

DEPTH_MARGIN_M = 0.05
DEPTH_MARGIN_REL = 0.03
MIN_INSIDE_FRACTION = 0.6
MIN_VISIBLE_FRACTION = 0.5
MAX_VIEWS = 2
CONTEXT_PAD = 0.75
MIN_CROP_PX = 192
BOX_BGR = (0, 255, 0)


def select_views(
    regions: list[ChangeRegion],
    frames: dict[str, list[Frame]],
    recon: SceneRecon,
    depths: dict[str, dict[str, np.ndarray]],
) -> dict[int, list[RegionView]]:
    """Best views of every region from each video, own video and the other one.

    In its own video the region's surface must be visible. In the other video
    the same 3D box must be in view and not hidden behind something closer, so
    the crop shows what is there instead.
    """
    lookup = {
        f"{scene_id}/{frame.path.name}": frame
        for scene_id, scene_frames in frames.items()
        for frame in scene_frames
    }
    scene_ids = sorted({pose.scene_id for pose in recon.poses.values()})
    return {
        region.id: [
            view
            for scene_id in scene_ids
            for view in _views_for(region, scene_id, lookup, recon, depths)
        ]
        for region in regions
    }


def best_view(
    views: dict[int, list[RegionView]] | None, region_id: int | None, scene_id: str
) -> RegionView | None:
    if views is None or region_id is None:
        return None
    for view in views.get(region_id) or []:
        if view.scene_id in ("", scene_id):
            return view
    return None


def _views_for(region, scene_id, lookup, recon, depths) -> list[RegionView]:
    own = region.scene_id == scene_id
    ranked = []
    for image_name, pose in recon.poses.items():
        if pose.scene_id != scene_id:
            continue
        frame = lookup.get(image_name)
        depth_map = depths.get(scene_id, {}).get(image_name)
        if frame is None or depth_map is None or not Path(frame.path).is_file():
            continue
        model_image = recon.images.get(scene_id, {}).get(image_name)
        scored = _score_view(region, pose, depth_map, model_image, own)
        if scored is not None:
            ranked.append((scored[0], image_name, frame, scored[1]))
    ranked.sort(key=lambda item: item[0], reverse=True)

    views = []
    for score, image_name, frame, pixels in ranked[:MAX_VIEWS]:
        image = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame_pixels = to_frame_pixels(pixels, recon.frame_maps.get(image_name))
        crop, plain, box = _boxed_crop(image, frame_pixels)
        if crop.size == 0:
            continue
        views.append(
            RegionView(region.id, image_name, crop, score, scene_id, plain, box)
        )
    return views


def _score_view(region, pose, depth_map, model_image, own: bool):
    pixels, z, keep = project(region.points, pose.R, pose.t, pose.K, depth_map.shape)
    if len(pixels) < 4 or float(keep.mean()) < MIN_INSIDE_FRACTION:
        return None
    observed = sample_nearest(depth_map, pixels).astype(float)
    margin = depth_margin(z, DEPTH_MARGIN_M, DEPTH_MARGIN_REL)
    valid = np.isfinite(observed) & (observed > 0)
    if own:
        visible = valid & (np.abs(observed - z) <= margin)
    else:
        visible = valid & (observed >= z - margin)
    fraction = float(visible.mean())
    if int(visible.sum()) < 4 or fraction < MIN_VISIBLE_FRACTION:
        return None

    x0, y0, x1, y1 = _box(pixels)
    area = max(1.0, (x1 - x0) * (y1 - y0))
    sharpness = 1.0
    if model_image is not None:
        patch = model_image[
            int(y0) : int(np.ceil(y1)) + 1, int(x0) : int(np.ceil(x1)) + 1
        ]
        if patch.size:
            sharpness = blur_score(patch)
    return area * fraction * sharpness, pixels


def _box(pixels: np.ndarray) -> tuple[float, float, float, float]:
    low = np.percentile(pixels, 2, axis=0)
    high = np.percentile(pixels, 98, axis=0)
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def _boxed_crop(image: np.ndarray, pixels: np.ndarray):
    """Context crop with the region boxed, the same crop unboxed, and the box."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = _box(pixels)
    pad_x = max(CONTEXT_PAD * (x1 - x0), (MIN_CROP_PX - (x1 - x0)) / 2, 0.0)
    pad_y = max(CONTEXT_PAD * (y1 - y0), (MIN_CROP_PX - (y1 - y0)) / 2, 0.0)
    left = max(0, int(np.floor(x0 - pad_x)))
    top = max(0, int(np.floor(y0 - pad_y)))
    right = min(width, int(np.ceil(x1 + pad_x)) + 1)
    bottom = min(height, int(np.ceil(y1 + pad_y)) + 1)
    crop = image[top:bottom, left:right].copy()
    if crop.size == 0:
        return crop, None, None
    plain = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    box = (
        max(0, int(x0) - left),
        max(0, int(y0) - top),
        min(crop.shape[1], int(np.ceil(x1)) - left),
        min(crop.shape[0], int(np.ceil(y1)) - top),
    )
    thickness = max(2, round(0.006 * max(crop.shape[:2])))
    cv2.rectangle(crop, box[:2], box[2:], BOX_BGR, thickness)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), plain, box
