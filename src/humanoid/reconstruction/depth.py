from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from humanoid.config import DepthConfig
from humanoid.types import Frame, SceneRecon
from humanoid.utils.geometry import transform_points
from humanoid.utils.logging import get_logger

logger = get_logger(__name__)


def estimate_aligned_depth(
    frames: dict[str, list[Frame]],
    recon: SceneRecon,
    cfg: DepthConfig,
    output_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    output_dir = Path(output_dir)
    model, processor = _load_depth_model(cfg.model)
    depths: dict[str, dict[str, np.ndarray]] = {}

    for scene_id, scene_frames in frames.items():
        depths[scene_id] = {}
        for frame in scene_frames:
            image_name = f"{scene_id}/{frame.path.name}"
            pose = recon.poses.get(image_name)
            observations = recon.observations.get(image_name)
            if pose is None or observations is None or len(observations) == 0:
                continue

            image = cv2.cvtColor(cv2.imread(str(frame.path)), cv2.COLOR_BGR2RGB)
            prediction = _predict_depth(model, processor, image, frame.height, frame.width)
            aligned, reason = _align(prediction, observations, recon.points_xyz, pose, cfg.min_sfm_points)
            if aligned is None:
                logger.info("Skipping %s: %s", image_name, reason)
                continue

            path = output_dir / scene_id / f"{frame.path.stem}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, aligned)
            depths[scene_id][image_name] = aligned

    return depths


def _load_depth_model(model_id: str):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
    return model, processor


def _predict_depth(model, processor, image: np.ndarray, height: int, width: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        predicted = model(**inputs).predicted_depth
    predicted = F.interpolate(
        predicted.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    return predicted.squeeze().cpu().numpy().astype(np.float32)


def _align(
    prediction, observations, points_xyz, pose, min_points: int
) -> tuple[np.ndarray | None, str | None]:
    indices = observations[:, 0].astype(int)
    camera_z = transform_points(points_xyz[indices], pose.R, pose.t)[:, 2]
    u = np.rint(observations[:, 1]).astype(int)
    v = np.rint(observations[:, 2]).astype(int)
    height, width = prediction.shape
    keep = (
        (camera_z > 0)
        & (u >= 0) & (u < width)
        & (v >= 0) & (v < height)
    )
    z_pred = prediction[v[keep], u[keep]]
    z_sfm = camera_z[keep]
    finite = np.isfinite(z_pred) & np.isfinite(z_sfm) & (z_pred > 0)
    z_pred, z_sfm = z_pred[finite], z_sfm[finite]
    if len(z_sfm) < min_points:
        return None, f"fewer than {min_points} SfM depths"

    fit = least_squares(
        lambda params: params[0] * z_pred + params[1] - z_sfm,
        x0=np.array([1.0, 0.0]),
        loss="huber",
        f_scale=0.05,
    )
    scale, shift = fit.x
    if scale <= 0:
        return None, f"non-positive depth scale {scale:.3f}"
    return (scale * prediction + shift).astype(np.float32), None