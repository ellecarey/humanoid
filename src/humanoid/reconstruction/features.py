import cv2
import numpy as np

from humanoid.config import FeaturesConfig
from humanoid.utils.logging import get_logger

log = get_logger(__name__)

PATCH = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PCA_SAMPLES = 200_000

Features = dict[str, dict[str, np.ndarray]]  # scene_id -> image name -> (gh, gw, D)


def dense_features(
    images: dict[str, dict[str, np.ndarray]], cfg: FeaturesConfig
) -> Features:
    """DINOv2 patch features per frame, PCA-reduced and L2-normalised, as float16."""
    if not cfg.enabled or not images:
        return {}
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", cfg.model).to(device).eval()
    raw: dict[tuple[str, str], np.ndarray] = {}
    for scene_id, by_name in images.items():
        for name, image in by_name.items():
            raw[(scene_id, name)] = _patch_tokens(model, image, cfg.scale, device)
    log.info("Extracted DINOv2 features for %d frames", len(raw))

    mean, basis = _pca(list(raw.values()), cfg.pca_dim)
    features: Features = {}
    for (scene_id, name), tokens in raw.items():
        features.setdefault(scene_id, {})[name] = _reduce(tokens, mean, basis)
    return features


def _patch_tokens(model, image: np.ndarray, scale: float, device: str) -> np.ndarray:
    import torch

    height, width = image.shape[:2]
    target_h = max(PATCH, round(height * scale / PATCH) * PATCH)
    target_w = max(PATCH, round(width * scale / PATCH) * PATCH)
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    normalised = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalised.transpose(2, 0, 1))[None].to(device)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
        ),
    ):
        tokens = model.forward_features(tensor)["x_norm_patchtokens"]
    grid = tokens.float().reshape(target_h // PATCH, target_w // PATCH, -1)
    return grid.cpu().numpy().astype(np.float16)


def _pca(grids: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([grid.reshape(-1, grid.shape[-1]) for grid in grids])
    rng = np.random.default_rng(0)
    if len(flat) > PCA_SAMPLES:
        flat = flat[rng.choice(len(flat), PCA_SAMPLES, replace=False)]
    flat = flat.astype(np.float32)
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt[: min(dim, vt.shape[0])].T


def _reduce(grid: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    reduced = (grid.astype(np.float32) - mean) @ basis
    norm = np.linalg.norm(reduced, axis=-1, keepdims=True)
    return (reduced / np.maximum(norm, 1e-6)).astype(np.float16)
