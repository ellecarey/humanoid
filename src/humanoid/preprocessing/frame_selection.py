from pathlib import Path

import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.types import Frame


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read frame {path}")
    return image


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumbnail(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float64).ravel()
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if np.linalg.norm(left) == 0.0 or np.linalg.norm(right) == 0.0:
        return 0.0
    return float(np.dot(left, right))


def select_frames(frames: list[Frame], cfg: FramesConfig) -> list[Frame]:
    kept: list[tuple[Frame, float]] = []
    previous: np.ndarray | None = None
    for frame in frames:
        image = _read_bgr(frame.path)
        score = blur_score(image)
        if score < cfg.blur_threshold:
            continue
        thumb = _thumbnail(image)
        if previous is not None and _cosine(thumb, previous) >= cfg.duplicate_threshold:
            continue
        kept.append((frame, score))
        previous = thumb
    return _cap(kept, cfg.max_frames)


def _cap(kept: list[tuple[Frame, float]], max_frames: int) -> list[Frame]:
    if len(kept) <= max_frames:
        return [frame for frame, _score in kept]
    bin_ids = np.linspace(0, max_frames, len(kept), endpoint=False).astype(int)
    best: dict[int, tuple[float, Frame]] = {}
    for (frame, score), bin_id in zip(kept, bin_ids):
        current = best.get(int(bin_id))
        if current is None or score > current[0]:
            best[int(bin_id)] = (score, frame)
    return [best[bin_id][1] for bin_id in sorted(best)]
