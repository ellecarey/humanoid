from pathlib import Path

import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.types import Frame

_Scored = tuple[Frame, float, np.ndarray]


def _read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read frame {path}")
    return image


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumbnail(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    vector = small.astype(np.float64).ravel()
    vector -= vector.mean()
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if np.linalg.norm(left) == 0.0 or np.linalg.norm(right) == 0.0:
        return 0.0
    return float(np.dot(left, right))


def select_frames(frames: list[Frame], cfg: FramesConfig) -> list[Frame]:
    """Sharpest frame per window, without leaving a gap longer than max_gap_s.

    Plain walls score low on sharpness and look alike, but dropping them breaks
    the chain of overlapping views the reconstruction needs.
    """
    best = _sharpest_per_window(_score(frames), cfg.window_s)
    best = _thin_without_gaps(best, cfg)
    return [frame for frame, _score, _thumb in _spread(best, cfg.max_per_video)]


def select_scene_frames(
    scenes: dict[str, list[Frame]], cfg: FramesConfig
) -> dict[str, list[Frame]]:
    return {scene_id: select_frames(frames, cfg) for scene_id, frames in scenes.items()}


def _score(frames: list[Frame]) -> list[_Scored]:
    scored: list[_Scored] = []
    for frame in frames:
        image = _read_bgr(frame.path)
        scored.append((frame, blur_score(image), _thumbnail(image)))
    return scored


def _sharpest_per_window(scored: list[_Scored], window_s: float) -> list[_Scored]:
    windows: dict[int, _Scored] = {}
    for item in scored:
        key = int(item[0].time_s // window_s) if window_s > 0 else item[0].index
        if key not in windows or item[1] > windows[key][1]:
            windows[key] = item
    return [windows[key] for key in sorted(windows)]


def _thin_without_gaps(scored: list[_Scored], cfg: FramesConfig) -> list[_Scored]:
    if not scored:
        return scored
    floor = cfg.min_relative_sharpness * float(
        np.median([score for _f, score, _t in scored])
    )
    kept: list[_Scored] = []
    for item in scored:
        if kept:
            gap = item[0].time_s - kept[-1][0].time_s
            blurry = item[1] < floor
            duplicate = _cosine(item[2], kept[-1][2]) >= cfg.duplicate_threshold
            if (blurry or duplicate) and gap < cfg.max_gap_s:
                continue
        kept.append(item)
    return kept


def _spread(scored: list[_Scored], max_frames: int) -> list[_Scored]:
    if len(scored) <= max_frames:
        return scored
    picks = np.unique(np.round(np.linspace(0, len(scored) - 1, max_frames)).astype(int))
    return [scored[index] for index in picks]
