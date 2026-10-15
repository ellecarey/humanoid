from pathlib import Path

import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.types import Frame

_Kept = tuple[Frame, float, np.ndarray]


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
    burst = _best_burst(_prepare(frames, cfg), cfg.max_frames, cfg.min_overlap)
    return [frame for frame, _score, _thumb in burst]


def select_scene_frames(
    scenes: dict[str, list[Frame]], cfg: FramesConfig
) -> dict[str, list[Frame]]:
    prepared = {
        scene_id: _prepare(frames, cfg) for scene_id, frames in scenes.items()
    }
    if not prepared:
        return {}
    anchor_id = next(iter(prepared))
    anchor = _best_burst(prepared[anchor_id], cfg.max_frames, cfg.min_overlap)
    anchor_thumbs = [thumb for _frame, _score, thumb in anchor]
    selected = {anchor_id: [frame for frame, _score, _thumb in anchor]}
    for scene_id, kept in prepared.items():
        if scene_id == anchor_id:
            continue
        matched = _matching_burst(kept, anchor_thumbs, cfg.max_frames, cfg.min_overlap)
        selected[scene_id] = [frame for frame, _score, _thumb in matched]
    return selected


def _prepare(frames: list[Frame], cfg: FramesConfig) -> list[_Kept]:
    kept: list[_Kept] = []
    previous: np.ndarray | None = None
    for frame in frames:
        image = _read_bgr(frame.path)
        score = blur_score(image)
        if score < cfg.blur_threshold:
            continue
        thumb = _thumbnail(image)
        if previous is not None and _cosine(thumb, previous) >= cfg.duplicate_threshold:
            continue
        kept.append((frame, score, thumb))
        previous = thumb
    return kept


def _best_burst(kept: list[_Kept], max_frames: int, min_overlap: float) -> list[_Kept]:
    if len(kept) <= max_frames:
        return kept
    ranked = _windows(kept, max_frames)
    eligible = [item for item in ranked if item[1] >= min_overlap]
    start, _overlap, _sharpness = max(
        eligible or ranked, key=lambda item: (item[2], item[1])
    )
    return kept[start : start + max_frames]


def _matching_burst(
    kept: list[_Kept],
    anchor_thumbs: list[np.ndarray],
    max_frames: int,
    min_overlap: float,
) -> list[_Kept]:
    if not anchor_thumbs or len(kept) <= max_frames:
        return _best_burst(kept, max_frames, min_overlap)
    best_start = 0
    best_score = -1.0
    found = False
    for start, overlap, _sharpness in _windows(kept, max_frames):
        if overlap < min_overlap:
            continue
        window = kept[start : start + max_frames]
        score = float(
            np.mean(
                [
                    max(_cosine(thumb, anchor) for anchor in anchor_thumbs)
                    for _frame, _blur, thumb in window
                ]
            )
        )
        if score > best_score:
            best_score = score
            best_start = start
            found = True
    if not found:
        return _best_burst(kept, max_frames, min_overlap)
    return kept[best_start : best_start + max_frames]


def _windows(kept: list[_Kept], max_frames: int) -> list[tuple[int, float, float]]:
    windows = []
    for start in range(len(kept) - max_frames + 1):
        window = kept[start : start + max_frames]
        windows.append((start, _min_overlap(window), _sharpness(window)))
    return windows


def _min_overlap(window: list[_Kept]) -> float:
    if len(window) < 2:
        return 1.0
    return float(
        min(
            _cosine(window[index][2], window[index + 1][2])
            for index in range(len(window) - 1)
        )
    )


def _sharpness(window: list[_Kept]) -> float:
    return float(np.mean([score for _frame, score, _thumb in window]))
