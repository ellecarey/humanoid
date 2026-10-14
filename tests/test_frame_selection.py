import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.preprocessing.frame_selection import select_frames
from humanoid.types import Frame


def _cfg(max_frames: int) -> FramesConfig:
    return FramesConfig(
        target_fps=4.0,
        max_frames=max_frames,
        max_width=1280,
        blur_threshold=1.0,
        duplicate_threshold=0.98,
    )


def _frame(tmp_path, index: int, strong: bool) -> Frame:
    rng = np.random.default_rng(index)
    high = 256 if strong else 30
    image = rng.integers(0, high, (64, 64), dtype=np.uint8)
    path = tmp_path / f"frame_{index:06d}.jpg"
    cv2.imwrite(str(path), image)
    return Frame("v1", path, index, 64, 64)


def test_select_frames_caps_at_35_sharp_frames(tmp_path):
    frames = [_frame(tmp_path, index, strong=index in {0, 99}) for index in range(100)]

    selected = select_frames(frames, _cfg(35))

    indexes = [frame.index for frame in selected]
    assert len(selected) == 35
    assert indexes[0] == 0
    assert indexes[-1] == 99
    assert indexes == sorted(indexes)


def test_select_frames_keeps_a_short_clip(tmp_path):
    frames = [_frame(tmp_path, index, strong=True) for index in range(4)]

    selected = select_frames(frames, _cfg(35))

    assert [frame.index for frame in selected] == [0, 1, 2, 3]
