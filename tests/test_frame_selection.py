import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.preprocessing.frame_selection import select_frames, select_scene_frames
from humanoid.types import Frame


def _cfg(max_frames: int) -> FramesConfig:
    return FramesConfig(
        target_fps=4.0,
        max_frames=max_frames,
        max_width=1280,
        blur_threshold=0.0,
        duplicate_threshold=1.01,
        min_overlap=0.0,
    )


def _frame(tmp_path, index: int, family: int) -> Frame:
    rng = np.random.default_rng(index + family * 1000)
    image = np.full((64, 64), 40, dtype=np.uint8)
    if family == 1:
        image[::2, :] = 255
        image[:, ::2] = 255
    elif family == 0:
        image[:, :32] = 180
    else:
        image[20:44, 20:44] = 160
    noise = rng.integers(0, 6, image.shape, dtype=np.uint8)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    path = tmp_path / f"frame_{index:06d}.jpg"
    cv2.imwrite(str(path), image)
    return Frame("v1", path, index, 64, 64)


def test_select_frames_keeps_the_sharp_overlapping_burst(tmp_path):
    frames = (
        [_frame(tmp_path, index, family=0) for index in range(8)]
        + [_frame(tmp_path, index, family=1) for index in range(8, 16)]
        + [_frame(tmp_path, index, family=2) for index in range(16, 24)]
    )

    selected = select_frames(frames, _cfg(4))

    indexes = [frame.index for frame in selected]
    assert indexes == sorted(indexes)
    assert max(indexes) - min(indexes) == 3
    assert min(indexes) >= 8
    assert max(indexes) <= 15


def test_select_frames_drops_near_duplicates(tmp_path):
    frames = [_frame(tmp_path, index, family=1) for index in range(4)]
    cfg = _cfg(4)
    cfg = cfg.model_copy(update={"duplicate_threshold": 0.98})

    selected = select_frames(frames, cfg)

    assert [frame.index for frame in selected] == [0]


def test_select_frames_keeps_a_short_clip(tmp_path):
    frames = [_frame(tmp_path, index, family=1) for index in range(4)]

    selected = select_frames(frames, _cfg(35))

    assert [frame.index for frame in selected] == [0, 1, 2, 3]


def test_select_scene_frames_matches_the_other_video(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    v1 = [_frame(tmp_path / "v1", index, family=1) for index in range(6)]
    v2 = (
        [_frame(tmp_path / "v2", index, family=0) for index in range(6)]
        + [_frame(tmp_path / "v2", index, family=1) for index in range(6, 12)]
    )
    for frame in v2:
        frame.scene_id = "v2"

    selected = select_scene_frames({"v1": v1, "v2": v2}, _cfg(4))

    assert [frame.index for frame in selected["v1"]] == [0, 1, 2, 3]
    assert min(frame.index for frame in selected["v2"]) >= 6
