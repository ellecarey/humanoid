import cv2
import numpy as np

from humanoid.config import FramesConfig
from humanoid.preprocessing.frame_selection import select_frames, select_scene_frames
from humanoid.types import Frame


def _cfg(max_per_video: int = 100, **update) -> FramesConfig:
    cfg = FramesConfig(
        target_fps=4.0,
        window_s=0.5,
        max_per_video=max_per_video,
        max_width=1280,
        duplicate_threshold=1.01,
        min_relative_sharpness=0.0,
    )
    return cfg.model_copy(update=update)


def _frame(tmp_path, index: int, sharp: bool = True, family: int = 0) -> Frame:
    rng = np.random.default_rng(index + 1000 * family)
    image = np.full((64, 64), 40, dtype=np.uint8)
    image[:, : 8 + 4 * (index % 12)] = 200
    image[family * 5 : family * 5 + 10, :] = 120
    image = np.clip(
        image.astype(np.int16) + rng.integers(0, 6, image.shape), 0, 255
    ).astype(np.uint8)
    if not sharp:
        image = cv2.GaussianBlur(image, (15, 15), 6)
    path = tmp_path / f"frame_{index:06d}_{family}.jpg"
    cv2.imwrite(str(path), image)
    return Frame("v1", path, index, 64, 64, time_s=index * 0.25)


def test_select_frames_keeps_the_sharpest_frame_per_window(tmp_path):
    frames = [_frame(tmp_path, index, sharp=index % 2 == 1) for index in range(8)]

    selected = select_frames(frames, _cfg())

    assert [frame.index for frame in selected] == [1, 3, 5, 7]


def test_select_frames_drops_near_duplicates(tmp_path):
    frames = [_frame(tmp_path, 0)] * 4
    for index, frame in enumerate(frames):
        frames[index] = Frame("v1", frame.path, index * 2, 64, 64, time_s=index * 0.5)

    selected = select_frames(frames, _cfg(duplicate_threshold=0.98, max_gap_s=10.0))

    assert [frame.index for frame in selected] == [0]


def test_select_frames_never_leaves_a_gap_longer_than_max_gap(tmp_path):
    frames = [_frame(tmp_path, 0)] * 6
    for index, frame in enumerate(frames):
        frames[index] = Frame("v1", frame.path, index * 2, 64, 64, time_s=index * 0.5)

    selected = select_frames(frames, _cfg(duplicate_threshold=0.98, max_gap_s=1.0))

    times = [frame.time_s for frame in selected]
    assert times == [0.0, 1.0, 2.0]


def test_select_frames_spreads_the_cap_over_the_whole_video(tmp_path):
    frames = [_frame(tmp_path, index) for index in range(0, 80, 2)]

    selected = select_frames(frames, _cfg(max_per_video=5))

    times = [frame.time_s for frame in selected]
    assert len(selected) == 5
    assert times[0] == 0.0
    assert times[-1] == frames[-1].time_s


def test_select_frames_drops_blurry_windows(tmp_path):
    frames = [_frame(tmp_path, index * 2, sharp=index != 2) for index in range(5)]

    selected = select_frames(frames, _cfg(min_relative_sharpness=0.5))

    assert 4 not in [frame.index for frame in selected]


def test_select_scene_frames_covers_each_video_independently(tmp_path):
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    v1 = [_frame(tmp_path / "v1", index, family=0) for index in range(0, 24, 2)]
    v2 = [_frame(tmp_path / "v2", index, family=1) for index in range(0, 24, 2)]
    for frame in v2:
        frame.scene_id = "v2"

    selected = select_scene_frames({"v1": v1, "v2": v2}, _cfg(max_per_video=4))

    for scene_id, frames in (("v1", v1), ("v2", v2)):
        times = [frame.time_s for frame in selected[scene_id]]
        assert times[0] == frames[0].time_s
        assert times[-1] == frames[-1].time_s
