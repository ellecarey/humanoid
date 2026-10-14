from pathlib import Path

import cv2

from humanoid.config import FramesConfig
from humanoid.types import Frame


def extract_frames(
    video: Path,
    cfg: FramesConfig,
    output_dir: Path,
    scene_id: str,
) -> list[Frame]:
    video = Path(video)
    output_dir = Path(output_dir)
    if video.suffix.lower() not in {".mp4", ".mov"}:
        raise ValueError(f"Unsupported video type: {video.suffix}")

    cap = cv2.VideoCapture(str(video))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Could not read fps from {video}")
    frame_skip = max(1, round(fps / cfg.target_fps))

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Frame] = []
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if index % frame_skip == 0:
                height, width = image.shape[:2]
                if width > cfg.max_width:
                    new_width = cfg.max_width
                    new_height = round(height * new_width / width)
                    image = cv2.resize(
                        image,
                        (new_width, new_height),
                        interpolation=cv2.INTER_AREA,
                    )
                    height, width = image.shape[:2]
                path = output_dir / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), image)
                frames.append(
                    Frame(
                        scene_id=scene_id,
                        path=path,
                        index=index,
                        width=width,
                        height=height,
                    )
                )
            index += 1
    finally:
        cap.release()
    return frames
