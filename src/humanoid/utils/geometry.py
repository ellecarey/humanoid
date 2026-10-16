import numpy as np


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (R @ points.T).T + np.asarray(t, dtype=float)


def project(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    shape: tuple[int, ...] | None = None,
):
    camera = transform_points(points, R, t)
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = (K @ camera.T).T
        pixels = pixels[:, :2] / pixels[:, 2:3]

    if shape is None:
        width, height = 2 * K[0, 2], 2 * K[1, 2]
    else:
        height, width = shape[0], shape[1]
    u, v = pixels[:, 0], pixels[:, 1]
    keep = np.isfinite(pixels).all(axis=1)
    keep &= (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return pixels[keep], z[keep], keep


def backproject(u, v, depth, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    depth = np.asarray(depth, dtype=float)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    camera = (pixels @ np.linalg.inv(K).T) * depth[..., None]
    return (camera - np.asarray(t, dtype=float)) @ R


def sample_nearest(grid: np.ndarray, pixels: np.ndarray, shape=None) -> np.ndarray:
    """Values of `grid` at `pixels` given in the coordinates of an image of `shape`.

    `grid` may be coarser than the image (a feature map). Pixel centres sit on
    integer coordinates, matching `backproject`.
    """
    height, width = (shape or grid.shape)[:2]
    rows = np.floor((pixels[:, 1] + 0.5) * grid.shape[0] / height).astype(int)
    cols = np.floor((pixels[:, 0] + 0.5) * grid.shape[1] / width).astype(int)
    rows = np.clip(rows, 0, grid.shape[0] - 1)
    cols = np.clip(cols, 0, grid.shape[1] - 1)
    return grid[rows, cols]


def depth_margin(z: np.ndarray, absolute: float, relative: float) -> np.ndarray:
    return np.maximum(absolute, relative * np.asarray(z, dtype=float))


def to_frame_pixels(
    pixels: np.ndarray, frame_map: tuple[float, float, float, float] | None
) -> np.ndarray:
    if frame_map is None:
        return np.asarray(pixels, dtype=float)
    scale_x, scale_y, offset_x, offset_y = frame_map
    pixels = np.asarray(pixels, dtype=float)
    return np.stack(
        [(pixels[:, 0] + offset_x) / scale_x, (pixels[:, 1] + offset_y) / scale_y],
        axis=1,
    )


def up_vector(poses) -> np.ndarray | None:
    """World up from OpenCV cameras (+Y down), averaged over every pose."""
    downs = [np.asarray(pose.R, dtype=float)[1] for pose in poses.values()]
    if not downs:
        return None
    up = -np.mean(downs, axis=0)
    norm = np.linalg.norm(up)
    return up / norm if norm > 0 else None
