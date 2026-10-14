import numpy as np


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return (R @ points.T).T + np.asarray(t, dtype=float)


def project(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    camera = transform_points(points, R, t)
    z = camera[:, 2]
    pixels = (K @ camera.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]

    width = 2 * K[0, 2]
    height = 2 * K[1, 2]
    u, v = pixels[:, 0], pixels[:, 1]
    keep = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    keep &= np.isfinite(pixels).all(axis=1)
    return pixels[keep], z[keep], keep


def backproject(u, v, depth, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    depth = np.asarray(depth, dtype=float)
    pixels = np.stack([u, v, np.ones_like(u)], axis=-1)
    camera = (pixels @ np.linalg.inv(K).T) * depth[..., None]
    return (camera - np.asarray(t, dtype=float)) @ R
