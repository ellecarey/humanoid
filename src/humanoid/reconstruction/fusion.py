import cv2
import numpy as np
import open3d as o3d

from humanoid.config import FusionConfig
from humanoid.types import Frame, SceneRecon


DEPTH_MAX_M = 10.0


def fuse_tsdf(
    frames: dict[str, list[Frame]],
    recon: SceneRecon,
    depths: dict[str, dict[str, np.ndarray]],
    cfg: FusionConfig,
) -> dict[str, o3d.geometry.PointCloud]:
    device = o3d.core.Device("CPU:0")
    clouds = {}
    for scene_id, scene_frames in frames.items():
        volume = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=cfg.voxel_size,
            block_resolution=16,
            block_count=10_000,
            device=device,
        )
        by_name = {frame.path.name: frame for frame in scene_frames}
        trunc_voxels = cfg.sdf_trunc / cfg.voxel_size
        for image_name, depth in depths.get(scene_id, {}).items():
            pose = recon.poses.get(image_name)
            frame = by_name.get(image_name.split("/", 1)[1])
            if pose is None or frame is None:
                continue
            color = cv2.cvtColor(cv2.imread(str(frame.path)), cv2.COLOR_BGR2RGB)
            depth_image, color_image = _images(depth, color, device)
            intrinsic = o3d.core.Tensor(pose.K, o3d.core.float64, device)
            extrinsic = o3d.core.Tensor(_extrinsic(pose.R, pose.t), o3d.core.float64, device)
            blocks = volume.compute_unique_block_coordinates(
                depth_image, intrinsic, extrinsic, 1000.0, DEPTH_MAX_M, trunc_voxels,
            )
            volume.integrate(
                blocks, depth_image, color_image, intrinsic, intrinsic, extrinsic,
                1000.0, DEPTH_MAX_M, trunc_voxels,
            )
        clouds[scene_id] = volume.extract_point_cloud(weight_threshold=0.0).to_legacy()
    return clouds


def _images(depth: np.ndarray, color: np.ndarray, device: o3d.core.Device):
    millimetres = np.rint(np.clip(depth, 0.0, DEPTH_MAX_M) * 1000.0).astype(np.uint16)
    depth_image = o3d.t.geometry.Image(np.ascontiguousarray(millimetres[..., None])).to(device)
    color_image = o3d.t.geometry.Image(np.ascontiguousarray(color)).to(device)
    return depth_image, color_image


def _extrinsic(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t
    return extrinsic
