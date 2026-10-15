import numpy as np
import open3d as o3d

from humanoid.config import ReconConfig
from humanoid.types import CameraPose, Frame, SceneRecon
from humanoid.utils.geometry import backproject


def reconstruct(
    frames: dict[str, list[Frame]], cfg: ReconConfig
) -> tuple[SceneRecon, dict[str, o3d.geometry.PointCloud]]:
    ordered = [
        (scene_id, frame)
        for scene_id, scene_frames in frames.items()
        for frame in scene_frames
    ]
    if cfg.model.startswith("facebook/VGGT"):
        poses, depths, confidence = _vggt(ordered, cfg)
    else:
        poses, depths, confidence = _mapanything(ordered, cfg)
    recon = SceneRecon(poses=poses, depths=depths, confidence=confidence)
    clouds = {scene_id: _cloud(recon, scene_id, cfg) for scene_id in frames}
    return recon, clouds


def _cloud(recon: SceneRecon, scene_id: str, cfg: ReconConfig) -> np.ndarray:
    chunks = []
    for image_name, pose in recon.poses.items():
        if pose.scene_id != scene_id:
            continue
        depth = recon.depths[scene_id][image_name]
        conf = recon.confidence[scene_id][image_name]
        keep = (conf >= cfg.min_confidence) & np.isfinite(depth) & (depth > 0)
        rows, cols = np.nonzero(keep)
        if len(rows) == 0:
            continue
        chunks.append(
            backproject(cols, rows, depth[rows, cols], pose.K, pose.R, pose.t)
        )
    if not chunks:
        return np.zeros((0, 3))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(chunks))
    cloud = cloud.voxel_down_sample(cfg.voxel_size)
    return np.asarray(cloud.points)


def _mapanything(ordered: list[tuple[str, Frame]], cfg: ReconConfig):
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(cfg.model).to(device).eval()
    predictions = model.infer(load_images([str(frame.path) for _, frame in ordered]))
    return _from_mapanything(ordered, predictions)


def _vggt(ordered: list[tuple[str, Frame]], cfg: ReconConfig):
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT.from_pretrained(cfg.model).to(device).eval()
    images = load_and_preprocess_images([str(frame.path) for _, frame in ordered]).to(
        device
    )
    with torch.no_grad():
        predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    return _from_vggt(ordered, extrinsic, intrinsic, predictions, cfg.scale_m_per_unit)


def _from_mapanything(ordered, predictions):
    poses: dict[str, CameraPose] = {}
    depths: dict[str, dict[str, np.ndarray]] = {}
    confidence: dict[str, dict[str, np.ndarray]] = {}
    for (scene_id, frame), pred in zip(ordered, predictions):
        name = f"{scene_id}/{frame.path.name}"
        cam_to_world = np.squeeze(_numpy(pred["camera_poses"]))
        world_to_cam = np.linalg.inv(cam_to_world)
        _store(
            poses,
            depths,
            confidence,
            scene_id,
            name,
            world_to_cam[:3, :3],
            world_to_cam[:3, 3],
            np.squeeze(_numpy(pred["intrinsics"])),
            np.squeeze(_numpy(pred["depth_z"])).astype(np.float32),
            np.squeeze(_numpy(pred["conf"])).astype(np.float32),
        )
    return poses, depths, confidence


def _from_vggt(ordered, extrinsic, intrinsic, predictions, scale_m_per_unit: float):
    poses: dict[str, CameraPose] = {}
    depths: dict[str, dict[str, np.ndarray]] = {}
    confidence: dict[str, dict[str, np.ndarray]] = {}
    extrinsics = np.squeeze(_numpy(extrinsic), axis=0)
    intrinsics = np.squeeze(_numpy(intrinsic), axis=0)
    depth_maps = (
        np.squeeze(_numpy(predictions["depth"])).astype(np.float32) * scale_m_per_unit
    )
    conf_maps = np.squeeze(_numpy(predictions["depth_conf"])).astype(np.float32)
    for index, (scene_id, frame) in enumerate(ordered):
        name = f"{scene_id}/{frame.path.name}"
        _store(
            poses,
            depths,
            confidence,
            scene_id,
            name,
            extrinsics[index, :3, :3],
            extrinsics[index, :3, 3],
            intrinsics[index],
            np.squeeze(depth_maps[index]),
            np.squeeze(conf_maps[index]),
        )
    return poses, depths, confidence


def _store(
    poses,
    depths,
    confidence,
    scene_id,
    name,
    rotation,
    translation,
    intrinsics,
    depth,
    conf,
):
    poses[name] = CameraPose(
        scene_id=scene_id,
        image_name=name,
        R=np.asarray(rotation, dtype=float),
        t=np.asarray(translation, dtype=float),
        K=np.asarray(intrinsics, dtype=float),
    )
    depths.setdefault(scene_id, {})[name] = depth
    confidence.setdefault(scene_id, {})[name] = conf


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)
