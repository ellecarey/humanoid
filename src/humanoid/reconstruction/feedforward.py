import numpy as np
import open3d as o3d

from humanoid.config import ReconConfig
from humanoid.reconstruction.refine import (
    align_clouds,
    overlap_fraction,
    self_consistent,
    transform_scene,
)
from humanoid.types import CameraPose, ColoredCloud, Frame, SceneRecon
from humanoid.utils.geometry import backproject
from humanoid.utils.logging import get_logger

log = get_logger(__name__)


def reconstruct(
    frames: dict[str, list[Frame]], cfg: ReconConfig
) -> tuple[SceneRecon, dict[str, ColoredCloud]]:
    ordered = [
        (scene_id, frame)
        for scene_id, scene_frames in frames.items()
        for frame in scene_frames
    ]
    log.info("Reconstructing %d frames in one pass", len(ordered))
    if cfg.model.startswith("facebook/VGGT"):
        recon = _vggt(ordered, cfg)
    else:
        recon = _mapanything(ordered, cfg)
    clouds = {scene_id: _cloud(recon, scene_id, cfg) for scene_id in frames}
    _refine(recon, clouds, cfg)
    return recon, clouds


def _refine(recon: SceneRecon, clouds: dict[str, ColoredCloud], cfg: ReconConfig):
    scene_ids = list(clouds)
    if len(scene_ids) < 2:
        return
    reference, moving = scene_ids[0], scene_ids[1]
    recon.alignment["overlap_before"] = overlap_fraction(
        clouds[moving].points, clouds[reference].points
    )
    if cfg.icp:
        transform, stats = align_clouds(
            clouds[moving].points, clouds[reference].points, cfg.icp_voxels_m
        )
        transform_scene(recon, moving, transform)
        clouds[moving] = ColoredCloud(
            _apply(transform, clouds[moving].points), clouds[moving].colors
        )
        recon.alignment.update(stats)
    for scene_id, cloud in clouds.items():
        keep = self_consistent(
            cloud.points,
            recon,
            scene_id,
            cfg.self_margin_m,
            cfg.self_margin_rel,
            cfg.min_self_views,
        )
        recon.alignment[f"self_consistent_{scene_id}"] = (
            float(keep.mean()) if len(keep) else 0.0
        )
        clouds[scene_id] = ColoredCloud(cloud.points[keep], cloud.colors[keep])
    recon.alignment["overlap_after"] = overlap_fraction(
        clouds[moving].points, clouds[reference].points
    )
    log.info("Alignment: %s", recon.alignment)


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _cloud(recon: SceneRecon, scene_id: str, cfg: ReconConfig) -> ColoredCloud:
    chunks, colours = [], []
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
        image = recon.images.get(scene_id, {}).get(image_name)
        if image is None:
            colours.append(np.full((len(rows), 3), 0.6))
        else:
            colours.append(image[rows, cols].astype(float) / 255.0)
    if not chunks:
        return ColoredCloud(np.zeros((0, 3)), np.zeros((0, 3)))
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(chunks))
    cloud.colors = o3d.utility.Vector3dVector(np.vstack(colours))
    cloud = cloud.voxel_down_sample(cfg.voxel_size)
    return ColoredCloud(np.asarray(cloud.points), np.asarray(cloud.colors))


def _mapanything(ordered: list[tuple[str, Frame]], cfg: ReconConfig) -> SceneRecon:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(cfg.model).to(device).eval()
    views = load_images([str(frame.path) for _, frame in ordered])
    predictions = model.infer(
        views,
        memory_efficient_inference=True,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=cfg.conf_percentile > 0,
        confidence_percentile=cfg.conf_percentile,
    )
    recon = SceneRecon(poses={}, depths={}, confidence={})
    for (scene_id, frame), pred in zip(ordered, predictions):
        name = f"{scene_id}/{frame.path.name}"
        world_to_cam = np.linalg.inv(np.squeeze(_numpy(pred["camera_poses"])))
        depth = np.squeeze(_numpy(pred["depth_z"])).astype(np.float32)
        if "mask" in pred:
            mask = np.squeeze(_numpy(pred["mask"])).astype(bool)
            depth = np.where(mask, depth, 0.0).astype(np.float32)
        image = _uint8(np.squeeze(_numpy(pred["img_no_norm"])))
        _store(
            recon,
            scene_id,
            name,
            world_to_cam[:3, :3],
            world_to_cam[:3, 3],
            np.squeeze(_numpy(pred["intrinsics"])),
            depth,
            _half(np.squeeze(_numpy(pred["conf"]))),
            image,
            _cover_crop_map(frame.width, frame.height, image.shape[1], image.shape[0]),
        )
    return recon


def _vggt(ordered: list[tuple[str, Frame]], cfg: ReconConfig) -> SceneRecon:
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
    extrinsics = np.squeeze(_numpy(extrinsic), axis=0)
    intrinsics = np.squeeze(_numpy(intrinsic), axis=0)
    depth_maps = (
        np.squeeze(_numpy(predictions["depth"])).astype(np.float32)
        * cfg.scale_m_per_unit
    )
    conf_maps = _half(np.squeeze(_numpy(predictions["depth_conf"])))
    rgb = _numpy(images).transpose(0, 2, 3, 1)
    recon = SceneRecon(poses={}, depths={}, confidence={})
    for index, (scene_id, frame) in enumerate(ordered):
        image = _uint8(rgb[index])
        height, width = image.shape[:2]
        resized_h = round(frame.height * width / frame.width / 14) * 14
        _store(
            recon,
            scene_id,
            f"{scene_id}/{frame.path.name}",
            extrinsics[index, :3, :3],
            extrinsics[index, :3, 3],
            intrinsics[index],
            np.squeeze(depth_maps[index]),
            np.squeeze(conf_maps[index]),
            image,
            (
                width / frame.width,
                resized_h / frame.height,
                0.0,
                float(max(0, (resized_h - height) // 2)),
            ),
        )
    return recon


def _cover_crop_map(
    frame_w: int, frame_h: int, model_w: int, model_h: int
) -> tuple[float, float, float, float]:
    """Mirror MapAnything's resize-to-cover followed by a centred crop."""
    scale = max(model_w / frame_w, model_h / frame_h) + 1e-8
    resized_w = int(np.floor(frame_w * scale))
    resized_h = int(np.floor(frame_h * scale))
    return (
        scale,
        scale,
        float((resized_w - model_w) // 2),
        float((resized_h - model_h) // 2),
    )


def _store(
    recon: SceneRecon,
    scene_id,
    name,
    rotation,
    translation,
    intrinsics,
    depth,
    conf,
    image,
    frame_map,
):
    recon.poses[name] = CameraPose(
        scene_id=scene_id,
        image_name=name,
        R=np.asarray(rotation, dtype=float),
        t=np.asarray(translation, dtype=float),
        K=np.asarray(intrinsics, dtype=float),
    )
    recon.depths.setdefault(scene_id, {})[name] = depth
    recon.confidence.setdefault(scene_id, {})[name] = conf
    recon.images.setdefault(scene_id, {})[name] = image
    recon.frame_maps[name] = frame_map


def _uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)
    if image.max() <= 1.0 + 1e-3:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _half(values: np.ndarray) -> np.ndarray:
    return np.minimum(np.nan_to_num(values, nan=0.0), 6.0e4).astype(np.float16)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)
