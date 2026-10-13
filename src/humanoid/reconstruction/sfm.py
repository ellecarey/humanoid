import shutil
from pathlib import Path

import numpy as np

from humanoid.config import SfmConfig
from humanoid.types import CameraPose, Frame, SceneRecon
from humanoid.utils.logging import get_logger

logger = get_logger(__name__)

SEQUENTIAL_OVERLAP = 10
RETRIEVAL_MATCHES = 10


def run_joint_sfm(
    frames: dict[str, list[Frame]],
    cfg: SfmConfig,
    output_dir: Path,
) -> SceneRecon:
    from hloc import (
        extract_features,
        match_features,
        pairs_from_retrieval,
        pairs_from_sequential,
        reconstruction,
    )
    import pycolmap

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    names = _copy_images(frames, images_dir)

    feature_conf = extract_features.confs[cfg.feature]
    retrieval_conf = extract_features.confs[cfg.retrieval]
    matcher_conf = match_features.confs[cfg.matcher]

    features = extract_features.main(feature_conf, images_dir, output_dir)
    global_features = extract_features.main(retrieval_conf, images_dir, output_dir)

    pair_paths = []
    for scene_id, scene_names in names.items():
        image_list = output_dir / f"{scene_id}.txt"
        image_list.write_text("\n".join(scene_names) + "\n")
        sequential = output_dir / f"pairs-{scene_id}.txt"
        pairs_from_sequential.main(
            sequential,
            image_list=image_list,
            overlap=SEQUENTIAL_OVERLAP,
            quadratic_overlap=True,
        )
        pair_paths.append(sequential)

    scene_ids = list(names)
    for query_id, db_id in (scene_ids, scene_ids[::-1]):
        cross = output_dir / f"pairs-{query_id}-{db_id}.txt"
        pairs_from_retrieval.main(
            global_features,
            cross,
            num_matched=RETRIEVAL_MATCHES,
            query_list=output_dir / f"{query_id}.txt",
            db_list=output_dir / f"{db_id}.txt",
        )
        pair_paths.append(cross)

    pairs = output_dir / "pairs-all.txt"
    _write_unique_pairs(pair_paths, pairs)
    matches = match_features.main(matcher_conf, pairs, features, output_dir)

    model = reconstruction.main(
        output_dir / "sfm",
        images_dir,
        pairs,
        features,
        matches,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )
    ratios = _registration_ratios(model, names)
    if all(ratio >= cfg.min_registration_ratio for ratio in ratios.values()):
        return _scene_from_reconstruction(model)

    logger.info(
        "Joint SfM ratios %s are below %.2f; running per-video registration fallback",
        {scene: round(ratio, 3) for scene, ratio in ratios.items()},
        cfg.min_registration_ratio,
    )
    return _fallback(names, images_dir, pairs, features, matches, output_dir, reconstruction)


def _copy_images(frames: dict[str, list[Frame]], images_dir: Path) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {}
    for scene_id, scene_frames in frames.items():
        folder = images_dir / scene_id
        folder.mkdir(parents=True, exist_ok=True)
        scene_names = []
        for frame in scene_frames:
            destination = folder / frame.path.name
            if not destination.exists():
                shutil.copy2(frame.path, destination)
            scene_names.append(f"{scene_id}/{frame.path.name}")
        names[scene_id] = scene_names
    return names


def _write_unique_pairs(paths: list[Path], destination: Path) -> None:
    seen: set[tuple[str, str]] = set()
    lines = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            left, right = line.split()
            key = tuple(sorted((left, right)))
            if key not in seen:
                seen.add(key)
                lines.append(f"{left} {right}")
    destination.write_text("\n".join(lines) + "\n")


def _registration_ratios(model, names: dict[str, list[str]]) -> dict[str, float]:
    registered = {image.name for image in model.images.values()}
    return {
        scene_id: sum(name in registered for name in scene_names) / len(scene_names)
        for scene_id, scene_names in names.items()
    }


def _scene_from_reconstruction(model) -> SceneRecon:
    poses = {}
    for image in model.images.values():
        rigid = image.cam_from_world
        scene_id = image.name.split("/", 1)[0]
        poses[image.name] = CameraPose(
            scene_id=scene_id,
            image_name=image.name,
            R=np.asarray(rigid.rotation.matrix(), dtype=float),
            t=np.asarray(rigid.translation, dtype=float).reshape(3),
            K=np.asarray(model.cameras[image.camera_id].calibration_matrix(), dtype=float),
        )

    order = {name: index for index, name in enumerate(sorted(poses))}
    point_ids = sorted(model.points3D)
    points = np.array([model.points3D[pid].xyz for pid in point_ids], dtype=float)
    point_index = {pid: index for index, pid in enumerate(point_ids)}
    observations: dict[str, list[list[float]]] = {name: [] for name in poses}
    point_image_ids = np.empty(len(point_ids), dtype=int)

    for pid in point_ids:
        point = model.points3D[pid]
        first_name = model.images[point.track.elements[0].image_id].name
        point_image_ids[point_index[pid]] = order[first_name]
        for element in point.track.elements:
            image = model.images[element.image_id]
            xy = image.points2D[element.point2D_idx].xy
            observations[image.name].append([point_index[pid], float(xy[0]), float(xy[1])])

    return SceneRecon(
        poses=poses,
        points_xyz=points,
        point_image_ids=point_image_ids,
        observations={
            name: np.asarray(rows, dtype=float).reshape(-1, 3)
            for name, rows in observations.items()
        },
    )


def _fallback(names, images_dir, pairs, features, matches, output_dir, reconstruction) -> SceneRecon:
    import pycolmap
    from humanoid.reconstruction.registration import register_clouds

    scenes = {}
    for scene_id, scene_names in names.items():
        model = reconstruction.main(
            output_dir / f"sfm-{scene_id}",
            images_dir,
            pairs,
            features,
            matches,
            camera_mode=pycolmap.CameraMode.SINGLE,
            image_list=scene_names,
        )
        scenes[scene_id] = _scene_from_reconstruction(model)

    target_id, source_id = list(scenes)
    target = scenes[target_id]
    source = scenes[source_id]
    transform = register_clouds(source.points_xyz, target.points_xyz)
    logger.info("Registration fallback aligned %s onto %s", source_id, target_id)
    return _merge(target, _apply_transform(source, transform))


def _apply_transform(scene: SceneRecon, transform: np.ndarray) -> SceneRecon:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    poses = {}
    for name, pose in scene.poses.items():
        new_rotation = pose.R @ rotation.T
        poses[name] = CameraPose(
            scene_id=pose.scene_id,
            image_name=pose.image_name,
            R=new_rotation,
            t=pose.t - new_rotation @ translation,
            K=pose.K,
        )
    points = (rotation @ scene.points_xyz.T).T + translation
    return SceneRecon(poses, points, scene.point_image_ids, scene.observations)


def _merge(target: SceneRecon, source: SceneRecon) -> SceneRecon:
    offset = len(target.points_xyz)
    target_order = {name: index for index, name in enumerate(sorted(target.poses))}
    source_order = {name: index for index, name in enumerate(sorted(source.poses))}
    merged_order = {name: index for index, name in enumerate(sorted({**target.poses, **source.poses}))}

    def retarget(scene: SceneRecon, old_order: dict[str, int]) -> np.ndarray:
        inverse = {index: name for name, index in old_order.items()}
        return np.array([merged_order[inverse[index]] for index in scene.point_image_ids], dtype=int)

    observations = dict(target.observations)
    for name, rows in source.observations.items():
        shifted = rows.copy()
        shifted[:, 0] += offset
        observations[name] = shifted

    return SceneRecon(
        poses={**target.poses, **source.poses},
        points_xyz=np.vstack([target.points_xyz, source.points_xyz]),
        point_image_ids=np.concatenate([
            retarget(target, target_order),
            retarget(source, source_order),
        ]),
        observations=observations,
    )