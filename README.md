# Spatial AI & 3D Change Detection

A 3D spatial change detection pipeline that compares two video recordings of an indoor scene to identify moved, added, or removed objects. Built for the **Humanoid Spatial AI & Navigation Internship Challenge**.

---

## Project Overview

Robots operating in human environments must understand how their surroundings evolve over time. This project processes two hand-held phone recordings taken from different viewpoints, aligns them in a shared 3D coordinate frame, and detects structural and semantic changes across the scene while accounting for viewpoint shifts, occlusions, and variable lighting.

---

## Repository Structure

```text
humanoid/
├── configs/
│   └── default.yaml
├── data/
│   ├── ground_truth/changes.json
│   └── raw_videos/                 # gitignored: scene_v1.MOV, scene_v2.MOV
├── src/humanoid/
│   ├── config.py
│   ├── pipeline.py
│   ├── types.py
│   ├── preprocessing/
│   ├── reconstruction/             # joint SfM and the registration fallback
│   └── utils/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## Joint SfM

`run_joint_sfm` in `src/humanoid/reconstruction/sfm.py` reconstructs both videos in one model: SuperPoint and LightGlue matches, one camera per video, and a point-cloud registration fallback when either video registers below 90%. Camera poses are stored world-to-camera.

That stage imports `hloc` and `pycolmap`, so install the GPU extra before running it:

```bash
uv sync --extra gpu
```

The alignment tests do not need the GPU extra:

```bash
uv run pytest tests/test_sfm_alignment.py
```

## Depth and TSDF fusion
`estimate_aligned_depth` in `src/humanoid/reconstruction/depth.py` runs Depth Anything V2 Metric Indoor on each registered frame and fits `z_sfm ≈ scale * z_pred + shift` with a Huber loss against the sparse SfM depths. Frames with fewer than 30 correspondences, or a non-positive scale, are skipped. Aligned maps are saved as float32 `.npy` files.
`fuse_tsdf` builds one coloured cloud per video with an Open3D voxel block grid (`voxel_size` 1.5 cm, truncation 4 cm). v1 and v2 stay in the shared SfM frame, so the unchanged room overlaps. Open3D 0.20's `ScalableTSDFVolume` places those metric depths in the wrong blocks, so fusion uses `VoxelBlockGrid` instead.
```bash
uv run pytest tests/test_depth_fusion.py tests/test_sfm_alignment.py tests/test_geometry.py
'''

## Moved objects and descriptions
`match_moved` pairs removed and added clusters with Hungarian matching. The cost mixes label agreement, crop-embedding similarity, and bounding-box size. Pairs above `match.max_cost` stay removed or added.
`describe_changes` writes one sentence per change. With `describe.use_llm: false`, or when the model call fails, the sentence comes from a template. Directions use the shared SfM axes (+X, +Y, +Z).
```bash
uv run pytest tests/test_matching.py tests/test_describe.py