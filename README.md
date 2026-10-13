# Spatial AI & 3D Change Detection

A 3D spatial change detection pipeline that compares two video recordings of an indoor scene to identify moved, added, or removed objects. Built for the **Humanoid Spatial AI & Navigation Internship Challenge**.

---

## 🏗️ Project Overview

Robots operating in human environments must understand how their surroundings evolve over time. This project processes two hand-held phone recordings taken from different viewpoints, aligns them in a shared 3D coordinate frame, and detects structural and semantic changes across the scene while accounting for viewpoint shifts, occlusions, and variable lighting.

---

## 📁 Repository Structure

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

## 🎥 Joint SfM

`run_joint_sfm` in `src/humanoid/reconstruction/sfm.py` reconstructs both videos in one model: SuperPoint and LightGlue matches, one camera per video, and a point-cloud registration fallback when either video registers below 90%. Camera poses are stored world-to-camera.

That stage imports `hloc` and `pycolmap`, so install the GPU extra before running it:

```bash
uv sync --extra gpu
```

The alignment tests do not need the GPU extra:

```bash
uv run pytest tests/test_sfm_alignment.py
```