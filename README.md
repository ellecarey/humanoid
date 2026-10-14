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
│   ├── reconstruction/             # registration fallback
│   └── utils/
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```
