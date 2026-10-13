# Spatial AI & 3D Change Detection

A 3D spatial change detection pipeline that compares two video recordings of an indoor scene to identify moved, added, or removed objects. Built for the **Humanoid Spatial AI & Navigation Internship Challenge**.

---

## 🏗️ Project Overview

Robots operating in human environments must understand how their surroundings evolve over time. This project processes two hand-held phone recordings taken from different viewpoints, aligns them in a shared 3D coordinate frame, and detects structural and semantic changes across the scene while accounting for viewpoint shifts, occlusions, and variable lighting.

---

## 📁 Repository Structure

```text
humanoid/
├── data/
│   └── raw_videos/
│       ├── scene_before.mp4   # Initial scene capture
│       └── scene_after.mp4    # Modified scene capture
├── src/
│   └── humanoid/
│       └── __init__.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md