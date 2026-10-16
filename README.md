# Spatial AI & 3D Change Detection

A 3D spatial change detection pipeline that compares two video recordings of an indoor scene to identify moved, added, or removed objects. Built for the **Humanoid Spatial AI & Navigation Internship Challenge**.

---

## Quickstart

- Colab (GPU runtime): open [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) and run the cells in order.
- Local CUDA machine: `pip install -e ".[gpu,vlm]"` then `humanoid run --config configs/default.yaml`.
- Retune on CPU from the cached reconstruction: `humanoid run --from-stage detect`.

Set `GEMINI_API_KEY` to let Gemini verify and label each change; without it, changes are labelled "object" and paired by geometry only.

---

## Pipeline

| Stage | What it does |
| --- | --- |
| `frames` | Sharpest frame per 0.5 s window over the whole walk, about 90 per video |
| `recon` | One MapAnything pass over both videos (memory-efficient, bf16), robust ICP of video 2 onto video 1, and a same-video consistency filter that drops floating points |
| `features` | DINOv2 patch features per frame, PCA to 64 dimensions |
| `detect` | Every point is voted against every view of the other video: *free* (the other video sees through it), *agree* (same surface), or *occluded*. Free-space majorities are geometric changes; agreeing surfaces whose features differ are appearance changes. DBSCAN clusters, with a gravity-aware floor filter |
| `describe` | Before/after crops of each region from both videos go to one Gemini Flash call, which rejects false positives, labels the objects, and proposes moved pairs. Pairs are re-checked against 3D extents, and displacement comes from geometry |
| `report` | `report.html` (coloured 3D scene plus before/after cards), `changes.json`, `scene.ply` |

Each run also writes `outputs/<run>/debug/`:

- `frames_v1.jpg` and `frames_v2.jpg`: contact sheets of the selected frames, to check coverage.
- `alignment.json`: v1/v2 overlap before and after ICP.
- `detect.json`: label counts and the regions found.
- `scores.html`: vote and appearance histograms.
- `gemini/`: the crops and JSON exchanged with Gemini.
- `ground_truth.json`: the result checked against `data/ground_truth/changes.json`.

Appearance matters for flat changes. An umbrella hanging on a door sits only a few centimetres in front of it, which is within feed-forward depth noise, but its texture is very different.

---

## Repository Structure

```text
humanoid/
├── configs/default.yaml
├── data/
│   ├── ground_truth/changes.json
│   └── raw_videos/                 # gitignored: scene_v1.MOV, scene_v2.MOV
├── notebooks/run_on_colab.ipynb
├── src/humanoid/
│   ├── cli.py                      # humanoid run
│   ├── preprocessing/              # frame extraction and selection
│   ├── reconstruction/             # MapAnything, ICP refinement, DINOv2 features
│   ├── change_detection/           # diff, visibility vote, clustering
│   ├── semantics/                  # best views, Gemini verification
│   ├── visualisation/report.py
│   └── utils/                      # geometry, debug outputs
└── tests/
```
