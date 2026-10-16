# Spatial AI & 3D Change Detection

A 3D spatial change detection pipeline that compares two video recordings of an indoor scene to identify moved, added, or removed objects. Built for the **Humanoid Spatial AI & Navigation Internship Challenge**.

The example scene has three changes: a flag pillow removed from the sofa, the desk chair moved into the room, and an umbrella hung on the door.

---

## Quickstart

- Colab (GPU runtime): open [notebooks/run_on_colab.ipynb](notebooks/run_on_colab.ipynb) and run the cells in order. A 40 GB or larger GPU runs the full walk at the default frame budget.
- Local CUDA machine: `pip install -e ".[gpu,vlm]"` then `humanoid run --config configs/default.yaml`.
- Rerun a later stage from the cache: `humanoid run --from-stage detect` retunes thresholds on CPU, and `humanoid run --from-stage describe` repeats only the Gemini call.

Set `GEMINI_API_KEY` to let Gemini verify and label each change. Without it, changes are labelled "object" and paired by geometry only. The model is `gemini-3.8-flash`; `gemini-2.5-flash` is no longer available to new API keys. A 503 from Gemini means the model is busy: run the describe stage again, the reconstruction is not repeated.

---

## Pipeline

| Stage | What it does |
| --- | --- |
| `frames` | Sharpest frame per 0.5 s window over the whole walk, about 90 per video. A frame is kept when dropping it would leave a gap of about a second, so plain walls still connect the views |
| `recon` | One MapAnything pass over both videos (memory-efficient, bf16), robust ICP of video 2 onto video 1, and a same-video consistency filter that drops floating points |
| `features` | DINOv2 patch features per frame, reduced to 64 dimensions |
| `detect` | Every point is voted against every view of the other video: *free* (the other video sees through it), *agree* (same surface), or *occluded*. Free-space majorities are geometric changes; agreeing surfaces whose features differ are appearance changes. DBSCAN clusters them, and the floor is the lowest horizontal plane |
| `describe` | Before and after crops of each region go to one Gemini Flash call, which rejects false positives, labels the objects, and proposes moved pairs. Pairs are checked against 3D size, and the displacement is computed from the geometry |
| `report` | `report.html` (coloured 3D scene plus before/after cards), `changes.json`, `scene.ply` |

Each run also writes `outputs/<run>/debug/`:

- `frames_v1.jpg` and `frames_v2.jpg`: contact sheets of the selected frames, to check that both videos cover the room.
- `alignment.json`: the share of video 2 points within 5 cm of video 1, before and after ICP. Above about 0.7 means the two reconstructions landed in one frame.
- `detect.json`: point counts per label and the clustered regions.
- `scores.html`: histograms of the free-space vote and the appearance cosine, used to set thresholds.
- `gemini/`: the crops sent to Gemini and the JSON it returned.
- `ground_truth.json`: each expected change against what was detected. "type only" means the geometry was right and the label was still the offline fallback.

Appearance matters for flat changes. The umbrella sits only a few centimetres in front of the door, which is within the depth noise, but its texture is different from the door. Lighting differences trigger the same check, so Gemini's rejection step is what keeps those out of the report.

---

## Repository structure

```text
humanoid/
├── configs/default.yaml
├── data/
│   ├── ground_truth/changes.json   # flag pillow removed, desk chair moved, umbrella added
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