import pickle
from pathlib import Path

import numpy as np
import typer

from humanoid.change_detection.clustering import cluster_changes
from humanoid.change_detection.geometric_diff import compute_geometric_diff
from humanoid.change_detection.visibility import apply_visibility
from humanoid.config import load_config
from humanoid.pipeline import Pipeline
from humanoid.preprocessing.extract_frames import extract_frames
from humanoid.preprocessing.frame_selection import select_frames
from humanoid.reconstruction.feedforward import reconstruct
from humanoid.semantics.views import select_views
from humanoid.semantics.vlm import interpret_regions

app = typer.Typer()
STAGES = ("frames", "recon", "detect", "describe", "report")


def _frames(cfg, pipe: Pipeline):
    selected = {}
    for scene_id, video in cfg.videos.items():
        images = pipe.root / "frames" / scene_id / "images"
        extracted = extract_frames(video, cfg.frames, images, scene_id)
        selected[scene_id] = select_frames(extracted, cfg.frames)
    return selected


def _detect(recon, clouds, cfg):
    diff = compute_geometric_diff(clouds["v1"], clouds["v2"], cfg.diff)
    labeled = apply_visibility(diff, recon, recon.depths, cfg.visibility)
    regions = cluster_changes(labeled, cfg.cluster)
    changed = labeled.label == "changed"
    points = {
        "static": labeled.points[labeled.label == "unchanged"],
        "removed": labeled.points[changed & (labeled.source_scene == "v1")],
        "added": labeled.points[changed & (labeled.source_scene == "v2")],
        "unknown": labeled.points[labeled.label == "unknown"],
    }
    return regions, {key: np.asarray(value) for key, value in points.items()}


def _describe(regions, frames, recon, cfg):
    views = select_views(regions, frames, recon, recon.depths)
    changes = interpret_regions(regions, views, cfg.vlm)
    return changes, views


def _cached(pipe: Pipeline, name: str):
    path = pipe.root / name / "result.pkl"
    if not path.is_file():
        raise typer.BadParameter(
            f"No cached {name} result at {path}. Run from an earlier stage first."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


@app.callback()
def main() -> None:
    """Detect changes between two phone videos."""


@app.command()
def run(
    config: Path = Path("configs/default.yaml"),
    from_stage: str = "frames",
    force: bool = False,
) -> None:
    """Run the pipeline. --from-stage detect reuses the cached reconstruction."""
    if from_stage not in STAGES:
        raise typer.BadParameter(f"from-stage must be one of: {', '.join(STAGES)}")

    cfg = load_config(config)
    pipe = Pipeline(cfg, force=force)
    start = STAGES.index(from_stage)

    if start <= STAGES.index("frames"):
        frames = pipe.stage("frames", lambda: _frames(cfg, pipe))
    else:
        frames = _cached(pipe, "frames")

    if start <= STAGES.index("recon"):
        recon, clouds = pipe.stage("recon", lambda: reconstruct(frames, cfg.recon))
    else:
        recon, clouds = _cached(pipe, "recon")

    if start <= STAGES.index("detect"):
        regions, points = pipe.stage("detect", lambda: _detect(recon, clouds, cfg))
    else:
        regions, points = _cached(pipe, "detect")

    if start <= STAGES.index("describe"):
        changes, views = pipe.stage(
            "describe", lambda: _describe(regions, frames, recon, cfg)
        )
    else:
        changes, views = _cached(pipe, "describe")

    pipe.stage("report", lambda: pipe.write_report(changes, points, views))


if __name__ == "__main__":
    app()
