from pathlib import Path

import numpy as np
import typer

from humanoid.change_detection.clustering import cluster_changes
from humanoid.change_detection.geometric_diff import compute_geometric_diff
from humanoid.change_detection.visibility import apply_visibility
from humanoid.config import load_config
from humanoid.pipeline import Pipeline
from humanoid.preprocessing.extract_frames import extract_frames
from humanoid.preprocessing.frame_selection import select_scene_frames
from humanoid.reconstruction.features import dense_features
from humanoid.reconstruction.feedforward import reconstruct
from humanoid.semantics.views import select_views
from humanoid.semantics.vlm import interpret_regions
from humanoid.utils.debug import (
    check_ground_truth,
    detection_summary,
    frame_sheets,
    write_json,
)
from humanoid.utils.geometry import up_vector

app = typer.Typer()
STAGES = ("frames", "recon", "features", "detect", "describe", "report")


def _frames(cfg, pipe: Pipeline):
    extracted = {}
    for scene_id, video in cfg.videos.items():
        images = pipe.root / "frames" / scene_id / "images"
        extracted[scene_id] = extract_frames(video, cfg.frames, images, scene_id)
    selected = select_scene_frames(extracted, cfg.frames)
    frame_sheets(selected, pipe.root / "debug")
    return selected


def _recon(frames, cfg, pipe: Pipeline):
    recon, clouds = reconstruct(frames, cfg.recon)
    write_json(pipe.root / "debug" / "alignment.json", recon.alignment)
    return recon, clouds


def _detect(recon, clouds, features, cfg, pipe: Pipeline):
    diff = compute_geometric_diff(clouds["v1"], clouds["v2"], cfg.diff)
    labeled = apply_visibility(diff, recon, recon.depths, cfg.visibility, features)
    regions = cluster_changes(labeled, cfg.cluster, up=up_vector(recon.poses))
    detection_summary(labeled, regions, pipe.root / "debug")
    changed = labeled.label == "changed"
    static = labeled.label == "unchanged"
    points = {
        "static": labeled.points[static],
        "removed": labeled.points[changed & (labeled.source_scene == "v1")],
        "added": labeled.points[changed & (labeled.source_scene == "v2")],
        "unknown": labeled.points[labeled.label == "unknown"],
    }
    if labeled.colors is not None:
        points["static_colors"] = labeled.colors[static]
    return regions, {key: np.asarray(value) for key, value in points.items()}


def _describe(regions, frames, recon, cfg, pipe: Pipeline):
    views = select_views(regions, frames, recon, recon.depths)
    changes = interpret_regions(
        regions, views, cfg.vlm, debug_dir=pipe.root / "debug" / "gemini"
    )
    return changes, views


def _require_cached(pipe: Pipeline, names) -> None:
    for name in names:
        path = pipe.root / name / "done.json"
        if not path.is_file():
            raise typer.BadParameter(
                f"No cached {name} result at {path}. Run from an earlier stage first."
            )


@app.callback()
def main() -> None:
    """Detect changes between two phone videos."""


@app.command()
def run(
    config: Path = Path("configs/default.yaml"),
    from_stage: str | None = None,
    force: bool = False,
) -> None:
    """Run the pipeline, reusing cached stages.

    --from-stage detect recomputes detect and everything after it from the cached
    reconstruction; --force recomputes every stage.
    """
    if from_stage is not None and from_stage not in STAGES:
        raise typer.BadParameter(f"from-stage must be one of: {', '.join(STAGES)}")

    cfg = load_config(config)
    pipe = Pipeline(cfg)
    start = STAGES.index(from_stage) if from_stage else len(STAGES)
    if from_stage and not force:
        _require_cached(pipe, STAGES[:start])

    def stage(name, fn):
        rerun = force or STAGES.index(name) >= start
        return Pipeline(cfg, force=rerun).stage(name, fn)

    frames = stage("frames", lambda: _frames(cfg, pipe))
    recon, clouds = stage("recon", lambda: _recon(frames, cfg, pipe))
    features = stage("features", lambda: dense_features(recon.images, cfg.features))
    regions, points = stage(
        "detect", lambda: _detect(recon, clouds, features, cfg, pipe)
    )
    changes, views = stage(
        "describe", lambda: _describe(regions, frames, recon, cfg, pipe)
    )

    Pipeline(cfg, force=True).stage(
        "report", lambda: pipe.write_report(changes, points, views)
    )
    _print_ground_truth(changes, cfg, pipe)


def _print_ground_truth(changes, cfg, pipe: Pipeline) -> None:
    rows = check_ground_truth(changes, cfg.ground_truth)
    typer.echo(f"Detected {len(changes)} change(s):")
    for change in changes:
        typer.echo(f"  {change.id}. {change.change_type}: {change.label}")
    if not rows:
        return
    write_json(pipe.root / "debug" / "ground_truth.json", rows)
    typer.echo("Ground truth:")
    for row in rows:
        typer.echo(
            f"  {row['result']:<9} expected {row['expected']!r}"
            f" detected {row['detected']!r}"
        )


if __name__ == "__main__":
    app()
