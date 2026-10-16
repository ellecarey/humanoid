import numpy as np

from humanoid.config import LabelerConfig, VlmConfig
from humanoid.semantics.vlm import interpret_regions
from humanoid.types import ChangeRegion, RegionView

CLASSES = ("chair", "umbrella", "wall")


class FakeEmbedder:
    """One-hot embeddings: a crop filled with value k shows CLASSES[k]."""

    logit_scale = 100.0

    def images(self, crops):
        return np.stack([self._onehot(int(crop[0, 0, 0])) for crop in crops])

    def texts(self, prompts):
        return np.stack(
            [
                self._onehot(next(i for i, name in enumerate(CLASSES) if name in p))
                for p in prompts
            ]
        )

    @staticmethod
    def _onehot(index: int) -> np.ndarray:
        vector = np.zeros(len(CLASSES))
        vector[index] = 1.0
        return vector


def _labeler() -> LabelerConfig:
    return LabelerConfig(
        object_classes=["chair", "umbrella"], background_classes=["wall"]
    )


def _region(region_id, change_type, centroid, cue="geometry") -> ChangeRegion:
    point = np.asarray(centroid, dtype=float)
    return ChangeRegion(
        id=region_id,
        change_type=change_type,
        scene_id="v1" if change_type == "removed" else "v2",
        points=point.reshape(1, 3),
        centroid=point,
        extent=np.array([0.5, 0.6, 1.0]),
        obb_rotation=np.eye(3),
        cue=cue,
    )


def _views(region_id: int, before: str, after: str) -> list[RegionView]:
    def view(scene_id, name):
        crop = np.full((8, 8, 3), CLASSES.index(name), dtype=np.uint8)
        return RegionView(region_id, f"{scene_id}/a.jpg", crop, 1.0, scene_id)

    return [view("v1", before), view("v2", after)]


def _run(regions, views):
    cfg = VlmConfig(provider="gemini", model="unused", cache=False)
    return interpret_regions(
        regions, views, cfg, labeler=_labeler(), embedder=FakeEmbedder()
    )


def test_labels_keep_real_changes_and_drop_background(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    regions = [
        _region(0, "removed", [0.0, 0.0, 0.0]),
        _region(1, "removed", [3.0, 0.0, 0.0]),
        _region(2, "added", [6.0, 0.0, 0.0], cue="appearance"),
    ]
    views = {
        0: _views(0, "chair", "wall"),
        1: _views(1, "wall", "wall"),
        2: _views(2, "umbrella", "umbrella"),
    }

    changes = _run(regions, views)

    assert [(c.change_type, c.label) for c in changes] == [("removed", "chair")]


def test_flat_appearance_change_flips_to_the_side_with_the_object(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    regions = [_region(0, "removed", [1.0, 0.0, 0.0], cue="appearance")]
    views = {0: _views(0, "wall", "umbrella")}

    changes = _run(regions, views)

    assert [(c.change_type, c.label) for c in changes] == [("added", "umbrella")]
    assert changes[0].to_centroid.tolist() == [1.0, 0.0, 0.0]


def test_matching_labels_and_looks_pair_into_a_move(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    regions = [
        _region(0, "removed", [0.0, 0.0, 0.0]),
        _region(1, "added", [1.0, 0.0, 0.0]),
    ]
    views = {0: _views(0, "chair", "wall"), 1: _views(1, "wall", "chair")}

    changes = _run(regions, views)

    assert [(c.change_type, c.label) for c in changes] == [("moved", "chair")]
    assert changes[0].displacement_m == 1.0


def test_gemini_failure_falls_back_to_the_labeller(monkeypatch):
    from humanoid.semantics import vlm

    monkeypatch.setenv("GEMINI_API_KEY", "test")

    def unavailable(*args):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(vlm, "_ask_gemini", unavailable)
    regions = [_region(0, "removed", [0.0, 0.0, 0.0])]

    changes = _run(regions, {0: _views(0, "chair", "wall")})

    assert [(c.change_type, c.label) for c in changes] == [("removed", "chair")]
