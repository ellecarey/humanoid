"""Offline verification and labelling of change regions with CLIP.

For each region, the crop from the video that contains the object ("own") and the
crop of the same 3D box from the other video ("other") are scored against a
generic indoor vocabulary. A real change has an object in the own crop, and the
other crop gives that object's label much lower probability. Regions whose own
crop is mostly wall, floor or fixed furniture, or whose two crops look the same,
are rejected.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from humanoid.change_detection.matching import flip_direction
from humanoid.config import LabelerConfig
from humanoid.semantics.views import best_view
from humanoid.types import DetectedChange, RegionView
from humanoid.utils.logging import get_logger

log = get_logger(__name__)

# COCO and LVIS indoor classes; nothing here is specific to the example scene.
# fmt: off
OBJECT_CLASSES = [
    "chair", "office chair", "stool", "pillow", "cushion", "umbrella", "backpack",
    "handbag", "suitcase", "bag", "box", "cardboard box", "bin", "laptop", "book",
    "bottle", "cup", "bowl", "plate", "potted plant", "vase", "lamp", "fan",
    "heater", "clock", "picture frame", "teddy bear", "toy", "shoe", "coat",
    "jacket", "towel", "blanket", "basket", "laundry basket", "bucket", "guitar",
    "monitor", "keyboard", "speaker", "printer", "paper shredder", "kettle",
    "broom", "vacuum cleaner", "hat", "scarf", "sports ball", "tennis racket",
    "skateboard", "remote control", "phone", "scissors", "dumbbell",
]
BACKGROUND_CLASSES = [
    "wall", "plain white wall", "floor", "carpet", "wooden floor", "ceiling",
    "door", "window", "curtain", "blind", "sofa", "bed", "desk", "table", "shelf",
    "bookshelf", "wardrobe", "cupboard", "radiator", "mirror", "skirting board",
    "light switch", "door frame", "power socket", "ceiling light",
]
# fmt: on
TEMPLATES = (
    "a photo of a {}.",
    "a photo of a {} in a room.",
    "a close-up photo of a {}.",
)
MIN_SIDE_PX = 32
FLIP_MARGIN = 0.15


class Embedder(Protocol):
    logit_scale: float

    def images(self, crops: list[np.ndarray]) -> np.ndarray: ...

    def texts(self, prompts: list[str]) -> np.ndarray: ...


class ClipEmbedder:
    """L2-normalised CLIP image and text embeddings via transformers."""

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.logit_scale = float(self.model.logit_scale.exp())

    def images(self, crops: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            return self._normalise(self.model.get_image_features(**inputs))

    def texts(self, prompts: list[str]) -> np.ndarray:
        inputs = self.processor(text=prompts, padding=True, return_tensors="pt")
        with self.torch.inference_mode():
            return self._normalise(
                self.model.get_text_features(**inputs.to(self.device))
            )

    def _normalise(self, output) -> np.ndarray:
        # transformers 5 returns a model output whose pooler_output is the projection
        tensor = (
            output if isinstance(output, self.torch.Tensor) else output.pooler_output
        )
        array = tensor.float().cpu().numpy()
        return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-6)


@dataclass
class Verification:
    rejected: set[int] = field(default_factory=set)
    embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)


def verify_changes(
    changes: list[DetectedChange],
    views: dict[int, list[RegionView]],
    cfg: LabelerConfig,
    embedder: Embedder | None = None,
    debug_dir: Path | None = None,
) -> Verification:
    """Label, reject or flip every change in place; return what was rejected."""
    result = Verification()
    if not changes:
        return result
    embedder = embedder or ClipEmbedder(cfg.model)
    objects = cfg.object_classes or OBJECT_CLASSES
    background = cfg.background_classes or BACKGROUND_CLASSES
    classes = _class_embeddings(embedder, objects + background)
    n_objects = len(objects)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for index, change in enumerate(changes):
        record = {
            "index": index,
            "region_id": change.region_ids[0],
            "type": change.change_type,
            "cue": change.attributes.get("cue", "geometry"),
        }
        result.records.append(record)
        own_id, other_id = _scenes(change)
        region_id = change.region_ids[0]
        own = _scored(best_view(views, region_id, own_id), cfg, embedder, classes)
        other = _scored(best_view(views, region_id, other_id), cfg, embedder, classes)
        if own is None and other is None:
            _reject(result, record, index, "no view")
            continue

        own_obj, other_obj = (
            _best_objectness(own, n_objects),
            _best_objectness(other, n_objects),
        )
        if record["cue"] == "appearance" and other_obj > own_obj + FLIP_MARGIN:
            flip_direction(change)
            own, other = other, own
            own_obj, other_obj = other_obj, own_obj
            record["flipped"] = True
        if own is None:
            _reject(result, record, index, "object side not seen")
            continue

        # The padding at which CLIP is surest of an object: small objects need
        # context (a closed umbrella alone looks like a vase), large ones do not.
        scale = max(range(len(own)), key=lambda k: float(own[k][2][:n_objects].max()))
        own_crop, own_emb, own_p = own[scale]
        label_index = int(np.argmax(own_p[:n_objects]))
        label = objects[label_index]
        objectness = _objectness(own_p, n_objects)
        record.update(
            label=label,
            top_class=(objects + background)[int(np.argmax(own_p))],
            box_pad=cfg.box_pads[scale],
            objectness=_round(objectness),
            other_objectness=_round(other_obj),
        )
        if debug_dir is not None:
            _save(
                debug_dir, index, own_crop, None if other is None else other[scale][0]
            )
        change.label = label
        change.confidence = float(change.confidence) * objectness
        result.embeddings[index] = own_emb

        if objectness < cfg.min_objectness:
            _reject(result, record, index, "background")
            continue
        # Geometric regions already have free-space evidence that the other
        # video sees through them; appearance regions need the other crop.
        if other is not None and record["cue"] == "appearance":
            _, other_emb, other_p = other[scale]
            similarity = float(own_emb @ other_emb)
            ratio = float(other_p[label_index] / max(own_p[label_index], 1e-6))
            record.update(same_similarity=_round(similarity), label_ratio=_round(ratio))
            if similarity > cfg.max_same_similarity:
                _reject(result, record, index, "looks the same in both videos")
                continue
            if ratio > cfg.max_label_ratio:
                _reject(result, record, index, "object in both videos")
                continue
        record["kept"] = True

    if debug_dir is not None:
        (debug_dir / "verdicts.json").write_text(
            json.dumps(result.records, indent=2) + "\n"
        )
    kept = len(changes) - len(result.rejected)
    log.info("CLIP kept %d of %d regions", kept, len(changes))
    return result


def _scenes(change: DetectedChange) -> tuple[str, str]:
    return ("v1", "v2") if change.change_type == "removed" else ("v2", "v1")


def _scored(
    view: RegionView | None, cfg: LabelerConfig, embedder: Embedder, classes: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
    """(crop, embedding, class probabilities) at each box padding."""
    if view is None:
        return None
    crops = [_object_crop(view, pad) for pad in cfg.box_pads]
    embeddings = embedder.images(crops)
    return [
        (crop, embedding, _probs(embedding, classes, embedder.logit_scale))
        for crop, embedding in zip(crops, embeddings)
    ]


def _best_objectness(scored, n_objects: int) -> float:
    if scored is None:
        return 0.0
    return max(_objectness(probs, n_objects) for _, _, probs in scored)


def _object_crop(view: RegionView, pad: float) -> np.ndarray:
    """The boxed area of an unboxed crop, padded by `pad` of its size."""
    if view.plain is None or view.box is None:
        return view.crop
    x0, y0, x1, y1 = view.box
    height, width = view.plain.shape[:2]
    pad_x = max(pad * (x1 - x0), (MIN_SIDE_PX - (x1 - x0)) / 2, 0.0)
    pad_y = max(pad * (y1 - y0), (MIN_SIDE_PX - (y1 - y0)) / 2, 0.0)
    left, top = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
    right, bottom = min(width, int(x1 + pad_x) + 1), min(height, int(y1 + pad_y) + 1)
    crop = view.plain[top:bottom, left:right]
    return crop if crop.size else view.crop


def _class_embeddings(embedder: Embedder, classes: list[str]) -> np.ndarray:
    prompts = [template.format(name) for name in classes for template in TEMPLATES]
    texts = (
        embedder.texts(prompts).reshape(len(classes), len(TEMPLATES), -1).mean(axis=1)
    )
    return texts / np.maximum(np.linalg.norm(texts, axis=1, keepdims=True), 1e-6)


def _probs(embedding: np.ndarray, classes: np.ndarray, scale: float) -> np.ndarray:
    logits = scale * (classes @ embedding)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _objectness(probs: np.ndarray, n_objects: int) -> float:
    return float(probs[:n_objects].sum())


def _reject(result: Verification, record: dict, index: int, reason: str) -> None:
    result.rejected.add(index)
    record["rejected"] = reason


def _save(directory: Path, index: int, own, other) -> None:
    for name, crop in (("own", own), ("other", other)):
        if crop is not None:
            cv2.imwrite(
                str(directory / f"region_{index:02d}_{name}.jpg"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
            )


def _round(value: float) -> float:
    return round(float(value), 3)
