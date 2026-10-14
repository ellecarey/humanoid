import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from humanoid.config import VlmConfig
from humanoid.types import RegionLabel, RegionView

_PROMPT = (
    "This crop shows one indoor object that changed between two visits. "
    'Reply with JSON only: {"label": "short noun phrase", '
    '"attributes": {"color": "", "category": ""}}.'
)


def label_with_vlm(
    views: dict[int, list[RegionView]],
    cfg: VlmConfig,
    cache_dir: Path | None = None,
) -> dict[int, RegionLabel]:
    cache = Path(cache_dir) if cache_dir is not None else Path(".vlm_cache")
    labels: dict[int, RegionLabel] = {}
    for region_id, region_views in views.items():
        if not region_views:
            continue
        crop = region_views[0].crop
        cached = _read_cache(cache, crop, cfg) if cfg.cache else None
        label, attributes = cached if cached is not None else _ask(crop, cfg)
        if cfg.cache and cached is None:
            _write_cache(cache, crop, cfg, label, attributes)
        labels[region_id] = RegionLabel(region_id, label, attributes)
    return labels


def _ask(crop: np.ndarray, cfg: VlmConfig) -> tuple[str, dict[str, str]]:
    if cfg.provider != "gemini":
        raise ValueError(f"Unknown VLM provider {cfg.provider}")
    return _gemini(crop, cfg)


    encoded = _jpeg_b64(crop)
    response = OpenAI().chat.completions.create(
        model=cfg.model,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": _PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
    )
    return _parse(response.choices[0].message.content or "")


def _gemini(crop: np.ndarray, cfg: VlmConfig) -> tuple[str, dict[str, str]]:
    from google import genai
    from google.genai import types

    response = genai.Client().models.generate_content(
        model=cfg.model,
        contents=[
            types.Part.from_bytes(data=_jpeg(crop), mime_type="image/jpeg"),
            _PROMPT,
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return _parse(response.text or "")


def _parse(text: str) -> tuple[str, dict[str, str]]:
    payload = json.loads(text)
    raw = payload.get("attributes") or {}
    attributes = {str(key): str(value) for key, value in raw.items()}
    return str(payload.get("label") or "unknown object"), attributes


def _jpeg(crop: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode crop as JPEG")
    return encoded.tobytes()



def _digest(crop: np.ndarray, cfg: VlmConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(cfg.provider.encode())
    hasher.update(b"\0")
    hasher.update(cfg.model.encode())
    hasher.update(b"\0")
    hasher.update(np.ascontiguousarray(crop).tobytes())
    return hasher.hexdigest()


def _read_cache(cache: Path, crop: np.ndarray, cfg: VlmConfig) -> tuple[str, dict[str, str]] | None:
    path = cache / f"{_digest(crop, cfg)}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return str(payload["label"]), {str(k): str(v) for k, v in payload["attributes"].items()}


def _write_cache(cache: Path, crop: np.ndarray, cfg: VlmConfig, label: str, attributes: dict[str, str]) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{_digest(crop, cfg)}.json"
    path.write_text(json.dumps({"label": label, "attributes": attributes}, indent=2) + "\n")