import numpy as np

from humanoid.config import EmbedConfig
from humanoid.types import RegionView


def embed_crops(
    views: dict[int, list[RegionView]],
    cfg: EmbedConfig,
) -> dict[int, np.ndarray]:
    pending = {region_id: region_views for region_id, region_views in views.items() if region_views}
    if not pending:
        return {}

    model, processor = _load(cfg.model)
    embeddings: dict[int, np.ndarray] = {}
    for region_id, region_views in pending.items():
        vectors = [_embed_image(model, processor, view.crop, cfg.model) for view in region_views]
        mean = np.mean(np.stack(vectors), axis=0)
        norm = np.linalg.norm(mean)
        embeddings[region_id] = mean if norm == 0 else mean / norm
    return embeddings


def _load(model_id: str):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "clip" in model_id.lower():
        from transformers import CLIPModel, CLIPProcessor

        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        return model, processor

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def _embed_image(model, processor, image: np.ndarray, model_id: str) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.no_grad():
        if "clip" in model_id.lower():
            vector = _as_vector(model.get_image_features(**inputs))
        else:
            vector = model(**inputs).last_hidden_state[:, 0]
    array = vector.squeeze().float().cpu().numpy()
    norm = np.linalg.norm(array)
    return array if norm == 0 else array / norm


def _as_vector(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    for name in ("image_embeds", "pooler_output"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise TypeError(f"Unexpected embedding output {type(output)}")