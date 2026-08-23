from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .common import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file, slugify


def _assert_pillow_runtime() -> None:
    try:
        from PIL import Image, ImageText  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Pillow is inconsistent in this runtime. In Colab use the numbered notebook environment cell; "
            "if it reports that Pillow was repaired or PIL was already loaded, restart the session once, "
            "then rerun Cells 1→4 before indexing."
        ) from error


def _records(root: Path) -> list[dict[str, Any]]:
    clean = root / "corpus_clean.jsonl"
    full = root / "corpus.jsonl"
    path = clean if clean.exists() else full
    if not path.exists():
        raise FileNotFoundError(f"No corpus found under {root}")
    return read_jsonl(path)


def _text(record: dict[str, Any]) -> str:
    input_obj = record.get("input") or {}
    return (str(input_obj.get("title") or "") + "\n" + str(input_obj.get("text") or "")).strip()


def _batch_key(model: str, content_keys: list[str]) -> str:
    # Cache identity must include content, not only record IDs. Otherwise a
    # corrected corpus record with the same ID could silently reuse a stale
    # embedding from an earlier paper run.
    payload = model + "\n" + "\n".join(content_keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_text_index(
    root: Path,
    *,
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> dict[str, Any]:
    _assert_pillow_runtime()
    import faiss
    from sentence_transformers import SentenceTransformer

    root = Path(root).expanduser().resolve()
    records = [record for record in _records(root) if str((record.get("input") or {}).get("text") or "").strip()]
    index_dir = root / "indexes"
    cache_dir = index_dir / "_embedding_cache" / "text" / slugify(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)
    vectors: list[np.ndarray] = []

    bar = tqdm(total=len(records), desc="Text embeddings", unit="record", dynamic_ncols=True)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        content_keys = [
            f"{r['id']}:{hashlib.sha256(_text(r).encode('utf-8')).hexdigest()}"
            for r in batch
        ]
        key = _batch_key(model_name, content_keys)
        cache_path = cache_dir / f"{key}.npy"
        if cache_path.exists():
            embedding = np.load(cache_path)
        else:
            embedding = model.encode(
                [_text(r) for r in batch],
                batch_size=len(batch),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")
            temporary = cache_path.with_suffix(".npy.part")
            with temporary.open("wb") as f:
                np.save(f, embedding)
            temporary.replace(cache_path)
        vectors.append(embedding.astype("float32"))
        bar.update(len(batch))
    bar.close()

    if not vectors:
        raise ValueError("No text records to index")
    matrix = np.concatenate(vectors, axis=0).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "text.faiss"))
    atomic_write_jsonl(index_dir / "text_metadata.jsonl", records)
    manifest = {
        "type": "text",
        "model": model_name,
        "count": len(records),
        "dimension": int(matrix.shape[1]),
        "normalized": True,
        "faiss_metric": "inner_product",
    }
    atomic_write_json(index_dir / "text_manifest.json", manifest)
    return manifest


def build_image_index(
    root: Path,
    *,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 16,
) -> dict[str, Any]:
    _assert_pillow_runtime()
    import faiss
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    root = Path(root).expanduser().resolve()
    records = [
        record
        for record in _records(root)
        if (record.get("input") or {}).get("modality") in {"map_image", "geo_image"} and (record.get("input") or {}).get("images")
    ]
    index_dir = root / "indexes"
    cache_dir = index_dir / "_embedding_cache" / "image" / slugify(model_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    vectors: list[np.ndarray] = []
    kept: list[dict[str, Any]] = []

    bar = tqdm(total=len(records), desc="Image embeddings", unit="image", dynamic_ncols=True)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        valid: list[dict[str, Any]] = []
        images: list[Any] = []
        for record in batch:
            relative = str((record.get("input") or {}).get("images")[0])
            path = root / relative
            try:
                images.append(Image.open(path).convert("RGB"))
                valid.append(record)
            except Exception as error:
                print(f"Image skipped {relative}: {error!r}")
        if not valid:
            bar.update(len(batch))
            continue
        content_keys = []
        for record in valid:
            relative = str((record.get("input") or {}).get("images")[0])
            content_keys.append(f"{record['id']}:{sha256_file(root / relative)}")
        key = _batch_key(model_name, content_keys)
        cache_path = cache_dir / f"{key}.npy"
        if cache_path.exists():
            embedding = np.load(cache_path)
        else:
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {key_: value.to(device) for key_, value in inputs.items()}
            with torch.no_grad():
                features = model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
            embedding = features.cpu().numpy().astype("float32")
            temporary = cache_path.with_suffix(".npy.part")
            with temporary.open("wb") as f:
                np.save(f, embedding)
            temporary.replace(cache_path)
        vectors.append(embedding.astype("float32"))
        kept.extend(valid)
        for image in images:
            image.close()
        bar.update(len(batch))
    bar.close()

    if not vectors:
        raise ValueError("No valid map/geo images to index")
    matrix = np.concatenate(vectors, axis=0).astype("float32")
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "image.faiss"))
    atomic_write_jsonl(index_dir / "image_metadata.jsonl", kept)
    manifest = {
        "type": "image",
        "model": model_name,
        "count": len(kept),
        "dimension": int(matrix.shape[1]),
        "normalized": True,
        "faiss_metric": "inner_product",
        "device": device,
    }
    atomic_write_json(index_dir / "image_manifest.json", manifest)
    return manifest
