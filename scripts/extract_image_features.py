"""Download item images and extract frozen CLIP embeddings for CTMID.

Two phases (``--phase``):
  download : fetch the first image URL per item (resized small variant) into
             ``data/processed/{ds}/images/`` with a thread pool. Resumable.
  encode   : encode downloaded images with CLIP ViT-B/32 (512-dim) and write
             ``image_features.npy`` (n_items, 512); missing images -> zero rows.

After a successful encode the ``images/`` directory is removed unless
``--keep-images`` is set.

Examples
--------
    python scripts/extract_image_features.py --config configs/all_beauty_ctmid.yaml
    python scripts/extract_image_features.py --config configs/home_and_kitchen_ctmid.yaml --num-threads 128
    python scripts/extract_image_features.py --config configs/toys_and_games_ctmid.yaml --phase encode
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.utils.config import load_merged_config, resolve_project_paths

VALID_PROTOCOLS = ("chronological", "leave_one_out")
SHARD = 10000
_USER_AGENT = "Mozilla/5.0 (compatible; CTMID-research/1.0)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download item images and extract CLIP features for CTMID.")
    parser.add_argument("--config", action="append", default=[], required=True,
                        help="Dataset YAML config. Can be specified multiple times (merged).")
    parser.add_argument("--phase", choices=["download", "encode", "all"], default="all")
    parser.add_argument("--model", default="clip-ViT-B-32", help="sentence-transformers CLIP model.")
    parser.add_argument("--num-threads", type=int, default=64, help="Concurrent download threads.")
    parser.add_argument("--image-size", type=int, default=224, help="Target resized image edge (px).")
    parser.add_argument("--batch-size", type=int, default=256, help="CLIP encoding batch size.")
    parser.add_argument("--device", default="cuda", help="Encoding device.")
    parser.add_argument("--keep-images", action="store_true", help="Keep the images/ dir after encoding.")
    return parser.parse_args()


def _dataset_root(processed_dir: Path) -> Path:
    return processed_dir.parent if processed_dir.name in VALID_PROTOCOLS else processed_dir


def _shard_path(images_dir: Path, item_id: int) -> Path:
    return images_dir / str(item_id // SHARD) / f"{item_id}.jpg"


def _small_variant(url: str, size: int) -> str:
    """Rewrite an Amazon media URL to a resized variant (~size px edge)."""
    prefix, _, filename = url.rpartition("/")
    image_id = filename.split(".")[0]
    if prefix and image_id:
        return f"{prefix}/{image_id}._SX{size}_.jpg"
    return url


def _load_url_entries(text_path: Path) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            urls = entry.get("urls") or []
            if urls:
                entries.append((entry["item_id"], urls[0]))
    return entries


def _download_one(item_id: int, url: str, images_dir: Path, size: int) -> bool:
    out_path = _shard_path(images_dir, item_id)
    if out_path.exists():
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for candidate in (_small_variant(url, size), url):
        try:
            req = urllib.request.Request(candidate, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            if len(data) < 256:
                continue
            tmp = out_path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.rename(out_path)
            return True
        except Exception:
            continue
    return False


def _download_phase(entries: list[tuple[int, str]], images_dir: Path, size: int, num_threads: int) -> int:
    images_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_download_one, iid, url, images_dir, size) for iid, url in entries]
        for fut in tqdm(futures, desc="download images", unit="img"):
            if fut.result():
                ok += 1
    print(f"[image-features] downloaded {ok:,}/{len(entries):,} images")
    return ok


def _encode_phase(
    root: Path, images_dir: Path, n_items: int, model_name: str, batch_size: int, device: str
) -> np.ndarray:
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    dim = model.get_embedding_dimension()
    features = np.zeros((n_items, dim), dtype=np.float32)

    chunk_ids: list[int] = []
    chunk_imgs: list = []
    encode_batch = max(batch_size, 4096)  # group disk reads before encoding

    def flush() -> None:
        if not chunk_imgs:
            return
        vecs = model.encode(
            chunk_imgs, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        for idx, vec in zip(chunk_ids, vecs):
            features[idx] = vec
        chunk_ids.clear()
        chunk_imgs.clear()

    found = 0
    for item_id in tqdm(range(n_items), desc="encode images", unit="item"):
        path = _shard_path(images_dir, item_id)
        if not path.exists():
            continue
        try:
            with Image.open(path) as img:
                chunk_imgs.append(img.convert("RGB"))
            chunk_ids.append(item_id)
            found += 1
        except Exception:
            continue
        if len(chunk_imgs) >= encode_batch:
            flush()
    flush()
    print(f"[image-features] encoded {found:,}/{n_items:,} items (missing -> zero vector)")
    return features


def main() -> None:
    args = parse_args()
    config = resolve_project_paths(load_merged_config(args.config), PROJECT_ROOT)
    root = _dataset_root(Path(config["data"]["processed_dir"]))
    images_dir = root / "images"
    url_path = root / "image_urls.jsonl"
    if not url_path.exists():
        raise FileNotFoundError(f"missing {url_path}; run preprocessing first")

    n_items = sum(1 for _ in url_path.open("r", encoding="utf-8"))
    started = time.time()
    print(f"[image-features] dataset={root.name} n_items={n_items:,} phase={args.phase}")

    if args.phase in ("download", "all"):
        entries = _load_url_entries(url_path)
        print(f"[image-features] {len(entries):,} items have an image URL")
        _download_phase(entries, images_dir, args.image_size, args.num_threads)

    if args.phase in ("encode", "all"):
        features = _encode_phase(root, images_dir, n_items, args.model, args.batch_size, args.device)
        out_path = root / "image_features.npy"
        np.save(out_path, features)
        print(f"[image-features] saved {out_path} shape={features.shape}")
        if not args.keep_images and images_dir.exists():
            shutil.rmtree(images_dir)
            print(f"[image-features] removed {images_dir} (use --keep-images to retain)")

    print(f"[image-features] done in {round(time.time() - started, 1)}s")


if __name__ == "__main__":
    main()
