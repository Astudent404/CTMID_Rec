"""Extract frozen item-text embeddings for CTMID.

Reads ``item_text.jsonl`` produced by preprocessing, encodes each item's text
with a sentence-transformers model, and writes ``text_features.npy`` of shape
(n_items, dim) to the dataset root. Row 0 (PAD) and empty-text rows are zeroed.

Examples
--------
    python scripts/extract_text_features.py --config configs/all_beauty_ctmid.yaml
    python scripts/extract_text_features.py --config configs/toys_and_games_ctmid.yaml --batch-size 1024
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# huggingface.co is unreachable on this server; route downloads through the mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.utils.config import load_merged_config, resolve_project_paths

VALID_PROTOCOLS = ("chronological", "leave_one_out")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode item text into frozen embeddings for CTMID.")
    parser.add_argument("--config", action="append", default=[], required=True,
                        help="Dataset YAML config. Can be specified multiple times (merged).")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="sentence-transformers model name.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Encoding batch size.")
    parser.add_argument("--device", default="cuda", help="Encoding device (cuda / cpu).")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Token truncation length.")
    return parser.parse_args()


def _dataset_root(processed_dir: Path) -> Path:
    return processed_dir.parent if processed_dir.name in VALID_PROTOCOLS else processed_dir


def main() -> None:
    args = parse_args()
    config = resolve_project_paths(load_merged_config(args.config), PROJECT_ROOT)
    root = _dataset_root(Path(config["data"]["processed_dir"]))
    text_path = root / "item_text.jsonl"
    if not text_path.exists():
        raise FileNotFoundError(f"missing {text_path}; run preprocessing first")

    # item_text.jsonl is ordered by item_id (0 = PAD).
    texts: list[str] = []
    with text_path.open("r", encoding="utf-8") as f:
        for expected_id, line in enumerate(f):
            entry = json.loads(line)
            if entry["item_id"] != expected_id:
                raise ValueError(f"item_text.jsonl not ordered at line {expected_id}: {entry['item_id']}")
            texts.append(entry.get("text") or "")
    n_items = len(texts)
    empty_mask = np.array([not t.strip() for t in texts], dtype=bool)
    print(f"[text-features] dataset={root.name} n_items={n_items:,} empty_text={int(empty_mask.sum()):,}")

    from sentence_transformers import SentenceTransformer

    started = time.time()
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length
    print(f"[text-features] model={args.model} device={args.device}")

    features = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    features[empty_mask] = 0.0  # PAD row and items without metadata text

    out_path = root / "text_features.npy"
    np.save(out_path, features)
    elapsed = round(time.time() - started, 1)
    print(f"[text-features] saved {out_path} shape={features.shape} in {elapsed}s")


if __name__ == "__main__":
    main()
