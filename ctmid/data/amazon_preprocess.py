"""Dataset-agnostic, memory-efficient preprocessing for Amazon Reviews 2023.

Pipeline: stream reviews -> intern string ids to integers -> exact-duplicate
removal -> iterative k-core filtering -> re-index -> build user sequences ->
two split protocols (chronological 80/10/10 and leave-one-out) -> stream meta
once for item features. Designed to scale from All_Beauty (~0.7M) to
Home_and_Kitchen (~67M interactions) within a ~100 GB RAM budget.
"""
from __future__ import annotations

import gc
import gzip
import json
import math
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import time
from array import array
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

VALID_PROTOCOLS = ("chronological", "leave_one_out")


# --------------------------------------------------------------------------
# Raw file readers
# --------------------------------------------------------------------------
def _iter_raw_lines(path: str | Path, limit: int | None = None) -> Iterable[bytes]:
    """Yield raw (bytes) lines from a .jsonl.gz file, using the fastest reader."""
    path = str(path)
    decompressor = shutil.which("pigz") or shutil.which("zcat")
    count = 0
    if decompressor:
        cmd = [decompressor, "-dc", path] if "pigz" in os.path.basename(decompressor) else [decompressor, path]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 22)
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if limit is not None and count >= limit:
                    break
                count += 1
                yield line
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.terminate()
            proc.wait()
    else:
        with gzip.open(path, "rb") as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                count += 1
                yield line


def iter_jsonl_gz(path: str | Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    """Backward-compatible streaming JSON reader."""
    for line in _iter_raw_lines(path, limit=limit):
        if line.strip():
            yield json.loads(line)


def _batched(iterator: Iterable[bytes], batch_size: int) -> Iterable[list[bytes]]:
    batch: list[bytes] = []
    for line in iterator:
        batch.append(line)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------
# Worker functions (module-level so they are picklable for multiprocessing)
# --------------------------------------------------------------------------
def _parse_review_batch(batch: list[bytes]) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for raw in batch:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        user_key = obj.get("user_id")
        item_key = obj.get("parent_asin") or obj.get("asin")
        timestamp = obj.get("timestamp")
        if not user_key or not item_key or timestamp is None:
            continue
        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError):
            continue
        if timestamp <= 0:
            continue
        out.append((user_key, item_key, timestamp))
    return out


def _parse_meta_batch(batch: list[bytes]) -> list[tuple[str, str, float, float, float, str, list[str]]]:
    out: list[tuple[str, str, float, float, float, str, list[str]]] = []
    for raw in batch:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        item_key = obj.get("parent_asin")
        if not item_key:
            continue
        out.append(
            (
                item_key,
                _category_for(obj),
                _safe_float(obj.get("price")),
                _safe_float(obj.get("average_rating")),
                _safe_float(obj.get("rating_number")),
                _item_text(obj),
                _extract_image_urls(obj.get("images") or []),
            )
        )
    return out


def _imap(pool: mp.pool.Pool | None, func, tasks: Iterable, desc: str) -> Iterable:
    """Map ``func`` over ``tasks`` using ``pool`` if available, else serially."""
    if pool is None:
        for task in tqdm(tasks, desc=desc, unit="batch"):
            yield func(task)
    else:
        for result in tqdm(pool.imap(func, tasks, chunksize=1), desc=desc, unit="batch"):
            yield result


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def preprocess_amazon(
    config: dict[str, Any],
    limit: int | None = None,
    dry_run: bool = False,
    num_workers: int | None = None,
) -> dict[str, Any]:
    """Clean, filter and slice an Amazon Reviews dataset into train/valid/test.

    ``config['data']`` keys used: raw_inter_path, raw_meta_path, processed_dir,
    min_user_inter, min_item_inter, max_seq_len, split ('both' | 'chronological'
    | 'leave_one_out'), batch_size_lines (optional).
    """
    data_cfg = config["data"]
    raw_inter_path = Path(data_cfg["raw_inter_path"])
    raw_meta_path = Path(data_cfg["raw_meta_path"])
    root_dir, _ = _resolve_dirs(Path(data_cfg["processed_dir"]))
    min_user_inter = int(data_cfg.get("min_user_inter", 5))
    min_item_inter = int(data_cfg.get("min_item_inter", 5))
    max_seq_len = int(data_cfg.get("max_seq_len", 50))
    split_mode = str(data_cfg.get("split", "both")).lower()
    protocols = _resolve_protocols(split_mode)
    batch_lines = int(data_cfg.get("batch_size_lines", 50_000))
    if num_workers is None:
        num_workers = min(32, max(1, (os.cpu_count() or 4) - 2))

    started = time.time()
    print(f"[preprocess] reviews={raw_inter_path.name} meta={raw_meta_path.name} workers={num_workers}")

    # Step 1-2: stream + intern + dedup ------------------------------------
    user_arr, item_arr, ts_arr, user2idx, item2idx = _load_interactions(
        raw_inter_path, limit=limit, num_workers=num_workers, batch_lines=batch_lines
    )
    raw_count = int(user_arr.shape[0])
    user_arr, item_arr, ts_arr = _drop_exact_duplicates(user_arr, item_arr, ts_arr)
    dedup_count = int(user_arr.shape[0])
    print(f"[preprocess] raw={raw_count:,} after_dedup={dedup_count:,} "
          f"unique_users={len(user2idx):,} unique_items={len(item2idx):,}")

    # Step 3: iterative k-core filtering -----------------------------------
    user_arr, item_arr, ts_arr = _kcore_filter(user_arr, item_arr, ts_arr, min_user_inter, min_item_inter)
    filtered_count = int(user_arr.shape[0])
    print(f"[preprocess] {min_user_inter}-core filtered={filtered_count:,}")

    if filtered_count == 0:
        stats = {
            "dataset": root_dir.name,
            "raw_interactions": raw_count,
            "after_dedup": dedup_count,
            "filtered_interactions": 0,
            "note": "no interactions survived k-core filtering (try a larger --limit)",
        }
        if not dry_run:
            root_dir.mkdir(parents=True, exist_ok=True)
            _write_json(root_dir / "stats.json", stats)
        return stats

    # Step 4: re-index users/items to contiguous 1..N (0 = PAD) ------------
    user2id, item2id, user_new, item_new = _reindex(user_arr, item_arr, user2idx, item2idx)
    del user_arr, item_arr, user2idx, item2idx
    gc.collect()
    n_users, n_items = len(user2id), len(item2id)
    print(f"[preprocess] re-indexed users={n_users:,} items={n_items:,}")

    # Step 5: build per-user sequences (sorted by user, then timestamp) ----
    user_s, item_s, ts_s, starts, ends = _build_user_sequences(user_new, item_new, ts_arr)
    del user_new, item_new
    gc.collect()
    t80 = float(np.quantile(ts_arr.astype(np.float64), 0.8))
    t90 = float(np.quantile(ts_arr.astype(np.float64), 0.9))
    seq_lengths = (ends - starts).astype(np.float64)
    ts_min, ts_max = int(ts_arr.min()), int(ts_arr.max())
    del ts_arr
    gc.collect()

    stats: dict[str, Any] = {
        "dataset": root_dir.name,
        "raw_interactions": raw_count,
        "after_dedup": dedup_count,
        "filtered_interactions": filtered_count,
        "num_users": n_users,
        "num_items": n_items,
        "max_seq_len": max_seq_len,
        "min_user_inter": min_user_inter,
        "min_item_inter": min_item_inter,
        "avg_seq_len": round(float(seq_lengths.mean()), 4),
        "max_user_seq_len": int(seq_lengths.max()),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "chronological_t80": t80,
        "chronological_t90": t90,
        "protocols": list(protocols),
        "splits": {},
    }

    # Step 6: build splits for each protocol -------------------------------
    split_counts: dict[str, dict[str, int]] = {}
    if not dry_run:
        root_dir.mkdir(parents=True, exist_ok=True)
        for protocol in protocols:
            counts = _build_and_write_splits(
                protocol, root_dir / protocol, user_s, item_s, ts_s, starts, ends, max_seq_len, t80, t90
            )
            split_counts[protocol] = counts
            print(f"[preprocess] protocol={protocol} splits={counts}")
    else:
        for protocol in protocols:
            split_counts[protocol] = _count_splits(protocol, ts_s, starts, ends, t80, t90)
    stats["splits"] = split_counts
    del user_s, item_s, ts_s, starts, ends
    gc.collect()

    # Step 7: item features (stream meta once) -----------------------------
    item_features, item_text, image_urls, category2id = _build_item_features(
        raw_meta_path, item2id, limit=limit, num_workers=num_workers, batch_lines=batch_lines
    )
    stats["num_categories"] = len(category2id)
    stats["price_missing_ratio"] = round(float(item_features["price_missing"].mean()), 4)
    stats["has_image_urls"] = any(entry["urls"] for entry in image_urls)
    stats["elapsed_seconds"] = round(time.time() - started, 1)

    if dry_run:
        print(f"[preprocess] dry-run done in {stats['elapsed_seconds']}s (no files written)")
        return stats

    # Step 8: write shared artifacts ---------------------------------------
    _write_json(root_dir / "user2id.json", user2id)
    _write_json(root_dir / "item2id.json", item2id)
    _write_json(root_dir / "category2id.json", category2id)
    _write_json(root_dir / "stats.json", stats)
    with (root_dir / "item_text.jsonl").open("w", encoding="utf-8") as f:
        for entry in item_text:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    with (root_dir / "image_urls.jsonl").open("w", encoding="utf-8") as f:
        for entry in image_urls:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    for protocol in protocols:
        (root_dir / protocol).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root_dir / protocol / "item_features.npz", **item_features)
    print(f"[preprocess] done in {stats['elapsed_seconds']}s -> {root_dir}")
    return stats


# Backward-compatible alias for the original All_Beauty-specific name.
preprocess_all_beauty = preprocess_amazon


# --------------------------------------------------------------------------
# Step helpers
# --------------------------------------------------------------------------
def _resolve_dirs(processed_dir: Path) -> tuple[Path, str | None]:
    """Return (dataset_root, protocol). ``processed_dir`` may point at the root
    or at a protocol subdirectory (chronological / leave_one_out)."""
    if processed_dir.name in VALID_PROTOCOLS:
        return processed_dir.parent, processed_dir.name
    return processed_dir, None


def _resolve_protocols(split_mode: str) -> tuple[str, ...]:
    if split_mode in ("both", "all"):
        return VALID_PROTOCOLS
    if split_mode in VALID_PROTOCOLS:
        return (split_mode,)
    if split_mode in ("loo", "leave-one-out"):
        return ("leave_one_out",)
    if split_mode in ("chrono", "global", "temporal"):
        return ("chronological",)
    raise ValueError(f"Unsupported split mode: {split_mode}")


def _load_interactions(
    path: Path, limit: int | None, num_workers: int, batch_lines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    user2idx: dict[str, int] = {}
    item2idx: dict[str, int] = {}
    users = array("i")
    items = array("i")
    times = array("q")
    pool = mp.Pool(num_workers) if num_workers > 1 else None
    try:
        tasks = _batched(_iter_raw_lines(path, limit=limit), batch_lines)
        for parsed in _imap(pool, _parse_review_batch, tasks, desc="parse reviews"):
            for user_key, item_key, timestamp in parsed:
                ui = user2idx.get(user_key)
                if ui is None:
                    ui = len(user2idx)
                    user2idx[user_key] = ui
                ii = item2idx.get(item_key)
                if ii is None:
                    ii = len(item2idx)
                    item2idx[item_key] = ii
                users.append(ui)
                items.append(ii)
                times.append(timestamp)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    user_arr = np.frombuffer(users, dtype=np.int32).copy()
    item_arr = np.frombuffer(items, dtype=np.int32).copy()
    ts_arr = np.frombuffer(times, dtype=np.int64).copy()
    del users, items, times
    gc.collect()
    return user_arr, item_arr, ts_arr, user2idx, item2idx


def _drop_exact_duplicates(
    user_arr: np.ndarray, item_arr: np.ndarray, ts_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove rows with identical (user, item, timestamp)."""
    if user_arr.shape[0] == 0:
        return user_arr, item_arr, ts_arr
    order = np.lexsort((ts_arr, item_arr, user_arr))
    u, i, t = user_arr[order], item_arr[order], ts_arr[order]
    keep = np.ones(u.shape[0], dtype=bool)
    keep[1:] = (u[1:] != u[:-1]) | (i[1:] != i[:-1]) | (t[1:] != t[:-1])
    return u[keep], i[keep], t[keep]


def _kcore_filter(
    user_arr: np.ndarray, item_arr: np.ndarray, ts_arr: np.ndarray, min_user: int, min_item: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively drop users/items with too few interactions until stable."""
    while True:
        n0 = user_arr.shape[0]
        if n0 == 0:
            break
        user_counts = np.bincount(user_arr)
        item_counts = np.bincount(item_arr)
        keep = (user_counts[user_arr] >= min_user) & (item_counts[item_arr] >= min_item)
        if keep.all():
            break
        user_arr, item_arr, ts_arr = user_arr[keep], item_arr[keep], ts_arr[keep]
        if user_arr.shape[0] == n0:
            break
    return user_arr, item_arr, ts_arr


def _reindex(
    user_arr: np.ndarray, item_arr: np.ndarray, user2idx: dict[str, int], item2idx: dict[str, int]
) -> tuple[dict[str, int], dict[str, int], np.ndarray, np.ndarray]:
    """Map surviving users/items to contiguous ids 1..N (0 reserved for PAD)."""
    uniq_users = np.unique(user_arr)
    uniq_items = np.unique(item_arr)
    user_remap = np.zeros(int(uniq_users.max()) + 1, dtype=np.int64)
    item_remap = np.zeros(int(uniq_items.max()) + 1, dtype=np.int64)
    user_remap[uniq_users] = np.arange(1, uniq_users.shape[0] + 1)
    item_remap[uniq_items] = np.arange(1, uniq_items.shape[0] + 1)
    user_new = user_remap[user_arr].astype(np.int32)
    item_new = item_remap[item_arr].astype(np.int32)

    idx2user = [""] * len(user2idx)
    for key, idx in user2idx.items():
        idx2user[idx] = key
    idx2item = [""] * len(item2idx)
    for key, idx in item2idx.items():
        idx2item[idx] = key
    user2id = {idx2user[int(old)]: int(new) for old, new in zip(uniq_users, range(1, uniq_users.shape[0] + 1))}
    item2id = {idx2item[int(old)]: int(new) for old, new in zip(uniq_items, range(1, uniq_items.shape[0] + 1))}
    return user2id, item2id, user_new, item_new


def _build_user_sequences(
    user_new: np.ndarray, item_new: np.ndarray, ts_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort interactions by (user, timestamp) and return per-user segment bounds."""
    order = np.lexsort((ts_arr, user_new))
    user_s = user_new[order]
    item_s = item_new[order]
    ts_s = ts_arr[order]
    if user_s.shape[0] == 0:
        empty = np.zeros(0, dtype=np.int64)
        return user_s, item_s, ts_s, empty, empty
    boundaries = np.flatnonzero(user_s[1:] != user_s[:-1]) + 1
    starts = np.concatenate([[0], boundaries]).astype(np.int64)
    ends = np.concatenate([boundaries, [user_s.shape[0]]]).astype(np.int64)
    return user_s, item_s, ts_s, starts, ends


def _make_sample(
    user_id: int,
    items: list[int],
    times: list[int],
    target_index: int,
    max_seq_len: int,
    include_history: bool,
) -> dict[str, Any]:
    start = max(0, target_index - max_seq_len)
    sample = {
        "user_id": user_id,
        "item_seq": items[start:target_index],
        "time_seq": times[start:target_index],
        "seq_len": target_index - start,
        "target_item": items[target_index],
    }
    if include_history:
        sample["history_items"] = items[:target_index]
    return sample


def _iter_split_samples(
    protocol: str,
    user_s: np.ndarray,
    item_s: np.ndarray,
    ts_s: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    max_seq_len: int,
    t80: float,
    t90: float,
):
    """Yield (split_name, sample) pairs for one protocol."""
    for k in range(starts.shape[0]):
        s, e = int(starts[k]), int(ends[k])
        n = e - s
        if n < 2:
            continue
        user_id = int(user_s[s])
        items = item_s[s:e].tolist()
        times = ts_s[s:e].tolist()
        if protocol == "leave_one_out":
            if n < 3:
                continue
            for ti in range(1, n - 2):
                yield "train", _make_sample(user_id, items, times, ti, max_seq_len, False)
            yield "valid", _make_sample(user_id, items, times, n - 2, max_seq_len, True)
            yield "test", _make_sample(user_id, items, times, n - 1, max_seq_len, True)
        else:  # chronological
            for ti in range(1, n):
                target_time = times[ti]
                if target_time < t80:
                    split = "train"
                elif target_time < t90:
                    split = "valid"
                else:
                    split = "test"
                yield split, _make_sample(user_id, items, times, ti, max_seq_len, split != "train")


def _build_and_write_splits(
    protocol: str,
    out_dir: Path,
    user_s: np.ndarray,
    item_s: np.ndarray,
    ts_s: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    max_seq_len: int,
    t80: float,
    t90: float,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for split, sample in _iter_split_samples(protocol, user_s, item_s, ts_s, starts, ends, max_seq_len, t80, t90):
        buckets[split].append(sample)
    counts: dict[str, int] = {}
    for split, samples in buckets.items():
        with (out_dir / f"{split}.pkl").open("wb") as f:
            pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
        counts[split] = len(samples)
        samples.clear()
    buckets.clear()
    gc.collect()
    return counts


def _count_splits(
    protocol: str, ts_s: np.ndarray, starts: np.ndarray, ends: np.ndarray, t80: float, t90: float
) -> dict[str, int]:
    counts = {"train": 0, "valid": 0, "test": 0}
    for k in range(starts.shape[0]):
        s, e = int(starts[k]), int(ends[k])
        n = e - s
        if n < 2:
            continue
        if protocol == "leave_one_out":
            if n < 3:
                continue
            counts["train"] += n - 3
            counts["valid"] += 1
            counts["test"] += 1
        else:
            times = ts_s[s:e]
            for ti in range(1, n):
                target_time = times[ti]
                if target_time < t80:
                    counts["train"] += 1
                elif target_time < t90:
                    counts["valid"] += 1
                else:
                    counts["test"] += 1
    return counts


def _build_item_features(
    meta_path: Path,
    item2id: dict[str, int],
    limit: int | None,
    num_workers: int,
    batch_lines: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    n_items = len(item2id) + 1
    price = np.full(n_items, np.nan, dtype=np.float32)
    average_rating = np.full(n_items, np.nan, dtype=np.float32)
    rating_number = np.full(n_items, np.nan, dtype=np.float32)
    raw_category = ["Unknown"] * n_items
    texts = [""] * n_items
    images: list[list[str]] = [[] for _ in range(n_items)]

    pool = mp.Pool(num_workers) if num_workers > 1 else None
    try:
        tasks = _batched(_iter_raw_lines(meta_path, limit=limit), batch_lines)
        for parsed in _imap(pool, _parse_meta_batch, tasks, desc="parse meta"):
            for item_key, category, item_price, avg_rating, num_rating, text, image_list in parsed:
                item_id = item2id.get(item_key)
                if item_id is None:
                    continue
                price[item_id] = item_price
                average_rating[item_id] = avg_rating
                rating_number[item_id] = num_rating
                raw_category[item_id] = category
                texts[item_id] = text
                images[item_id] = image_list
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    categories = sorted(set(raw_category[1:]))
    category2id = {category: idx + 1 for idx, category in enumerate(categories)}
    category_ids = np.zeros(n_items, dtype=np.int64)
    for item_id in range(1, n_items):
        category_ids[item_id] = category2id.get(raw_category[item_id], 0)

    price_value, price_missing = _normalize_log_price(price)
    rating_value, rating_missing = _normalize_numeric(average_rating)
    count_value, count_missing = _normalize_numeric(np.log1p(np.clip(np.nan_to_num(rating_number, nan=0.0), 0, None)))
    item_features = {
        "category_ids": category_ids,
        "price_value": price_value[:, None].astype(np.float32),
        "price_missing": price_missing[:, None].astype(np.float32),
        "rating_stats": np.stack([rating_value, rating_missing, count_value, count_missing], axis=1).astype(np.float32),
    }

    id2item = {idx: key for key, idx in item2id.items()}
    item_text = [{"item_id": 0, "item_key": "[PAD]", "text": ""}]
    image_urls = [{"item_id": 0, "item_key": "[PAD]", "urls": []}]
    for item_id in range(1, n_items):
        item_key = id2item.get(item_id, "")
        item_text.append({"item_id": item_id, "item_key": item_key, "text": texts[item_id]})
        image_urls.append({"item_id": item_id, "item_key": item_key, "urls": images[item_id]})
    return item_features, item_text, image_urls, category2id


# --------------------------------------------------------------------------
# Normalization / parsing utilities (shared with worker functions)
# --------------------------------------------------------------------------
def _normalize_numeric(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing = ~np.isfinite(values)
    valid = values[~missing]
    if valid.size == 0:
        return np.zeros_like(values, dtype=np.float32), missing.astype(np.float32)
    mean = float(valid.mean())
    std = float(valid.std()) or 1.0
    filled = np.where(missing, mean, values)
    return ((filled - mean) / std).astype(np.float32), missing.astype(np.float32)


def _normalize_log_price(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    missing = ~np.isfinite(values) | (values < 0)
    transformed = np.zeros_like(values, dtype=np.float32)
    transformed[~missing] = np.log1p(values[~missing])
    valid = transformed[~missing]
    if valid.size == 0:
        return np.zeros_like(values, dtype=np.float32), missing.astype(np.float32)
    mean = float(valid.mean())
    std = float(valid.std()) or 1.0
    transformed[missing] = mean
    return ((transformed - mean) / std).astype(np.float32), missing.astype(np.float32)


def _category_for(meta: dict[str, Any]) -> str:
    category = meta.get("main_category")
    if not category:
        categories = meta.get("categories")
        if isinstance(categories, list) and categories:
            category = categories[-1]
    category = str(category).strip() if category else ""
    return category or "Unknown"


def _item_text(meta: dict[str, Any]) -> str:
    pieces = [meta.get("title"), meta.get("store"), meta.get("main_category")]
    pieces.extend(meta.get("features") or [])
    pieces.extend(meta.get("description") or [])
    details = meta.get("details") or {}
    if isinstance(details, dict):
        pieces.extend(f"{k}: {v}" for k, v in details.items())
    return " ".join(_to_text(piece) for piece in pieces if _to_text(piece))


def _extract_image_urls(images: list[Any]) -> list[str]:
    urls: list[str] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        for key in ("hi_res", "large", "thumb", "large_image_url", "medium_image_url", "small_image_url"):
            value = image.get(key)
            if value and isinstance(value, str):
                urls.append(value)
    return list(dict.fromkeys(urls))


def _safe_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_to_text(v) for v in value)
    return str(value).strip()


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
