from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.data import ItemFeatureStore, SequentialDataset, sequential_collate_fn
from ctmid.model.ctmid import CTMID
from ctmid.training.metrics import ranking_metrics
from ctmid.utils.config import load_merged_config, resolve_project_paths
from ctmid.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RA-CTMID checkpoints with grouped diagnostics.")
    parser.add_argument("--config", action="append", default=[], help="YAML config path. Can be specified multiple times.")
    parser.add_argument("--split", choices=["valid", "test"], default="valid", help="Evaluation split.")
    parser.add_argument("--output", required=True, help="Path to the output diagnostics JSON.")
    return parser.parse_args()


class MetricAccumulator:
    def __init__(self, topk: list[int]) -> None:
        self.topk = topk
        self.count = 0
        self.sums: dict[str, float] = {}

    def add(self, scores: torch.Tensor, target: torch.Tensor) -> None:
        batch_size = int(target.numel())
        if batch_size == 0:
            return
        metrics = ranking_metrics(scores, target, self.topk)
        self.count += batch_size
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * batch_size

    def as_dict(self) -> dict[str, Any]:
        metrics = {key: value / self.count for key, value in self.sums.items()} if self.count else {}
        return {"count": self.count, "metrics": metrics}


class GroupMetricAccumulator:
    def __init__(self, topk: list[int]) -> None:
        self.topk = topk
        self.groups: dict[str, MetricAccumulator] = {}

    def add(self, labels: list[str], scores: torch.Tensor, target: torch.Tensor) -> None:
        if len(labels) != int(target.numel()):
            raise ValueError(f"Label count {len(labels)} != batch size {int(target.numel())}")
        for label in sorted(set(labels)):
            rows = [index for index, value in enumerate(labels) if value == label]
            if not rows:
                continue
            row_index = torch.tensor(rows, device=scores.device, dtype=torch.long)
            self.groups.setdefault(label, MetricAccumulator(self.topk)).add(scores.index_select(0, row_index), target.index_select(0, row_index))

    def as_dict(self) -> dict[str, Any]:
        return {label: self.groups[label].as_dict() for label in sorted(self.groups)}


class ComponentAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: dict[str, float] = {}
        self.sum_squares: dict[str, float] = {}

    def add(self, values: dict[str, torch.Tensor]) -> None:
        if not values:
            return
        batch_size = int(next(iter(values.values())).numel())
        self.count += batch_size
        for key, tensor in values.items():
            tensor = tensor.detach().float().cpu()
            self.sums[key] = self.sums.get(key, 0.0) + float(tensor.sum().item())
            self.sum_squares[key] = self.sum_squares.get(key, 0.0) + float(torch.square(tensor).sum().item())

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"count": self.count, "mean": {}, "std": {}}
        if not self.count:
            return result
        for key, value in sorted(self.sums.items()):
            mean = value / self.count
            variance = max(self.sum_squares[key] / self.count - mean * mean, 0.0)
            result["mean"][key] = mean
            result["std"][key] = float(np.sqrt(variance))
        return result


def main() -> None:
    args = parse_args()
    config_paths = args.config or [str(PROJECT_ROOT / "configs" / "all_beauty_ctmid.yaml")]
    config = resolve_project_paths(load_merged_config(config_paths), PROJECT_ROOT)
    set_seed(int(config.get("train", {}).get("seed", 2026)))

    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    processed_dir = Path(data_cfg["processed_dir"])
    checkpoint_path = Path(train_cfg.get("checkpoint_dir", "checkpoints")) / "best.pt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_store = ItemFeatureStore.load(
        processed_dir,
        text_feature_path=data_cfg.get("text_feature_path"),
        image_feature_path=data_cfg.get("image_feature_path"),
    )
    dataset = SequentialDataset(processed_dir, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=int(eval_cfg.get("batch_size", train_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=sequential_collate_fn,
    )

    device = _resolve_device(str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = CTMID(config, feature_store)
    checkpoint = _load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    topk = list(eval_cfg.get("topk", [10]))
    mask_history = True
    item_chunk_size = eval_cfg.get("item_chunk_size")
    amp = bool(train_cfg.get("amp", False))

    popularity_counts = _target_popularity_counts(processed_dir / "train.pkl", feature_store.n_items)
    popularity_thresholds = _tertile_thresholds(popularity_counts[popularity_counts > 0])
    time_scale = _time_scale(dataset.samples)
    delta_thresholds = _tertile_thresholds(np.asarray(_delta_proxy_values(dataset.samples, time_scale), dtype=np.float64))
    category_ids = feature_store.category_ids.detach().cpu().numpy()

    overall = MetricAccumulator(topk)
    groups = {
        "seq_len": GroupMetricAccumulator(topk),
        "delta_proxy": GroupMetricAccumulator(topk),
        "item_popularity": GroupMetricAccumulator(topk),
        "same_category": GroupMetricAccumulator(topk),
    }
    target_components = ComponentAccumulator()
    top1_components = ComponentAccumulator()

    model.eval()
    use_amp = amp and device.type == "cuda"
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        item_repr = model.compute_item_representations(chunk_size=item_chunk_size)
        for batch in tqdm(dataloader, desc=f"diagnostics:{args.split}", leave=False):
            tensor_batch = _move_batch(batch, device)
            scores, seq_output, modality_states = _full_sort_scores(model, tensor_batch, item_repr)
            if mask_history:
                _mask_history(scores, _history_items(batch))

            target = tensor_batch["target_item"].long()
            overall.add(scores, target)

            batch_labels = _batch_group_labels(batch, popularity_counts, popularity_thresholds, delta_thresholds, time_scale, category_ids)
            for name, labels in batch_labels.items():
                groups[name].add(labels, scores, target)

            top1_items = torch.argmax(scores, dim=1).long()
            target_components.add(_score_components(model, tensor_batch, seq_output, modality_states, item_repr, target))
            top1_components.add(_score_components(model, tensor_batch, seq_output, modality_states, item_repr, top1_items))

    payload = {
        "config_paths": [str(path) for path in config_paths],
        "split": args.split,
        "checkpoint_path": str(checkpoint_path),
        "num_samples": len(dataset),
        "topk": topk,
        "mask_history": mask_history,
        "requested_device": str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        "device": str(device),
        "overall": overall.as_dict(),
        "groups": {name: accumulator.as_dict() for name, accumulator in groups.items()},
        "score_components": {
            "target": target_components.as_dict(),
            "top1": top1_components.as_dict(),
        },
        "metadata": {
            "delta_proxy": {
                "source": "last two history timestamps",
                "scale": time_scale,
                "tertile_threshold_days": _json_thresholds(delta_thresholds),
                "labels": ["unknown", "short", "medium", "long"],
            },
            "item_popularity": {
                "source": "train.pkl target_item frequency",
                "tertile_threshold_counts": _json_thresholds(popularity_thresholds),
                "labels": ["tail", "mid", "popular"],
            },
            "same_category": {
                "source": "item_features.npz category_ids for target item and last history item",
                "labels": ["same", "different", "unknown"],
            },
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_checkpoint(path: Path, device: torch.device) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _move_batch(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _full_sort_scores(
    model: CTMID,
    batch: dict[str, torch.Tensor],
    item_repr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    seq_output, modality_states = model.forward(batch)
    scores = torch.matmul(seq_output, item_repr.transpose(0, 1)) + model.output_bias
    id_logits = model._id_residual_logits(modality_states)
    if id_logits.dim() != 0:
        scores = scores + id_logits
    last_logits = model._last_item_residual_logits(batch)
    if last_logits.dim() != 0:
        scores = scores + last_logits
    scores[:, model.pad_item_id] = -torch.inf
    return scores, seq_output, modality_states


def _mask_history(scores: torch.Tensor, histories: list[list[int]]) -> None:
    for row, history in enumerate(histories):
        if history:
            scores[row, torch.tensor(history, device=scores.device, dtype=torch.long)] = -torch.inf


def _history_items(batch: dict[str, Any]) -> list[list[int]]:
    histories = batch.get("history_items", [])
    item_seq = batch["item_seq"].detach().cpu().numpy()
    seq_len = batch["seq_len"].detach().cpu().numpy()
    result: list[list[int]] = []
    for row, length in enumerate(seq_len):
        if row < len(histories) and histories[row]:
            result.append(list(histories[row]))
        else:
            result.append(item_seq[row, : int(length)].astype(int).tolist())
    return result


def _score_components(
    model: CTMID,
    batch: dict[str, torch.Tensor],
    seq_output: torch.Tensor,
    modality_states: dict[str, torch.Tensor],
    item_repr: torch.Tensor,
    item_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    item_ids = item_ids.long()
    fused = (seq_output * item_repr[item_ids]).sum(dim=-1)
    id_residual = _batch_vector(model._id_residual_logits(modality_states, item_ids), item_ids.size(0), item_ids.device)
    last_residual = _batch_vector(model._last_item_residual_logits(batch, item_ids), item_ids.size(0), item_ids.device)
    return {
        "s_fuse": fused,
        "id_residual": id_residual,
        "last_residual": last_residual,
        "output_bias": model.output_bias[item_ids],
    }


def _batch_vector(value: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if value.dim() == 0:
        return torch.zeros(batch_size, device=device, dtype=torch.float)
    return value.float()


def _target_popularity_counts(train_path: Path, n_items: int) -> np.ndarray:
    with train_path.open("rb") as f:
        samples = pickle.load(f)
    counts = np.zeros(n_items, dtype=np.int64)
    for sample in samples:
        item_id = int(sample["target_item"])
        if 0 <= item_id < n_items:
            counts[item_id] += 1
    return counts


def _tertile_thresholds(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    low, high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    return float(low), float(high)


def _time_scale(samples: list[dict[str, Any]]) -> float:
    max_abs = 0.0
    for sample in samples:
        for value in sample.get("time_seq", []):
            max_abs = max(max_abs, abs(float(value)))
    return 1000.0 if max_abs > 10_000_000_000 else 1.0


def _delta_proxy_values(samples: list[dict[str, Any]], scale: float) -> list[float]:
    values: list[float] = []
    for sample in samples:
        if int(sample.get("seq_len", 0)) < 2:
            continue
        time_seq = sample.get("time_seq", [])
        if len(time_seq) < 2:
            continue
        values.append(max(float(time_seq[-1]) - float(time_seq[-2]), 0.0) / scale / 86400.0)
    return values


def _batch_group_labels(
    batch: dict[str, Any],
    popularity_counts: np.ndarray,
    popularity_thresholds: tuple[float | None, float | None],
    delta_thresholds: tuple[float | None, float | None],
    time_scale: float,
    category_ids: np.ndarray,
) -> dict[str, list[str]]:
    seq_len = batch["seq_len"].detach().cpu().numpy()
    target_item = batch["target_item"].detach().cpu().numpy()
    item_seq = batch["item_seq"].detach().cpu().numpy()
    time_seq = batch["time_seq"].detach().cpu().numpy()

    labels = {
        "seq_len": [],
        "delta_proxy": [],
        "item_popularity": [],
        "same_category": [],
    }
    for row, length_value in enumerate(seq_len):
        length = int(length_value)
        target = int(target_item[row])
        labels["seq_len"].append(_seq_len_label(length))
        labels["delta_proxy"].append(_delta_label(time_seq[row], length, delta_thresholds, time_scale))
        labels["item_popularity"].append(_popularity_label(popularity_counts[target] if 0 <= target < len(popularity_counts) else 0, popularity_thresholds))
        labels["same_category"].append(_same_category_label(item_seq[row], length, target, category_ids))
    return labels


def _seq_len_label(length: int) -> str:
    if length <= 5:
        return "<=5"
    if length <= 10:
        return "6-10"
    return ">10"


def _delta_label(time_row: np.ndarray, length: int, thresholds: tuple[float | None, float | None], scale: float) -> str:
    low, high = thresholds
    if length < 2 or low is None or high is None:
        return "unknown"
    delta_days = max(float(time_row[length - 1]) - float(time_row[length - 2]), 0.0) / scale / 86400.0
    if delta_days <= low:
        return "short"
    if delta_days <= high:
        return "medium"
    return "long"


def _popularity_label(count: int, thresholds: tuple[float | None, float | None]) -> str:
    low, high = thresholds
    if low is None or high is None:
        return "tail"
    if count <= low:
        return "tail"
    if count <= high:
        return "mid"
    return "popular"


def _same_category_label(item_row: np.ndarray, length: int, target: int, category_ids: np.ndarray) -> str:
    if length <= 0 or target < 0 or target >= len(category_ids):
        return "unknown"
    last_item = int(item_row[length - 1])
    if last_item < 0 or last_item >= len(category_ids):
        return "unknown"
    last_category = int(category_ids[last_item])
    target_category = int(category_ids[target])
    if last_category <= 0 or target_category <= 0:
        return "unknown"
    return "same" if last_category == target_category else "different"


def _json_thresholds(thresholds: tuple[float | None, float | None]) -> list[float | None]:
    return [None if value is None else float(value) for value in thresholds]


if __name__ == "__main__":
    main()
