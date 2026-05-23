from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctmid.training.metrics import merge_metric_dicts, ranking_metrics


class Evaluator:
    def __init__(
        self,
        topk: list[int],
        mask_history: bool = True,
        item_chunk_size: int | None = None,
        amp: bool = True,
    ) -> None:
        self.topk = topk
        self.mask_history = mask_history
        self.item_chunk_size = item_chunk_size
        self.amp = amp

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device | str) -> dict[str, float]:
        model.eval()
        device = torch.device(device)
        use_amp = self.amp and device.type == "cuda"
        metric_dicts: list[dict[str, float]] = []
        weights: list[int] = []
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            # The item catalogue representation is identical across eval batches
            # (parameters are frozen), so compute it once and reuse.
            item_repr = model.compute_item_representations(chunk_size=self.item_chunk_size)
            for batch in tqdm(dataloader, desc="evaluate", leave=False):
                tensor_batch = _move_batch(batch, device)
                scores = model.full_sort_predict(tensor_batch, item_repr=item_repr)
                if self.mask_history:
                    _mask_history(scores, batch.get("history_items", []))
                metrics = ranking_metrics(scores, tensor_batch["target_item"], self.topk)
                metric_dicts.append(metrics)
                weights.append(int(tensor_batch["target_item"].numel()))
        return merge_metric_dicts(metric_dicts, weights)


def _move_batch(batch: dict, device: torch.device | str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _mask_history(scores: torch.Tensor, histories: list[list[int]]) -> None:
    for row, history in enumerate(histories):
        if history:
            scores[row, torch.tensor(history, device=scores.device, dtype=torch.long)] = -torch.inf
