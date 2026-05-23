from __future__ import annotations

import math

import torch


def ranking_metrics(scores: torch.Tensor, target: torch.Tensor, topk: list[int]) -> dict[str, float]:
    max_k = max(topk)
    _, indices = torch.topk(scores, k=max_k, dim=1)
    target = target.view(-1, 1)
    hits = indices.eq(target)
    result: dict[str, float] = {}
    batch_size = scores.size(0)
    for k in topk:
        hit_k = hits[:, :k]
        any_hit = hit_k.any(dim=1).float()
        result[f"Hit@{k}"] = any_hit.mean().item()
        result[f"Recall@{k}"] = any_hit.mean().item()
        result[f"Precision@{k}"] = (hit_k.float().sum(dim=1) / k).mean().item()
        reciprocal = torch.zeros(batch_size, device=scores.device)
        if hit_k.any():
            positions = hit_k.float().argmax(dim=1) + 1
            reciprocal = torch.where(any_hit.bool(), 1.0 / positions.float(), reciprocal)
        result[f"MRR@{k}"] = reciprocal.mean().item()
        discounts = 1.0 / torch.log2(torch.arange(2, k + 2, device=scores.device).float())
        dcg = (hit_k.float() * discounts.view(1, -1)).sum(dim=1)
        result[f"NDCG@{k}"] = dcg.mean().item()
    return result


def merge_metric_dicts(metric_dicts: list[dict[str, float]], weights: list[int]) -> dict[str, float]:
    total = sum(weights)
    if total == 0:
        return {}
    keys = metric_dicts[0].keys() if metric_dicts else []
    return {key: sum(metrics[key] * weight for metrics, weight in zip(metric_dicts, weights)) / total for key in keys}
