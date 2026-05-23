from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def normalize_numeric_feature(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = value.float()
    finite_mask = torch.isfinite(value)
    valid = value[finite_mask]
    if valid.numel() == 0:
        return torch.zeros_like(value), (~finite_mask).float()
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(1e-6)
    normalized = (torch.nan_to_num(value, nan=float(mean.item()), posinf=float(mean.item()), neginf=float(mean.item())) - mean) / std
    return normalized, (~finite_mask).float()


def normalize_log_price(price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    price = price.float()
    finite_mask = torch.isfinite(price) & (price >= 0)
    safe_price = torch.where(finite_mask, price, torch.zeros_like(price))
    transformed = torch.log1p(safe_price)
    valid = transformed[finite_mask]
    if valid.numel() == 0:
        return torch.zeros_like(price), (~finite_mask).float()
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(1e-6)
    normalized = torch.where(finite_mask, (transformed - mean) / std, torch.zeros_like(transformed))
    return normalized, (~finite_mask).float()


def load_optional_feature_matrix(path_value: str | None, n_items: int) -> torch.Tensor | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        matrix = torch.from_numpy(np.load(path))
    elif path.suffix.lower() in {".pt", ".pth"}:
        matrix = torch.load(path, map_location="cpu")
        if isinstance(matrix, dict):
            for key in ("features", "feat", "embedding", "embeddings"):
                if key in matrix:
                    matrix = matrix[key]
                    break
    else:
        raise ValueError(f"Unsupported feature file extension: {path}")
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.as_tensor(matrix)
    matrix = matrix.float().detach().cpu()
    if matrix.dim() != 2:
        raise ValueError(f"Feature matrix must be 2-D, got shape {tuple(matrix.shape)} from {path}")
    if matrix.size(0) == n_items - 1:
        pad = torch.zeros(1, matrix.size(1), dtype=matrix.dtype)
        matrix = torch.cat([pad, matrix], dim=0)
    if matrix.size(0) != n_items:
        raise ValueError(f"Feature row count {matrix.size(0)} does not match n_items={n_items}: {path}")
    return matrix
