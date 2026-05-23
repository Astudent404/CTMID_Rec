from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class MultiGranularityTimeEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        modalities: Iterable[str],
        dropout_prob: float = 0.1,
        adaptive_weights: bool = True,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.modalities = list(modalities)
        self.enabled = enabled
        self.granularities = ["year", "month", "week", "day", "interval"]
        self.projections = nn.ModuleDict(
            {name: nn.Linear(2, hidden_size) for name in self.granularities}
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.adaptive_weights = adaptive_weights
        if adaptive_weights:
            self.modality_logits = nn.ParameterDict(
                {
                    modality: nn.Parameter(self._initial_logits(modality))
                    for modality in self.modalities
                }
            )
        else:
            self.register_buffer("uniform_weights", torch.full((len(self.granularities),), 1.0 / len(self.granularities)))

    def _initial_logits(self, modality: str) -> torch.Tensor:
        priors = torch.zeros(len(self.granularities), dtype=torch.float)
        if modality in {"image", "category"}:
            priors += torch.tensor([1.2, 0.8, 0.3, 0.0, -0.2])
        elif modality in {"price", "rating"}:
            priors += torch.tensor([-0.2, 0.2, 0.7, 1.0, 0.9])
        elif modality == "text":
            priors += torch.tensor([0.5, 0.7, 0.5, 0.3, 0.0])
        return priors

    def forward(self, timestamps: torch.Tensor | None, modality: str) -> torch.Tensor | None:
        if not self.enabled or timestamps is None:
            return None
        timestamps = timestamps.float()
        if timestamps.numel() == 0:
            return None
        scale = self._timestamp_scale(timestamps)
        seconds = timestamps / scale
        days = seconds / 86400.0
        features = {
            "year": self._sin_cos(days / 365.2425),
            "month": self._sin_cos(days / 30.436875),
            "week": self._sin_cos(days / 7.0),
            "day": self._sin_cos(days),
            "interval": self._sin_cos(self._relative_interval(days)),
        }
        encoded = []
        for name in self.granularities:
            encoded.append(self.projections[name](features[name]))
        stack = torch.stack(encoded, dim=-2)
        weights = self._weights_for(modality, stack.device, stack.dtype)
        weight_shape = [1] * (stack.dim() - 2) + [len(self.granularities), 1]
        output = (stack * weights.view(*weight_shape)).sum(dim=-2)
        return self.dropout(output)

    def _weights_for(self, modality: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.adaptive_weights:
            logits = self.modality_logits[modality] if modality in self.modality_logits else next(iter(self.modality_logits.values()))
            return torch.softmax(logits.to(device=device, dtype=dtype), dim=0)
        return self.uniform_weights.to(device=device, dtype=dtype)

    def _timestamp_scale(self, timestamps: torch.Tensor) -> float:
        max_value = float(torch.nan_to_num(timestamps.detach(), nan=0.0).abs().max().item())
        return 1000.0 if max_value > 10_000_000_000 else 1.0

    def _sin_cos(self, value: torch.Tensor) -> torch.Tensor:
        angle = 2.0 * math.pi * value
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def _relative_interval(self, days: torch.Tensor) -> torch.Tensor:
        if days.dim() < 2:
            return torch.zeros_like(days)
        interval = torch.zeros_like(days)
        interval[:, 1:] = torch.log1p(torch.relu(days[:, 1:] - days[:, :-1]))
        return interval

    def modality_weight_table(self) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for modality in self.modalities:
            weights = self._weights_for(modality, torch.device("cpu"), torch.float).detach().cpu()
            table[modality] = {name: float(weights[i].item()) for i, name in enumerate(self.granularities)}
        return table
