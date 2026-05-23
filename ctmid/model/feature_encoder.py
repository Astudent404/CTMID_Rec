from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ctmid.data.feature_store import ItemFeatureStore


class ModalityFeatureEncoder(nn.Module):
    def __init__(self, feature_store: ItemFeatureStore, modalities: Iterable[str], hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.hidden_size = hidden_size
        self.n_items = feature_store.n_items
        self.register_buffer("category_ids", feature_store.category_ids.long())
        self.register_buffer("price_value", feature_store.price_value.float())
        self.register_buffer("price_missing", feature_store.price_missing.float())
        self.register_buffer("rating_stats", feature_store.rating_stats.float())
        self.register_buffer("text_features", feature_store.text_features.float() if feature_store.text_features is not None else None)
        self.register_buffer("image_features", feature_store.image_features.float() if feature_store.image_features is not None else None)

        self.item_id_embedding = nn.Embedding(self.n_items, hidden_size, padding_idx=0)
        category_vocab_size = int(self.category_ids.max().item()) + 1 if self.category_ids.numel() else 1
        self.category_embedding = nn.Embedding(max(category_vocab_size, 1), hidden_size, padding_idx=0)
        self.category_projection = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size))
        self.price_projection = nn.Sequential(nn.Linear(2, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.rating_projection = nn.Sequential(nn.Linear(4, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.text_missing = nn.Parameter(torch.zeros(hidden_size))
        self.image_missing = nn.Parameter(torch.zeros(hidden_size))
        self.text_projection = self._optional_projection(self.text_features)
        self.image_projection = self._optional_projection(self.image_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, item_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        item_ids = item_ids.long().clamp(min=0, max=self.n_items - 1)
        outputs: dict[str, torch.Tensor] = {}
        if "id" in self.modalities:
            outputs["id"] = self.dropout(self.item_id_embedding(item_ids))
        if "category" in self.modalities:
            category = self.category_ids[item_ids].clamp(min=0, max=self.category_embedding.num_embeddings - 1)
            outputs["category"] = self.dropout(self.category_projection(self.category_embedding(category)))
        if "price" in self.modalities:
            price = torch.cat([self.price_value[item_ids], self.price_missing[item_ids]], dim=-1)
            outputs["price"] = self.dropout(self.price_projection(price))
        if "rating" in self.modalities:
            outputs["rating"] = self.dropout(self.rating_projection(self.rating_stats[item_ids]))
        if "text" in self.modalities:
            outputs["text"] = self.dropout(self._encode_optional(item_ids, "text"))
        if "image" in self.modalities:
            outputs["image"] = self.dropout(self._encode_optional(item_ids, "image"))
        return outputs

    def all_item_embeddings(self) -> dict[str, torch.Tensor]:
        item_ids = torch.arange(self.n_items, device=self.category_ids.device)
        return self.forward(item_ids)

    def _encode_optional(self, item_ids: torch.Tensor, modality: str) -> torch.Tensor:
        matrix = getattr(self, f"{modality}_features")
        missing = getattr(self, f"{modality}_missing")
        projection = getattr(self, f"{modality}_projection")
        if matrix is None:
            shape = (*item_ids.shape, self.hidden_size)
            return missing.view(*([1] * item_ids.dim()), self.hidden_size).expand(shape)
        return projection(matrix[item_ids])

    def _optional_projection(self, matrix: torch.Tensor | None) -> nn.Module:
        if matrix is None:
            return nn.Identity()
        return nn.Sequential(nn.Linear(matrix.size(1), self.hidden_size), nn.GELU(), nn.LayerNorm(self.hidden_size))
