from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ItemFeatureStore:
    category_ids: torch.Tensor
    price_value: torch.Tensor
    price_missing: torch.Tensor
    rating_stats: torch.Tensor
    text_features: torch.Tensor | None = None
    image_features: torch.Tensor | None = None

    @property
    def n_items(self) -> int:
        return int(self.category_ids.size(0))

    @classmethod
    def load(cls, processed_dir: str | Path, text_feature_path: str | None = None, image_feature_path: str | None = None) -> "ItemFeatureStore":
        processed_dir = Path(processed_dir)
        data = np.load(processed_dir / "item_features.npz")
        store = cls(
            category_ids=torch.from_numpy(data["category_ids"]).long(),
            price_value=torch.from_numpy(data["price_value"]).float(),
            price_missing=torch.from_numpy(data["price_missing"]).float(),
            rating_stats=torch.from_numpy(data["rating_stats"]).float(),
            text_features=_load_optional_matrix(text_feature_path),
            image_features=_load_optional_matrix(image_feature_path),
        )
        store.validate()
        return store

    def validate(self) -> None:
        n_items = self.n_items
        for name in ("price_value", "price_missing", "rating_stats"):
            value = getattr(self, name)
            if value.size(0) != n_items:
                raise ValueError(f"{name} row count {value.size(0)} != n_items {n_items}")
        for name in ("text_features", "image_features"):
            value = getattr(self, name)
            if value is not None and value.size(0) != n_items:
                raise ValueError(f"{name} row count {value.size(0)} != n_items {n_items}")

    def to(self, device: torch.device | str) -> "ItemFeatureStore":
        return ItemFeatureStore(
            category_ids=self.category_ids.to(device),
            price_value=self.price_value.to(device),
            price_missing=self.price_missing.to(device),
            rating_stats=self.rating_stats.to(device),
            text_features=None if self.text_features is None else self.text_features.to(device),
            image_features=None if self.image_features is None else self.image_features.to(device),
        )


def _load_optional_matrix(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    loaded = np.load(path) if str(path).endswith(".npy") else torch.load(path, map_location="cpu")
    if isinstance(loaded, dict):
        for key in ("features", "embeddings", "feat"):
            if key in loaded:
                loaded = loaded[key]
                break
    if not isinstance(loaded, torch.Tensor):
        loaded = torch.as_tensor(loaded)
    return loaded.float()
