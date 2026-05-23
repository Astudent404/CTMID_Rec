from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


class SequentialDataset(Dataset):
    def __init__(self, processed_dir: str | Path, split: str) -> None:
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.processed_dir = Path(processed_dir)
        self.split = split
        with (self.processed_dir / f"{split}.pkl").open("rb") as f:
            self.samples: list[dict[str, Any]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]
