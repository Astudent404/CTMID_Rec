from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.data import ItemFeatureStore, SequentialDataset, sequential_collate_fn
from ctmid.model.ctmid import CTMID
from ctmid.training.trainer import Trainer
from ctmid.utils.config import load_merged_config, resolve_project_paths
from ctmid.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTMID with a native PyTorch pipeline.")
    parser.add_argument("--config", action="append", default=[], help="YAML config path. Can be specified multiple times.")
    parser.add_argument("--no-train", action="store_true", help="Build data loaders and model, then exit before training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = args.config or [str(PROJECT_ROOT / "configs" / "all_beauty_ctmid.yaml")]
    config = resolve_project_paths(load_merged_config(config_paths), PROJECT_ROOT)
    set_seed(int(config.get("train", {}).get("seed", 2026)))
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    processed_dir = data_cfg["processed_dir"]
    feature_store = ItemFeatureStore.load(
        processed_dir,
        text_feature_path=data_cfg.get("text_feature_path"),
        image_feature_path=data_cfg.get("image_feature_path"),
    )
    train_dataset = SequentialDataset(processed_dir, "train")
    valid_dataset = SequentialDataset(processed_dir, "valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=sequential_collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.get("eval", {}).get("batch_size", train_cfg.get("batch_size", 256))),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=sequential_collate_fn,
    )
    model = CTMID(config, feature_store)
    print(f"model_items={feature_store.n_items} train_samples={len(train_dataset)} valid_samples={len(valid_dataset)}")
    if args.no_train:
        print("--no-train set; exiting before training.")
        return
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    trainer = Trainer(model, optimizer, train_loader, valid_loader, config)
    best_metrics = trainer.fit()
    print(f"best_valid={best_metrics}")


if __name__ == "__main__":
    main()
