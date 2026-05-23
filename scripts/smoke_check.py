from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ctmid.data.feature_store import ItemFeatureStore
from ctmid.model.ctmid import CTMID
from ctmid.training.metrics import ranking_metrics
from ctmid.utils.config import load_config
from ctmid.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke check CTMID without training.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "all_beauty_smoke.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("train", {}).get("seed", 2026)))
    feature_store = _synthetic_feature_store(n_items=32)
    model = CTMID(config, feature_store)
    batch = {
        "item_seq": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long),
        "time_seq": torch.tensor([[1588687728923, 1588787728923, 1588887728923, 0], [1588687728923, 1588787728923, 0, 0]], dtype=torch.float),
        "seq_len": torch.tensor([3, 2], dtype=torch.long),
        "target_item": torch.tensor([4, 6], dtype=torch.long),
    }
    model.train()
    loss = model.calculate_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"loss is not finite: {loss}")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise RuntimeError("expected at least one gradient from sampled-softmax loss")

    model.eval()
    with torch.no_grad():
        eval_loss = model.calculate_loss(batch)
        point_scores = model.predict(batch, torch.tensor([4, 6], dtype=torch.long))
        scores = model.full_sort_predict(batch)
    if not torch.isfinite(eval_loss):
        raise RuntimeError(f"eval loss is not finite: {eval_loss}")
    if point_scores.shape != (2,):
        raise RuntimeError(f"unexpected point score shape: {tuple(point_scores.shape)}")
    if scores.shape != (2, feature_store.n_items):
        raise RuntimeError(f"unexpected score shape: {tuple(scores.shape)}")
    metrics = ranking_metrics(scores, batch["target_item"], [5, 10])
    print(f"loss={float(loss.item()):.6f}")
    print(f"eval_loss={float(eval_loss.item()):.6f}")
    print(f"point_scores_shape={tuple(point_scores.shape)}")
    print(f"scores_shape={tuple(scores.shape)}")
    print(f"metrics={metrics}")
    print("Smoke check passed. No training loop was executed.")


def _synthetic_feature_store(n_items: int) -> ItemFeatureStore:
    category_ids = torch.arange(n_items) % 4
    price_value = torch.randn(n_items, 1)
    price_missing = torch.zeros(n_items, 1)
    rating_stats = torch.randn(n_items, 4)
    category_ids[0] = 0
    price_value[0] = 0
    rating_stats[0] = 0
    return ItemFeatureStore(
        category_ids=category_ids.long(),
        price_value=price_value.float(),
        price_missing=price_missing.float(),
        rating_stats=rating_stats.float(),
    )


if __name__ == "__main__":
    main()
