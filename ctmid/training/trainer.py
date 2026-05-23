from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctmid.training.evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.train_cfg = config.get("train", {})
        self.eval_cfg = config.get("eval", {})
        self.device = torch.device(self.train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.grad_clip_norm = float(self.train_cfg.get("grad_clip_norm", 5.0))
        self.amp = bool(self.train_cfg.get("amp", False))
        self.evaluator = Evaluator(
            topk=list(self.eval_cfg.get("topk", [10])),
            mask_history=bool(self.eval_cfg.get("mask_history", True)),
            item_chunk_size=self.eval_cfg.get("item_chunk_size"),
            amp=self.amp,
        )

    def fit(self) -> dict[str, float]:
        self.model.to(self.device)
        best_metrics: dict[str, float] = {}
        best_score = float("-inf")
        valid_metric = str(self.eval_cfg.get("valid_metric", "NDCG@10"))
        patience = int(self.train_cfg.get("stopping_step", 10))
        bad_epochs = 0
        epochs = int(self.train_cfg.get("epochs", 200))
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            metrics = self.evaluator.evaluate(self.model, self.valid_loader, self.device)
            score = metrics.get(valid_metric, float("-inf"))
            print(f"epoch={epoch} train_loss={train_loss:.6f} valid={metrics}")
            if score > best_score:
                best_score = score
                best_metrics = metrics
                bad_epochs = 0
                self._save_checkpoint(epoch, metrics)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        return best_metrics

    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_count = 0
        use_amp = self.amp and self.device.type == "cuda"
        for batch in tqdm(self.train_loader, desc=f"train {epoch}", leave=False):
            batch = _move_batch(batch, self.device)
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = self.model.calculate_loss(batch)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            count = int(batch["target_item"].numel())
            total_loss += float(loss.item()) * count
            total_count += count
        return total_loss / max(total_count, 1)

    def _save_checkpoint(self, epoch: int, metrics: dict[str, float]) -> None:
        checkpoint_dir = Path(self.train_cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.config,
            },
            checkpoint_dir / "best.pt",
        )


def _move_batch(batch: dict, device: torch.device | str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
