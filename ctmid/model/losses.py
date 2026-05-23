from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTMIDLoss(nn.Module):
    def __init__(
        self,
        lambda_tempo: float = 0.05,
        lambda_contrast: float = 0.0,
        lambda_cross: float = 0.0,
        lambda_rank: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_tempo = float(lambda_tempo)
        self.lambda_contrast = float(lambda_contrast)
        self.lambda_cross = float(lambda_cross)
        self.lambda_rank = float(lambda_rank)
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        derivative_norms: dict[str, torch.Tensor] | None = None,
        modality_states: dict[str, torch.Tensor] | None = None,
        rank_negative_start: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec_loss = self.ce(logits, target.long())
        rank_loss = self._pairwise_rank_loss(logits, target, rank_negative_start)
        tempo_loss = self._tempo_diversity_loss(derivative_norms, logits)
        contrast_loss = self._modality_contrast_loss(modality_states, logits)
        cross_loss = self._cross_prediction_proxy_loss(modality_states, logits)
        total = (
            rec_loss
            + self.lambda_rank * rank_loss
            + self.lambda_tempo * tempo_loss
            + self.lambda_contrast * contrast_loss
            + self.lambda_cross * cross_loss
        )
        return total, {
            "rec_loss": rec_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "tempo_loss": tempo_loss.detach(),
            "contrast_loss": contrast_loss.detach(),
            "cross_loss": cross_loss.detach(),
        }

    def _pairwise_rank_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        negative_start: int | None,
    ) -> torch.Tensor:
        if self.lambda_rank <= 0:
            return logits.new_zeros(())
        labels = target.long()
        positive = logits.gather(1, labels.view(-1, 1))
        if negative_start is not None:
            negatives = logits[:, int(negative_start) :]
        else:
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, labels.view(-1, 1), False)
            negatives = logits[mask].view(logits.size(0), -1)
        if negatives.numel() == 0:
            return logits.new_zeros(())
        return F.softplus(negatives - positive).mean()

    def _tempo_diversity_loss(self, norms: dict[str, torch.Tensor] | None, reference: torch.Tensor) -> torch.Tensor:
        if not norms or len(norms) < 2 or self.lambda_tempo <= 0:
            return reference.new_zeros(())
        values = torch.stack([value.reshape(()) for value in norms.values()])
        distances = torch.abs(values.unsqueeze(0) - values.unsqueeze(1))
        mask = ~torch.eye(values.numel(), dtype=torch.bool, device=values.device)
        return torch.exp(-distances[mask]).mean()

    def _modality_contrast_loss(self, states: dict[str, torch.Tensor] | None, reference: torch.Tensor) -> torch.Tensor:
        if not states or len(states) < 2 or self.lambda_contrast <= 0:
            return reference.new_zeros(())
        normalized = [F.normalize(state, dim=-1) for state in states.values()]
        losses = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                similarity = (normalized[i] * normalized[j]).sum(dim=-1)
                losses.append((1.0 - similarity).mean())
        return torch.stack(losses).mean() if losses else reference.new_zeros(())

    def _cross_prediction_proxy_loss(self, states: dict[str, torch.Tensor] | None, reference: torch.Tensor) -> torch.Tensor:
        if not states or len(states) < 2 or self.lambda_cross <= 0:
            return reference.new_zeros(())
        stacked = torch.stack(list(states.values()), dim=0)
        centroid = stacked.mean(dim=0)
        return (stacked - centroid.unsqueeze(0)).pow(2).mean()
