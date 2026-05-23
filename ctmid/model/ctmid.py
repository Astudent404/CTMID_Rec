from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ctmid.data.feature_store import ItemFeatureStore
from ctmid.model.cross_modal_fusion import TimeAlignedFusion
from ctmid.model.feature_encoder import ModalityFeatureEncoder
from ctmid.model.losses import CTMIDLoss
from ctmid.model.modality_ode import DualODELayer
from ctmid.model.time_encoding import MultiGranularityTimeEncoder


class CTMID(nn.Module):
    def __init__(self, config: dict[str, Any], feature_store: ItemFeatureStore) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        self.n_items = feature_store.n_items
        self.hidden_size = int(model_cfg.get("hidden_size", model_cfg.get("embedding_size", 64)))
        self.modalities = list(model_cfg.get("modalities", ["id", "category", "price", "rating"]))
        self.dropout_prob = float(model_cfg.get("dropout_prob", 0.2))
        self.pad_item_id = 0
        # Sampled-softmax training: >0 computes loss over (batch targets + N
        # sampled negatives) instead of the full catalogue. 0 = full softmax.
        self.num_negatives = int(config.get("train", {}).get("num_negatives", 0) or 0)
        self.sequence_encoder_type = str(model_cfg.get("sequence_encoder", "none")).lower()
        self.use_sequence_encoder = self.sequence_encoder_type not in {"", "none", "false"}
        self.id_residual_weight = float(model_cfg.get("id_residual_weight", 0.0))
        self.use_id_residual = self.id_residual_weight != 0.0 and "id" in self.modalities
        self.last_item_residual_weight = float(model_cfg.get("last_item_residual_weight", 0.0))
        self.use_last_item_residual = self.last_item_residual_weight != 0.0

        self.feature_encoder = ModalityFeatureEncoder(feature_store, self.modalities, self.hidden_size, self.dropout_prob)
        if self.use_sequence_encoder:
            if self.sequence_encoder_type != "gru":
                raise ValueError(f"Unsupported sequence_encoder: {self.sequence_encoder_type}")
            self.sequence_encoders = nn.ModuleDict(
                {modality: nn.GRU(self.hidden_size, self.hidden_size, batch_first=True) for modality in self.modalities}
            )
            self.sequence_encoder_norms = nn.ModuleDict(
                {modality: nn.LayerNorm(self.hidden_size) for modality in self.modalities}
            )
            self.sequence_encoder_dropout = nn.Dropout(self.dropout_prob)
        self.time_encoder = MultiGranularityTimeEncoder(
            self.hidden_size,
            self.modalities,
            dropout_prob=self.dropout_prob,
            adaptive_weights=bool(model_cfg.get("use_modality_adaptive_time_weight", True)),
            enabled=bool(model_cfg.get("use_multi_granularity_time", True)),
        )
        self.dual_ode = DualODELayer(
            self.hidden_size,
            self.modalities,
            dropout_prob=self.dropout_prob,
            solver=str(model_cfg.get("ode_solver", "rk4")),
            steps=int(model_cfg.get("ode_steps", 2)),
            modality_specific=bool(model_cfg.get("use_modality_specific_ode", True)),
            use_user_ode=bool(model_cfg.get("use_user_ode", True)),
            use_item_ode=bool(model_cfg.get("use_item_ode", True)),
            exp_decay=bool(model_cfg.get("use_exp_decay", False)),
        )
        self.fusion = TimeAlignedFusion(
            self.hidden_size,
            self.modalities,
            num_heads=int(model_cfg.get("fusion_num_heads", 2)),
            dropout_prob=self.dropout_prob,
            enabled=bool(model_cfg.get("use_time_aligned_fusion", True)),
        )
        self.sequence_norm = nn.LayerNorm(self.hidden_size)
        self.item_norm = nn.LayerNorm(self.hidden_size)
        self.id_residual_norm = nn.LayerNorm(self.hidden_size)
        self.output_bias = nn.Parameter(torch.zeros(self.n_items))
        self.loss_module = CTMIDLoss(
            lambda_tempo=float(model_cfg.get("lambda_tempo", 0.05)),
            lambda_contrast=float(model_cfg.get("lambda_contrast", 0.0)),
            lambda_cross=float(model_cfg.get("lambda_cross", 0.0)),
            lambda_rank=float(model_cfg.get("lambda_rank", 0.0)),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.zeros_(parameter)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        item_seq = batch["item_seq"].long()
        item_seq_len = batch["seq_len"].long()
        time_seq = batch.get("time_seq")
        modality_sequence = self.feature_encoder(item_seq)
        mask = (item_seq != self.pad_item_id).float().unsqueeze(-1)
        encoded_sequences: dict[str, torch.Tensor] = {}
        for modality, sequence in modality_sequence.items():
            encoded = sequence
            time_embedding = self.time_encoder(time_seq, modality)
            if time_embedding is not None:
                encoded = encoded + time_embedding
            encoded_sequences[modality] = self._encode_modality_sequence(modality, encoded)
        context_sequences = encoded_sequences if self.use_sequence_encoder else modality_sequence
        modality_states: dict[str, torch.Tensor] = {}
        delta_t = self._target_delta_time(time_seq, item_seq_len)
        for modality, encoded in encoded_sequences.items():
            pooled = self._last_valid_state(encoded, item_seq_len)
            context = self._mean_context(context_sequences, exclude=modality, mask=mask)
            context = self._last_valid_state(context, item_seq_len)
            modality_states[modality] = self.dual_ode.evolve_user(modality, pooled, delta_t=delta_t, context=context)
        return self.sequence_norm(self.fusion(modality_states)), modality_states

    def calculate_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_output, modality_states = self.forward(batch)
        target = batch["target_item"].long()
        if self.training and self.num_negatives > 0 and self.n_items > 2:
            num_neg = min(self.num_negatives, self.n_items - 1)
            negatives = torch.randint(1, self.n_items, (num_neg,), device=seq_output.device)
            candidate_ids = torch.cat([target, negatives], dim=0)
            candidate_repr = self._item_representations(candidate_ids)
            logits = torch.matmul(seq_output, candidate_repr.transpose(0, 1)) + self.output_bias[candidate_ids]
            logits = logits + self._id_residual_logits(modality_states, candidate_ids)
            logits = logits + self._last_item_residual_logits(batch, candidate_ids)
            labels = torch.arange(target.size(0), device=seq_output.device)
            rank_negative_start = target.size(0)
        else:
            logits = torch.matmul(seq_output, self.compute_item_representations().transpose(0, 1)) + self.output_bias
            logits = logits + self._id_residual_logits(modality_states)
            logits = logits + self._last_item_residual_logits(batch)
            labels = target
            rank_negative_start = None
        loss, _ = self.loss_module(
            logits,
            labels,
            self.dual_ode.derivative_norms(),
            modality_states,
            rank_negative_start=rank_negative_start,
        )
        return loss

    def predict(self, batch: dict[str, torch.Tensor], item_ids: torch.Tensor) -> torch.Tensor:
        seq_output, modality_states = self.forward(batch)
        item_repr = self._item_representations(item_ids.long())
        scores = (seq_output * item_repr).sum(dim=-1) + self.output_bias[item_ids.long()]
        scores = scores + self._id_residual_logits(modality_states, item_ids.long())
        return scores + self._last_item_residual_logits(batch, item_ids.long())

    def full_sort_predict(
        self,
        batch: dict[str, torch.Tensor],
        item_chunk_size: int | None = None,
        item_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_output, modality_states = self.forward(batch)
        if item_repr is None:
            item_repr = self.compute_item_representations(chunk_size=item_chunk_size)
        scores = torch.matmul(seq_output, item_repr.transpose(0, 1)) + self.output_bias
        scores = scores + self._id_residual_logits(modality_states)
        scores = scores + self._last_item_residual_logits(batch)
        scores[:, self.pad_item_id] = -torch.inf
        return scores

    def modality_speed_table(self) -> dict[str, float]:
        with torch.no_grad():
            return {name: float(value.detach().cpu().item()) for name, value in self.dual_ode.derivative_norms().items()}

    def time_weight_table(self) -> dict[str, dict[str, float]]:
        return self.time_encoder.modality_weight_table()

    def compute_item_representations(self, chunk_size: int | None = None) -> torch.Tensor:
        """Item representations for the whole catalogue, computed in chunks to
        bound memory. During evaluation this is computed once and reused."""
        chunk = int(chunk_size) if chunk_size else 32768
        device = self.output_bias.device
        chunks = [
            self._item_representations(torch.arange(start, min(start + chunk, self.n_items), device=device))
            for start in range(0, self.n_items, chunk)
        ]
        return torch.cat(chunks, dim=0)

    def _item_representations(self, item_ids: torch.Tensor) -> torch.Tensor:
        modality_items = self.feature_encoder(item_ids)
        evolved: dict[str, torch.Tensor] = {}
        for modality, state in modality_items.items():
            context = self._item_context(modality_items, exclude=modality)
            evolved[modality] = self.dual_ode.evolve_item(modality, state, delta_t=1.0, context=context)
        return self.item_norm(self.fusion(evolved))

    def _id_item_representations(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = item_ids.long().clamp(min=0, max=self.n_items - 1)
        return self.id_residual_norm(self.feature_encoder.item_id_embedding(item_ids))

    def _id_residual_logits(
        self,
        modality_states: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_id_residual:
            return next(iter(modality_states.values())).new_zeros(())
        id_user = self.id_residual_norm(modality_states["id"])
        if item_ids is None:
            item_ids = torch.arange(self.n_items, device=id_user.device)
        id_items = self._id_item_representations(item_ids)
        if item_ids.dim() == 1 and item_ids.size(0) == id_user.size(0):
            return self.id_residual_weight * (id_user * id_items).sum(dim=-1)
        if item_ids.dim() == 1:
            return self.id_residual_weight * torch.matmul(id_user, id_items.transpose(0, 1))
        return self.id_residual_weight * (id_user * id_items).sum(dim=-1)

    def _last_item_residual_logits(
        self,
        batch: dict[str, torch.Tensor],
        item_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_last_item_residual:
            return batch["item_seq"].new_zeros((), dtype=self.output_bias.dtype)
        last_ids = self._last_item_ids(batch)
        last_items = self._id_item_representations(last_ids)
        if item_ids is None:
            item_ids = torch.arange(self.n_items, device=last_items.device)
        id_items = self._id_item_representations(item_ids)
        if item_ids.dim() == 1 and item_ids.size(0) == last_items.size(0):
            return self.last_item_residual_weight * (last_items * id_items).sum(dim=-1)
        if item_ids.dim() == 1:
            return self.last_item_residual_weight * torch.matmul(last_items, id_items.transpose(0, 1))
        return self.last_item_residual_weight * (last_items * id_items).sum(dim=-1)

    def _last_item_ids(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        item_seq = batch["item_seq"].long()
        item_seq_len = batch["seq_len"].long()
        index = (item_seq_len - 1).clamp(min=0, max=item_seq.size(1) - 1)
        return item_seq.gather(dim=1, index=index.view(-1, 1)).squeeze(1)

    def _encode_modality_sequence(self, modality: str, sequence: torch.Tensor) -> torch.Tensor:
        if not self.use_sequence_encoder:
            return sequence
        encoded, _ = self.sequence_encoders[modality](sequence)
        encoded = self.sequence_encoder_dropout(encoded)
        return self.sequence_encoder_norms[modality](sequence + encoded)

    def _last_valid_state(self, sequence: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
        index = (item_seq_len - 1).clamp(min=0, max=sequence.size(1) - 1)
        gather_index = index.view(-1, 1, 1).expand(-1, 1, sequence.size(-1))
        return sequence.gather(dim=1, index=gather_index).squeeze(1)

    def _mean_context(self, modality_sequence: dict[str, torch.Tensor], exclude: str, mask: torch.Tensor) -> torch.Tensor:
        others = [value for key, value in modality_sequence.items() if key != exclude]
        if not others:
            return torch.zeros_like(next(iter(modality_sequence.values())))
        stacked = torch.stack(others, dim=0).mean(dim=0)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        sequence_mean = (stacked * mask).sum(dim=1, keepdim=True) / denom
        return stacked + sequence_mean

    def _item_context(self, modality_items: dict[str, torch.Tensor], exclude: str) -> torch.Tensor:
        others = [value for key, value in modality_items.items() if key != exclude]
        if not others:
            return torch.zeros_like(next(iter(modality_items.values())))
        return torch.stack(others, dim=0).mean(dim=0)

    def _target_delta_time(self, time_seq: torch.Tensor | None, item_seq_len: torch.Tensor) -> torch.Tensor | None:
        if time_seq is None or time_seq.dim() < 2:
            return None
        batch = torch.arange(time_seq.size(0), device=time_seq.device)
        last_idx = (item_seq_len - 1).clamp(min=0, max=time_seq.size(1) - 1)
        prev_idx = (last_idx - 1).clamp(min=0, max=time_seq.size(1) - 1)
        last_time = time_seq[batch, last_idx].float()
        prev_time = time_seq[batch, prev_idx].float()
        scale = 1000.0 if torch.nan_to_num(time_seq.float(), nan=0.0).abs().max() > 10_000_000_000 else 1.0
        return torch.relu(last_time - prev_time) / scale / 86400.0
