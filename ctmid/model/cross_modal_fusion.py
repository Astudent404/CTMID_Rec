from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class TimeAlignedFusion(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        modalities: Iterable[str],
        num_heads: int = 2,
        dropout_prob: float = 0.1,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.modalities = list(modalities)
        self.enabled = enabled
        if enabled:
            self.shared_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout_prob,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.output = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
            )
        self.fallback_gate = nn.Linear(hidden_size * len(self.modalities), len(self.modalities))

    def forward(self, modality_states: dict[str, torch.Tensor]) -> torch.Tensor:
        states = [modality_states[name] for name in self.modalities if name in modality_states]
        if not states:
            raise ValueError("TimeAlignedFusion received no modality states")
        stacked = torch.stack(states, dim=1)
        if not self.enabled:
            weights = torch.softmax(self.fallback_gate(stacked.flatten(start_dim=1)), dim=-1)
            return (stacked * weights.unsqueeze(-1)).sum(dim=1)
        batch_size = stacked.size(0)
        shared = self.shared_token.expand(batch_size, -1, -1)
        sequence = torch.cat([shared, stacked], dim=1)
        # The flash/efficient SDPA kernels reject batch dims over 65535, which
        # is exceeded when fusing representations for a large item catalogue.
        # The math backend has no such limit and is appropriate here since the
        # modality sequence is only a few tokens long.
        with sdpa_kernel(SDPBackend.MATH):
            attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        fused = self.norm(attended[:, 0, :] + shared.squeeze(1))
        return self.output(fused)
