from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ODEFunction(nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if context is None:
            context = torch.zeros_like(state)
        return self.net(torch.cat([state, context], dim=-1))


class ModalityODEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        modalities: Iterable[str],
        dropout_prob: float = 0.1,
        solver: str = "rk4",
        steps: int = 2,
        modality_specific: bool = True,
        exp_decay: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.modalities = list(modalities)
        self.solver = solver.lower()
        self.steps = max(int(steps), 1)
        self.modality_specific = modality_specific
        self.exp_decay = exp_decay
        if modality_specific:
            self.functions = nn.ModuleDict(
                {modality: ODEFunction(hidden_size, dropout_prob) for modality in self.modalities}
            )
            self.decay_rates = nn.ParameterDict(
                {modality: nn.Parameter(torch.tensor(0.1)) for modality in self.modalities}
            )
        else:
            self.shared_function = ODEFunction(hidden_size, dropout_prob)
            self.shared_decay = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        modality: str,
        state: torch.Tensor,
        delta_t: torch.Tensor | float | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dt = self._prepare_delta_t(delta_t, state)
        if self.exp_decay:
            rate = self._decay_rate(modality).to(device=state.device, dtype=state.dtype).abs()
            return state * torch.exp(-rate * dt)
        if self.solver == "euler":
            return self._euler(modality, state, dt, context)
        if self.solver == "rk4":
            return self._rk4(modality, state, dt, context)
        raise ValueError(f"Unsupported ODE solver: {self.solver}")

    def derivative_norms(self) -> dict[str, torch.Tensor]:
        norms: dict[str, torch.Tensor] = {}
        if self.exp_decay:
            if self.modality_specific:
                for modality, value in self.decay_rates.items():
                    norms[modality] = value.abs()
            else:
                for modality in self.modalities:
                    norms[modality] = self.shared_decay.abs()
            return norms
        if self.modality_specific:
            for modality, function in self.functions.items():
                total = torch.zeros((), device=next(function.parameters()).device)
                count = 0
                for param in function.parameters():
                    total = total + param.pow(2).mean()
                    count += 1
                norms[modality] = torch.sqrt(total / max(count, 1))
        else:
            total = torch.zeros((), device=next(self.shared_function.parameters()).device)
            count = 0
            for param in self.shared_function.parameters():
                total = total + param.pow(2).mean()
                count += 1
            shared = torch.sqrt(total / max(count, 1))
            for modality in self.modalities:
                norms[modality] = shared
        return norms

    def _function(self, modality: str) -> ODEFunction:
        if self.modality_specific:
            return self.functions[modality]
        return self.shared_function

    def _decay_rate(self, modality: str) -> torch.Tensor:
        if self.modality_specific:
            return self.decay_rates[modality]
        return self.shared_decay

    def _euler(self, modality: str, state: torch.Tensor, dt: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        step_dt = dt / self.steps
        output = state
        function = self._function(modality)
        for _ in range(self.steps):
            output = output + step_dt * function(output, context)
        return output

    def _rk4(self, modality: str, state: torch.Tensor, dt: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        step_dt = dt / self.steps
        output = state
        function = self._function(modality)
        for _ in range(self.steps):
            k1 = function(output, context)
            k2 = function(output + 0.5 * step_dt * k1, context)
            k3 = function(output + 0.5 * step_dt * k2, context)
            k4 = function(output + step_dt * k3, context)
            output = output + (step_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return output

    def _prepare_delta_t(self, delta_t: torch.Tensor | float | None, state: torch.Tensor) -> torch.Tensor:
        if delta_t is None:
            dt = torch.ones((*state.shape[:-1], 1), device=state.device, dtype=state.dtype)
        elif isinstance(delta_t, torch.Tensor):
            dt = delta_t.to(device=state.device, dtype=state.dtype)
            while dt.dim() < state.dim():
                dt = dt.unsqueeze(-1)
        else:
            dt = torch.full((*state.shape[:-1], 1), float(delta_t), device=state.device, dtype=state.dtype)
        return torch.log1p(torch.relu(dt)).clamp(max=10.0)


class DualODELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        modalities: Iterable[str],
        dropout_prob: float = 0.1,
        solver: str = "rk4",
        steps: int = 2,
        modality_specific: bool = True,
        use_user_ode: bool = True,
        use_item_ode: bool = True,
        exp_decay: bool = False,
    ) -> None:
        super().__init__()
        self.use_user_ode = use_user_ode
        self.use_item_ode = use_item_ode
        self.user_ode = ModalityODEBlock(
            hidden_size,
            modalities,
            dropout_prob=dropout_prob,
            solver=solver,
            steps=steps,
            modality_specific=modality_specific,
            exp_decay=exp_decay,
        )
        self.item_ode = ModalityODEBlock(
            hidden_size,
            modalities,
            dropout_prob=dropout_prob,
            solver=solver,
            steps=steps,
            modality_specific=modality_specific,
            exp_decay=exp_decay,
        )

    def evolve_user(
        self,
        modality: str,
        state: torch.Tensor,
        delta_t: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_user_ode:
            return state
        return self.user_ode(modality, state, delta_t, context)

    def evolve_item(
        self,
        modality: str,
        state: torch.Tensor,
        delta_t: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_item_ode:
            return state
        return self.item_ode(modality, state, delta_t, context)

    def derivative_norms(self) -> dict[str, torch.Tensor]:
        user = self.user_ode.derivative_norms()
        item = self.item_ode.derivative_norms()
        return {modality: user[modality] + item[modality] for modality in user}
