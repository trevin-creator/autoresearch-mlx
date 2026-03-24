from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model_experiments.lewm_feature_model import FeatureJEPA
from world_model_experiments.motor_constraints import apply_motor_constraints


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 16
    candidates: int = 256
    elites: int = 32
    iters: int = 5
    action_low: float = -1.0
    action_high: float = 1.0
    max_action_delta: float | None = None
    action_slew_penalty: float = 0.0
    action_energy_penalty: float = 0.0
    rollout_norm_limit: float | None = None
    invalid_cost: float = 1e6


@torch.no_grad()
def plan_actions_cem(
    model: FeatureJEPA,
    emb_history: torch.Tensor,
    goal_embedding: torch.Tensor,
    action_dim: int,
    cfg: CEMConfig,
) -> torch.Tensor:
    """Dreamer-like latent planning using CEM over imagined embeddings.

    emb_history: [1, H, D]
    goal_embedding: [1, D]
    returns: best action sequence [1, horizon, action_dim]
    """

    device = emb_history.device
    mean = torch.zeros((1, cfg.horizon, action_dim), device=device)
    std = torch.full((1, cfg.horizon, action_dim), 0.5, device=device)

    for _ in range(cfg.iters):
        noise = torch.randn((cfg.candidates, cfg.horizon, action_dim), device=device)
        actions = mean + std * noise
        actions = apply_motor_constraints(
            actions,
            low=cfg.action_low,
            high=cfg.action_high,
            max_delta=cfg.max_action_delta,
        )

        hist = emb_history.expand(cfg.candidates, -1, -1)
        rollout = model.rollout_embeddings(hist, actions)
        final = rollout[:, -1]
        cost = torch.sum((final - goal_embedding.expand_as(final)) ** 2, dim=-1)

        if cfg.action_slew_penalty > 0.0 and cfg.horizon > 1:
            slew = torch.mean(torch.abs(actions[:, 1:] - actions[:, :-1]), dim=(1, 2))
            cost = cost + cfg.action_slew_penalty * slew

        if cfg.action_energy_penalty > 0.0:
            energy = torch.mean(actions * actions, dim=(1, 2))
            cost = cost + cfg.action_energy_penalty * energy

        if cfg.rollout_norm_limit is not None and cfg.rollout_norm_limit > 0.0:
            rollout_norm = torch.mean(torch.linalg.norm(rollout, dim=-1), dim=-1)
            invalid = rollout_norm > cfg.rollout_norm_limit
            cost = torch.where(invalid, torch.full_like(cost, cfg.invalid_cost), cost)

        elite_idx = torch.topk(cost, k=cfg.elites, largest=False).indices
        elite = actions[elite_idx]

        mean = elite.mean(dim=0, keepdim=True)
        std = elite.std(dim=0, keepdim=True).clamp_min(1e-3)

    return apply_motor_constraints(
        mean,
        low=cfg.action_low,
        high=cfg.action_high,
        max_delta=cfg.max_action_delta,
    )
