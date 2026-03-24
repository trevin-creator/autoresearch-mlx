from __future__ import annotations

from dataclasses import dataclass

import torch

from world_model_experiments.lewm_feature_model import FeatureJEPA


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 16
    candidates: int = 256
    elites: int = 32
    iters: int = 5
    action_low: float = -1.0
    action_high: float = 1.0


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
        actions = actions.clamp(cfg.action_low, cfg.action_high)

        hist = emb_history.expand(cfg.candidates, -1, -1)
        rollout = model.rollout_embeddings(hist, actions)
        final = rollout[:, -1]
        cost = torch.sum((final - goal_embedding.expand_as(final)) ** 2, dim=-1)

        elite_idx = torch.topk(cost, k=cfg.elites, largest=False).indices
        elite = actions[elite_idx]

        mean = elite.mean(dim=0, keepdim=True)
        std = elite.std(dim=0, keepdim=True).clamp_min(1e-3)

    return mean
