from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class InformedDreamerConfig:
    feature_dim: int
    action_dim: int
    pose_dim: int = 6
    embed_dim: int = 128
    hidden_dim: int = 192
    horizon: int = 8
    discount: float = 0.99
    lambda_return: float = 0.95
    wm_latent_weight: float = 1.0
    wm_pose_weight: float = 1.0
    wm_pose_delta_weight: float = 1.0
    wm_reward_weight: float = 1.0
    wm_continue_weight: float = 1.0
    smooth_weight: float = 2e-3


class GaussianRegularizer(nn.Module):
    def __init__(self, num_proj: int = 256):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        b, t, d = emb.shape
        x = emb.reshape(b * t, d)
        proj = torch.randn(d, self.num_proj, device=emb.device, dtype=emb.dtype)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-6)
        y = x @ proj
        return y.mean(dim=0).pow(2).mean() + (y.std(dim=0) - 1.0).pow(2).mean()


class InformedFeatureDreamer(nn.Module):
    """Informed Dreamer-style latent model for SNN features.

    - World model learns latent dynamics + privileged decoders (pose, pose_delta).
    - Reward/continue heads support imagined actor-critic learning.
    - Actor smoothness regularization is applied on action means only.
    """

    def __init__(self, cfg: InformedDreamerConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.embed_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, cfg.embed_dim),
            nn.SiLU(),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
        )

        self.rssm = nn.GRU(input_size=2 * cfg.embed_dim, hidden_size=cfg.hidden_dim, batch_first=True)
        self.prior = nn.Linear(cfg.hidden_dim, cfg.embed_dim)

        # Informed decoder heads (SkyDreamer-style privileged targets)
        self.pose_head = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.pose_dim))
        self.pose_delta_head = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, cfg.pose_dim))
        self.reward_head = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 1))
        self.continue_head = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 1))

        # Actor-critic on latent state
        self.actor = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 2 * cfg.action_dim))
        self.critic = nn.Sequential(nn.Linear(cfg.hidden_dim + cfg.embed_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 1))

        self.reg = GaussianRegularizer()

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def action_emb(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(actions)

    def world_forward(self, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        z_post = self.encode(features)
        a_emb = self.action_emb(actions)
        x = torch.cat([z_post, a_emb], dim=-1)
        h_seq, _ = self.rssm(x)
        z_prior = self.prior(h_seq)
        state = torch.cat([h_seq, z_post], dim=-1)

        return {
            "z_post": z_post,
            "z_prior": z_prior,
            "h": h_seq,
            "state": state,
            "pose": self.pose_head(state),
            "pose_delta": self.pose_delta_head(state),
            "reward": self.reward_head(state).squeeze(-1),
            "continue_logit": self.continue_head(state).squeeze(-1),
        }

    def world_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        pose_tgt: torch.Tensor,
        pose_delta_tgt: torch.Tensor,
        reward_tgt: torch.Tensor,
        continue_tgt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out = self.world_forward(features, actions)

        latent_loss = F.mse_loss(out["z_prior"][:, :-1], out["z_post"][:, 1:])
        pose_loss = F.mse_loss(out["pose"], pose_tgt)
        pose_delta_loss = F.mse_loss(out["pose_delta"], pose_delta_tgt)
        reward_loss = F.mse_loss(out["reward"], reward_tgt)
        continue_loss = F.binary_cross_entropy_with_logits(out["continue_logit"], continue_tgt)
        sigreg = self.reg(out["z_post"])

        total = (
            self.cfg.wm_latent_weight * latent_loss
            + self.cfg.wm_pose_weight * pose_loss
            + self.cfg.wm_pose_delta_weight * pose_delta_loss
            + self.cfg.wm_reward_weight * reward_loss
            + self.cfg.wm_continue_weight * continue_loss
            + 0.1 * sigreg
        )

        return {
            "loss": total,
            "latent_loss": latent_loss,
            "pose_loss": pose_loss,
            "pose_delta_loss": pose_delta_loss,
            "reward_loss": reward_loss,
            "continue_loss": continue_loss,
            "sigreg_loss": sigreg,
            **out,
        }

    def actor_critic_loss(self, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            wm = self.world_forward(features, actions)
            h0 = wm["h"][:, -1]
            z0 = wm["z_post"][:, -1]

        h = h0
        z = z0

        rewards = []
        conts = []
        values = []
        mus = []

        for _ in range(self.cfg.horizon):
            s = torch.cat([h, z], dim=-1)
            actor_out = self.actor(s)
            mu, log_std = torch.chunk(actor_out, 2, dim=-1)
            log_std = torch.clamp(log_std, -5.0, 1.0)
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            action = torch.tanh(mu + eps * std)

            a_emb = self.action_emb(action.unsqueeze(1)).squeeze(1)
            rssm_in = torch.cat([z, a_emb], dim=-1).unsqueeze(1)
            h_next, _ = self.rssm(rssm_in, h.unsqueeze(0))
            h = h_next.squeeze(1)
            z = self.prior(h)

            s_next = torch.cat([h, z], dim=-1)
            rewards.append(self.reward_head(s_next).squeeze(-1))
            conts.append(torch.sigmoid(self.continue_head(s_next).squeeze(-1)))
            values.append(self.critic(s_next).squeeze(-1))
            mus.append(mu)

        rewards_t = torch.stack(rewards, dim=1)
        conts_t = torch.stack(conts, dim=1)
        values_t = torch.stack(values, dim=1)

        with torch.no_grad():
            target_list = []
            g = values_t[:, -1]
            for t in reversed(range(self.cfg.horizon)):
                bootstrap = (1.0 - self.cfg.lambda_return) * values_t[:, t] + self.cfg.lambda_return * g
                g = rewards_t[:, t] + self.cfg.discount * conts_t[:, t] * bootstrap
                target_list.append(g)
            targets = torch.stack(list(reversed(target_list)), dim=1)

        actor_loss = -targets.mean()
        critic_loss = F.mse_loss(values_t, targets)

        if len(mus) > 1:
            mu_t = torch.stack(mus, dim=1)
            smooth = torch.mean((mu_t[:, 1:] - mu_t[:, :-1]) ** 2)
        else:
            smooth = torch.zeros((), device=features.device)

        total_actor = actor_loss + self.cfg.smooth_weight * smooth
        total = total_actor + critic_loss

        return {
            "loss": total,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "smooth_loss": smooth,
            "total_actor_loss": total_actor,
        }
