from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureLeWmConfig:
    feature_dim: int
    action_dim: int
    embed_dim: int = 256
    hidden_dim: int = 384
    history_size: int = 8
    num_preds: int = 1
    depth: int = 4
    heads: int = 8
    dropout: float = 0.0
    sigreg_weight: float = 10.0


class GaussianRegularizer(nn.Module):
    """Lightweight isotropic Gaussian regularizer inspired by SIGReg."""

    def __init__(self, num_proj: int = 256):
        super().__init__()
        self.num_proj = num_proj

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: [B, T, D]
        b, t, d = emb.shape
        x = emb.reshape(b * t, d)
        proj = torch.randn(d, self.num_proj, device=emb.device, dtype=emb.dtype)
        proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-6)
        y = x @ proj
        mean_err = y.mean(dim=0).pow(2).mean()
        std_err = (y.std(dim=0) - 1.0).pow(2).mean()
        return mean_err + std_err


class FeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class ARPredictor(nn.Module):
    """Autoregressive predictor conditioned on action embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int, depth: int, heads: int, dropout: float):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, num_layers=depth)
        self.cond = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        # emb, act_emb: [B, T, D]
        t = emb.shape[1]
        x = emb + self.pos_emb[:, :t] + self.cond(act_emb)
        # Causal mask so prediction remains autoregressive.
        mask = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)
        h = self.tr(x, mask=mask)
        return self.out(h)


class FeatureJEPA(nn.Module):
    """LeWM-style JEPA for feature vectors instead of RGB pixels."""

    def __init__(self, cfg: FeatureLeWmConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = FeatureEncoder(cfg.feature_dim, cfg.embed_dim, cfg.hidden_dim)
        self.action_encoder = ActionEncoder(cfg.action_dim, cfg.embed_dim)
        self.predictor = ARPredictor(cfg.embed_dim, cfg.hidden_dim, cfg.depth, cfg.heads, cfg.dropout)
        self.reg = GaussianRegularizer()

    def encode(self, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encoder(features)
        act_emb = self.action_encoder(actions)
        return {"emb": emb, "act_emb": act_emb}

    def predict(self, emb_ctx: torch.Tensor, act_ctx: torch.Tensor) -> torch.Tensor:
        return self.predictor(emb_ctx, act_ctx)

    def compute_loss(self, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encode(features, actions)
        emb = out["emb"]
        act_emb = out["act_emb"]

        t_total = emb.shape[1]
        if t_total <= self.cfg.num_preds:
            raise ValueError("Sequence too short for configured num_preds")

        ctx_len = min(self.cfg.history_size, t_total - self.cfg.num_preds)
        emb_ctx = emb[:, :ctx_len]
        act_ctx = act_emb[:, :ctx_len]
        tgt = emb[:, self.cfg.num_preds : self.cfg.num_preds + ctx_len]

        pred = self.predict(emb_ctx, act_ctx)

        pred_loss = F.mse_loss(pred, tgt)
        sigreg_loss = self.reg(emb)
        total = pred_loss + self.cfg.sigreg_weight * sigreg_loss

        return {
            "loss": total,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "pred": pred,
            "target": tgt,
            "emb": emb,
        }

    @torch.no_grad()
    def rollout_embeddings(self, emb_hist: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        """Dreamer-like imagined rollout in embedding space.

        emb_hist: [B, H, D]
        action_seq: [B, K, A]
        returns: [B, K, D]
        """

        hist = emb_hist
        preds = []
        for k in range(action_seq.shape[1]):
            act_k = action_seq[:, k : k + 1]
            act_hist = self.action_encoder(act_k)
            emb_next = self.predict(hist[:, -1:], act_hist)[:, -1:]
            preds.append(emb_next)
            hist = torch.cat([hist, emb_next], dim=1)
            if hist.shape[1] > self.cfg.history_size:
                hist = hist[:, -self.cfg.history_size :]
        return torch.cat(preds, dim=1)


class FeatureJepaOnnxWrapper(nn.Module):
    """ONNX export wrapper: emits predicted next embeddings from context."""

    def __init__(self, model: FeatureJEPA):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        encoded = self.model.encode(features, actions)
        return self.model.predict(encoded["emb"], encoded["act_emb"])
