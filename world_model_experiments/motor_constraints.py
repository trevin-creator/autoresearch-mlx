from __future__ import annotations

import torch


def apply_motor_constraints(
    actions: torch.Tensor,
    low: float = -1.0,
    high: float = 1.0,
    max_delta: float | None = None,
) -> torch.Tensor:
    """Clamp actions and optionally enforce slew-rate limits along time axis.

    actions: [B, T, A] or [T, A]
    """

    x = actions
    if x.dim() == 2:
        x = x.unsqueeze(0)

    x = x.clamp(low, high)

    if max_delta is not None and max_delta > 0.0 and x.shape[1] > 1:
        out = [x[:, 0:1, :]]
        prev = x[:, 0:1, :]
        for t in range(1, x.shape[1]):
            cur = x[:, t : t + 1, :]
            d = (cur - prev).clamp(-max_delta, max_delta)
            nxt = (prev + d).clamp(low, high)
            out.append(nxt)
            prev = nxt
        x = torch.cat(out, dim=1)

    return x if actions.dim() == 3 else x.squeeze(0)
