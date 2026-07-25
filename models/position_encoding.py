"""2D sine positional encoding (DETR-style)."""
from __future__ import annotations

import math

import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, b: int, h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
        """Return [b, 2*num_pos_feats, h, w] positional encoding (no padding mask; square inputs)."""
        y_embed = torch.arange(1, h + 1, dtype=dtype, device=device).view(1, h, 1).repeat(b, 1, w)
        x_embed = torch.arange(1, w + 1, dtype=dtype, device=device).view(1, 1, w).repeat(b, h, 1)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (h + eps) * self.scale
            x_embed = x_embed / (w + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=dtype, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # b, 2*npf, h, w
        return pos
