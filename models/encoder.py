"""Hybrid encoder (RT-DETR / RF-DETR inspired): AIFI + CCFM.

AIFI  - Attention-based Intra-scale Feature Interaction: run transformer self-attention
        ONLY on the smallest map (P5), where the token count is small and global context
        matters most. Cheap enough for a T4.
CCFM  - CNN-based Cross-scale Feature Fusion: a light PAN (top-down + bottom-up) that mixes
        P3/P4/P5 with convolutions.

Inputs/outputs are lists of [B, C, H, W] maps already projected to hidden_dim.
The AIFI attention weights are cached on `self.last_attn` for the visualization suite.
"""
from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from .position_encoding import PositionEmbeddingSine


class ConvNormAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FusionBlock(nn.Module):
    """Concat two same-size maps -> conv back to hidden_dim (light RepBlock stand-in)."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(ConvNormAct(2 * dim, dim, 1), ConvNormAct(dim, dim, 3))

    def forward(self, a, b):
        return self.block(torch.cat([a, b], dim=1))


class AIFI(nn.Module):
    def __init__(self, dim, nheads, ffn_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.pos = PositionEmbeddingSine(dim // 2, normalize=True)
        self.layers = nn.ModuleList([_EncLayer(dim, nheads, ffn_dim, dropout)
                                     for _ in range(num_layers)])
        self.last_attn = None  # [B, H*W, H*W] from the final layer (for viz)

    def forward(self, x):
        B, C, H, W = x.shape
        pos = self.pos(B, H, W, x.device, x.dtype).flatten(2).transpose(1, 2)  # B, HW, C
        src = x.flatten(2).transpose(1, 2)  # B, HW, C
        attn = None
        for lyr in self.layers:
            src, attn = lyr(src, pos)
        self.last_attn = attn.detach() if attn is not None else None
        return src.transpose(1, 2).reshape(B, C, H, W)


class _EncLayer(nn.Module):
    """Transformer encoder layer that adds sine pos to q/k and returns attn weights."""

    def __init__(self, dim, nheads, ffn_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, src, pos):
        q = k = src + pos
        out, attn = self.attn(q, k, value=src, need_weights=True, average_attn_weights=True)
        src = self.ln1(src + self.drop(out))
        src = self.ln2(src + self.drop(self.ffn(src)))
        return src, attn


class HybridEncoder(nn.Module):
    def __init__(self, dim, nheads, ffn_dim, enc_layers=1, num_levels=3, dropout=0.0):
        super().__init__()
        assert num_levels == 3, "CCFM here assumes P3/P4/P5"
        self.aifi = AIFI(dim, nheads, ffn_dim, num_layers=enc_layers, dropout=dropout)
        # top-down
        self.td_p4 = FusionBlock(dim)
        self.td_p3 = FusionBlock(dim)
        # bottom-up
        self.down3 = ConvNormAct(dim, dim, 3, s=2)
        self.down4 = ConvNormAct(dim, dim, 3, s=2)
        self.bu_p4 = FusionBlock(dim)
        self.bu_p5 = FusionBlock(dim)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        p3, p4, p5 = feats
        p5 = self.aifi(p5)  # global context on smallest map

        # top-down
        p4 = self.td_p4(p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.td_p3(p3, F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))
        # bottom-up
        n3 = p3
        n4 = self.bu_p4(p4, self.down3(n3))
        n5 = self.bu_p5(p5, self.down4(n4))
        return [n3, n4, n5]

    @property
    def last_attn(self):
        return self.aifi.last_attn
