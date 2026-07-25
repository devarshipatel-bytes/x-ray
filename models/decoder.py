"""Transformer decoder with reference points + iterative box refinement.

Standard multi-head attention (no custom deformable CUDA op) so it runs anywhere,
including Colab Free. Cross-attention weights of the last layer are cached on
`self.last_cross_attn` (+ level shapes) so the viz suite can render per-object heatmaps.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers):
        super().__init__()
        h = [dim_hidden] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip([dim_in] + h, h + [dim_out]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def gen_sineembed_for_position(pos_tensor, dim=128, temperature=10000):
    """pos_tensor [B, Q, 4] (cxcywh in [0,1]) -> [B, Q, 4*dim] sine embedding."""
    scale = 2 * math.pi
    dim_t = torch.arange(dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / dim)
    out = []
    for i in range(4):
        e = pos_tensor[:, :, i:i + 1] * scale / dim_t
        e = torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=3).flatten(2)
        out.append(e)
    return torch.cat(out, dim=2)  # B, Q, 4*dim


class DecoderLayer(nn.Module):
    def __init__(self, dim, nheads, ffn_dim, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ln3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, memory, memory_pos, need_attn=False):
        q = k = tgt + query_pos
        sa, _ = self.self_attn(q, k, value=tgt, need_weights=False)
        tgt = self.ln1(tgt + self.drop(sa))

        ca, attn = self.cross_attn(query=tgt + query_pos, key=memory + memory_pos,
                                   value=memory, need_weights=need_attn,
                                   average_attn_weights=True)
        tgt = self.ln2(tgt + self.drop(ca))
        tgt = self.ln3(tgt + self.drop(self.ffn(tgt)))
        return tgt, attn


class TransformerDecoder(nn.Module):
    def __init__(self, dim, nheads, ffn_dim, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(dim, nheads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.num_layers = num_layers
        self.ref_point_head = MLP(4 * (dim // 2), dim, dim, 2)
        self.bbox_embed: Optional[nn.ModuleList] = None  # set by XDETR (per-layer refinement)
        self.dim = dim
        self.last_cross_attn = None   # [B, Q, num_tokens] from final layer
        self.level_shapes: List = []  # [(H,W), ...] to reshape attn per level

    def forward(self, tgt, reference_points, memory, memory_pos):
        """tgt [B,Q,C]; reference_points [B,Q,4] in [0,1]; memory [B,T,C]."""
        output = tgt
        inter_refs, inter_out = [], []
        ref = reference_points
        for lid, layer in enumerate(self.layers):
            query_sine = gen_sineembed_for_position(ref, dim=self.dim // 2)
            query_pos = self.ref_point_head(query_sine)
            need = lid == self.num_layers - 1
            output, attn = layer(output, query_pos, memory, memory_pos, need_attn=need)
            if need and attn is not None:
                self.last_cross_attn = attn.detach()
            # iterative box refinement
            if self.bbox_embed is not None:
                delta = self.bbox_embed[lid](output)
                new_ref = (delta + inverse_sigmoid(ref)).sigmoid()
                ref = new_ref.detach()
                inter_refs.append(new_ref)
            else:
                inter_refs.append(ref)
            inter_out.append(output)
        return torch.stack(inter_out), torch.stack(inter_refs)  # [L,B,Q,C], [L,B,Q,4]
