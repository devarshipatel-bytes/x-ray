"""X-DETR: compact RT-DETR/RF-DETR-inspired detector for X-ray screening.

Pipeline:
  image (3 or 5 ch) -> ResNet backbone -> P3/P4/P5
    -> input_proj (1x1 -> hidden_dim) -> HybridEncoder (AIFI + CCFM)
    -> flatten to memory + sine/level pos + grid anchors
    -> query selection (top-k by encoder class score) -> initial content + reference boxes
    -> TransformerDecoder (self+cross attn, iterative box refinement)
    -> per-layer class + box heads
Outputs: {pred_logits [B,Q,C], pred_boxes [B,Q,4] cxcywh-normalized, aux_outputs[...]}.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from .backbone import build_backbone
from .box_ops import box_cxcywh_to_xyxy
from .decoder import TransformerDecoder, inverse_sigmoid
from .encoder import HybridEncoder
from .heads import build_bbox_head, build_class_head
from .position_encoding import PositionEmbeddingSine


class XDETR(nn.Module):
    def __init__(self, cfg: Dict, in_channels: int):
        super().__init__()
        m = cfg["model"]
        dim = m["hidden_dim"]
        self.num_classes = cfg["dataset"]["num_classes"]
        self.num_queries = m["num_queries"]
        self.num_levels = m["num_feature_levels"]
        self.aux_loss = m.get("aux_loss", True)

        self.backbone = build_backbone(cfg, in_channels)
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch, dim, 1), nn.GroupNorm(32, dim))
            for ch in self.backbone.num_channels
        ])
        self.encoder = HybridEncoder(dim, m["nheads"], m["ffn_dim"],
                                     enc_layers=m["enc_layers"], num_levels=self.num_levels,
                                     dropout=m.get("dropout", 0.0))
        self.decoder = TransformerDecoder(dim, m["nheads"], m["ffn_dim"],
                                          num_layers=m["dec_layers"], dropout=m.get("dropout", 0.0))
        self.pos_embed = PositionEmbeddingSine(dim // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.zeros(self.num_levels, dim))
        nn.init.normal_(self.level_embed)

        # query-selection (encoder) heads
        self.enc_class = build_class_head(dim, self.num_classes)
        self.enc_bbox = build_bbox_head(dim)
        # per-decoder-layer heads
        self.class_embed = nn.ModuleList([build_class_head(dim, self.num_classes)
                                          for _ in range(m["dec_layers"])])
        self.bbox_embed = nn.ModuleList([build_bbox_head(dim) for _ in range(m["dec_layers"])])
        self.decoder.bbox_embed = self.bbox_embed
        self.dim = dim

    # ----- memory construction -------------------------------------------------
    def _flatten(self, fused: List[torch.Tensor]):
        srcs, poss, anchors, shapes = [], [], [], []
        B = fused[0].shape[0]
        for lvl, f in enumerate(fused):
            _, C, H, W = f.shape
            shapes.append((H, W))
            pos = self.pos_embed(B, H, W, f.device, f.dtype).flatten(2).transpose(1, 2)  # B,HW,C
            pos = pos + self.level_embed[lvl].view(1, 1, -1)
            srcs.append(f.flatten(2).transpose(1, 2))  # B,HW,C
            poss.append(pos)
            # grid anchors (normalized cxcywh)
            gy, gx = torch.meshgrid(
                torch.arange(H, device=f.device, dtype=f.dtype),
                torch.arange(W, device=f.device, dtype=f.dtype), indexing="ij")
            cx = (gx + 0.5) / W
            cy = (gy + 0.5) / H
            wh = torch.full_like(cx, 0.05 * (2.0 ** lvl))
            a = torch.stack([cx, cy, wh, wh], dim=-1).view(1, H * W, 4).repeat(B, 1, 1)
            anchors.append(a)
        memory = torch.cat(srcs, dim=1)
        memory_pos = torch.cat(poss, dim=1)
        anchors = torch.cat(anchors, dim=1).clamp(1e-4, 1 - 1e-4)
        return memory, memory_pos, anchors, shapes

    def forward(self, x: torch.Tensor) -> Dict:
        feats = self.backbone(x)
        srcs = [self.input_proj[i](feats[k]) for i, k in enumerate(["p3", "p4", "p5"])]
        fused = self.encoder(srcs)
        memory, memory_pos, anchors, shapes = self._flatten(fused)
        self.decoder.level_shapes = shapes  # for viz

        enc_logits = self.enc_class(memory)                                    # B,T,C
        enc_boxes = (self.enc_bbox(memory) + inverse_sigmoid(anchors)).sigmoid()  # B,T,4

        # top-k query selection by max class score
        topk = min(self.num_queries, memory.shape[1])
        scores = enc_logits.max(-1).values                                     # B,T
        topk_idx = scores.topk(topk, dim=1).indices                            # B,Q
        idx = topk_idx.unsqueeze(-1)
        ref = enc_boxes.gather(1, idx.repeat(1, 1, 4)).detach()                 # B,Q,4
        tgt = memory.gather(1, idx.repeat(1, 1, self.dim)).detach()            # B,Q,C

        hs, refs = self.decoder(tgt, ref, memory, memory_pos)                   # [L,B,Q,C], [L,B,Q,4]
        out_class = torch.stack([self.class_embed[l](hs[l]) for l in range(hs.shape[0])])
        out_coord = refs                                                       # already sigmoid

        out = {"pred_logits": out_class[-1], "pred_boxes": out_coord[-1]}
        if self.aux_loss:
            out["aux_outputs"] = [{"pred_logits": out_class[i], "pred_boxes": out_coord[i]}
                                  for i in range(hs.shape[0] - 1)]
        return out


@torch.no_grad()
def postprocess(out: Dict, targets: List[Dict], score_thresh: float = 0.3,
                max_dets: int = 100) -> List[Dict]:
    """Map normalized cxcywh preds back to ORIGINAL image xyxy pixel coords.

    Returns per-image dict: {boxes [N,4] xyxy, scores [N], labels [N]} sorted by score.
    """
    logits = out["pred_logits"].sigmoid()          # B,Q,C
    boxes = box_cxcywh_to_xyxy(out["pred_boxes"])   # B,Q,4 normalized to square
    results = []
    B = logits.shape[0]
    for b in range(B):
        scores, labels = logits[b].max(-1)
        keep = scores > score_thresh
        s, l = scores[keep], labels[keep]
        bx = boxes[b][keep]
        # sort + cap
        order = s.argsort(descending=True)[:max_dets]
        s, l, bx = s[order], l[order], bx[order]

        t = targets[b]
        size = int(t["size"][0]);  ratio = float(t["ratio"])
        pad_x, pad_y = int(t["pad"][0]), int(t["pad"][1])
        oh, ow = int(t["orig_size"][0]), int(t["orig_size"][1])
        # normalized-square -> square pixels -> remove pad -> undo resize
        bx = bx * size
        bx[:, [0, 2]] = (bx[:, [0, 2]] - pad_x) / ratio
        bx[:, [1, 3]] = (bx[:, [1, 3]] - pad_y) / ratio
        bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, ow)
        bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, oh)
        results.append({"boxes": bx.cpu(), "scores": s.cpu(), "labels": l.cpu(),
                        "image_id": t.get("image_id")})
    return results


def build_model(cfg: Dict, in_channels: int) -> XDETR:
    return XDETR(cfg, in_channels)
