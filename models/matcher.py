"""Hungarian matcher (DETR-style) with focal classification cost."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal_alpha=0.25):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha

    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B, Q = outputs["pred_logits"].shape[:2]
        # fp32 throughout: under AMP the logits arrive as fp16, where sigmoid() of a
        # confidently-negative logit underflows to 0 and the 1e-8 floor below is itself
        # subnormal -> log(0) = -inf poisons the cost matrix and breaks the assignment.
        out_prob = outputs["pred_logits"].float().flatten(0, 1).sigmoid()   # [B*Q, C]
        out_bbox = outputs["pred_boxes"].float().flatten(0, 1)              # [B*Q, 4]

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets]).float()

        if tgt_ids.numel() == 0:
            return [(torch.as_tensor([], dtype=torch.int64),
                     torch.as_tensor([], dtype=torch.int64)) for _ in range(B)]

        # focal classification cost
        alpha, gamma = self.focal_alpha, 2.0
        neg = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos[:, tgt_ids] - neg[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
        C = C.view(B, Q, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, dim=-1)):
            if sizes[i] == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64),
                                torch.as_tensor([], dtype=torch.int64)))
                continue
            row, col = linear_sum_assignment(c[i])
            indices.append((torch.as_tensor(row, dtype=torch.int64),
                            torch.as_tensor(col, dtype=torch.int64)))
        return indices


def build_matcher(cfg: Dict) -> HungarianMatcher:
    m = cfg["matcher"]
    return HungarianMatcher(m["cost_class"], m["cost_bbox"], m["cost_giou"], m["focal_alpha"])
