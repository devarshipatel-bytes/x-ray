"""Set-prediction criterion: focal classification + L1 + GIoU, with aux losses."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sigmoid_focal_loss(logits, targets, num_boxes, alpha=0.25, gamma=2.0):
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / max(num_boxes, 1)


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, cfg):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        loss_cfg = cfg["loss"]
        self.alpha = loss_cfg["focal_alpha"]
        self.gamma = loss_cfg["focal_gamma"]
        self.weights = {
            "loss_ce": loss_cfg["cls_loss_coef"],
            "loss_bbox": loss_cfg["bbox_loss_coef"],
            "loss_giou": loss_cfg["giou_loss_coef"],
        }

    @staticmethod
    def _src_perm(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        logits = outputs["pred_logits"]  # [B, Q, C]
        idx = self._src_perm(indices)
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target = torch.zeros_like(logits)
        if target_classes_o.numel():
            target[idx[0], idx[1], target_classes_o] = 1.0
        loss_ce = sigmoid_focal_loss(logits, target, num_boxes, self.alpha, self.gamma) * logits.shape[1]
        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._src_perm(indices)
        src_boxes = outputs["pred_boxes"][idx[0], idx[1]]
        tgt_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)
        if src_boxes.numel() == 0:
            z = src_boxes.sum() * 0.0
            return {"loss_bbox": z, "loss_giou": z}
        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="none").sum() / num_boxes
        giou = torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                              box_cxcywh_to_xyxy(tgt_boxes)))
        loss_giou = (1 - giou).sum() / num_boxes
        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def _compute(self, outputs, targets, indices, num_boxes):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)
        device = outputs["pred_logits"].device
        num_boxes_t = torch.as_tensor(num_boxes, dtype=torch.float, device=device)

        outputs_main = {k: v for k, v in outputs.items() if k not in ("aux_outputs", "enc_outputs")}
        indices = self.matcher(outputs_main, targets)
        losses = self._compute(outputs_main, targets, indices, num_boxes_t)

        # auxiliary losses on every intermediate decoder layer
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(aux, targets)
            for k, v in self._compute(aux, targets, aux_indices, num_boxes_t).items():
                losses[f"{k}_aux{i}"] = v

        # weighted total
        total = torch.zeros((), device=device)
        for k, v in losses.items():
            base = k.split("_aux")[0]
            total = total + self.weights[base] * v
        losses["loss"] = total
        return losses


def build_criterion(cfg: Dict, matcher) -> SetCriterion:
    return SetCriterion(cfg["dataset"]["num_classes"], matcher, cfg)
