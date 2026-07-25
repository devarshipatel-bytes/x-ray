"""Box format conversions and IoU (operates on normalized boxes)."""
from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def box_area(b):
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)


def box_iou(a, b):
    area_a = box_area(a)
    area_b = box_area(b)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(a, b):
    """GIoU between two sets of xyxy boxes. Returns [len(a), len(b)]."""
    iou, union = box_iou(a, b)
    lt = torch.min(a[:, None, :2], b[None, :, :2])
    rb = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[..., 0] * wh[..., 1]
    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)
