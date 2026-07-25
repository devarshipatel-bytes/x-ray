"""Attention visualizations for X-DETR.

* decode_detections  - recover (query index, score, label, box) for confident detections
                       so we can tie each detection to its decoder query.
* cross_attention_map- per-object heatmap: where a detection's query attended in the image.
* encoder_saliency   - global AIFI self-attention saliency (P5), a class-agnostic "look here".

All maps are returned in the letterboxed square space (matches tensor_to_rgb / square boxes).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def decode_detections(caches: Dict, score_thresh=0.3, max_dets=100):
    logits = caches["pred_logits"][0].sigmoid()   # Q,C
    boxes = caches["pred_boxes"][0]               # Q,4 (cxcywh norm, square)
    scores, labels = logits.max(-1)
    keep = torch.where(scores > score_thresh)[0]
    order = scores[keep].argsort(descending=True)[:max_dets]
    q = keep[order]
    return [{"query": int(qi), "score": float(scores[qi]), "label": int(labels[qi]),
             "box": boxes[qi].cpu().numpy()} for qi in q]


def _split_levels(vec: torch.Tensor, level_shapes: List) -> List[np.ndarray]:
    maps, off = [], 0
    for (H, W) in level_shapes:
        n = H * W
        m = vec[off:off + n].reshape(H, W).float().cpu().numpy()
        maps.append(m)
        off += n
    return maps


def _upsample(m: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray((m * 255).astype(np.uint8) if m.max() <= 1 else m.astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)


def cross_attention_map(caches: Dict, query: int, size: int) -> np.ndarray:
    """Aggregate a query's cross-attention across feature levels -> HxW map in [0,1]."""
    attn = caches.get("dec_cross_attn")
    shapes = caches.get("level_shapes", [])
    if attn is None or not shapes:
        return np.zeros((size, size), np.float32)
    vec = attn[0, query]  # T
    maps = _split_levels(vec, shapes)
    acc = np.zeros((size, size), np.float32)
    for m in maps:
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        acc += _upsample(m, size) / 255.0
    acc /= max(len(maps), 1)
    return (acc - acc.min()) / (acc.max() - acc.min() + 1e-8)


def encoder_saliency(caches: Dict, size: int) -> np.ndarray:
    """Mean AIFI attention received per P5 location -> global saliency map in [0,1]."""
    attn = caches.get("enc_attn")
    shapes = caches.get("level_shapes", [])
    if attn is None or not shapes:
        return np.zeros((size, size), np.float32)
    # P5 is the last level; enc_attn is [B, HW_p5, HW_p5]
    H, W = shapes[-1]
    received = attn[0].mean(0)  # HW  (avg attention each key receives)
    m = received.reshape(H, W).float().cpu().numpy()
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return _upsample(m, size) / 255.0
