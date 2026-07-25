"""Detection overlays (clean boxes) on original or letterboxed-square images."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from PIL import Image

from models.box_ops import box_cxcywh_to_xyxy
import torch

from .palette import draw_detections


def draw_on_original(pil: Image.Image, result: Dict, classes: List[str]) -> Image.Image:
    return draw_detections(pil, result["boxes"], result["scores"], result["labels"], classes)


def square_boxes(caches: Dict, size: int, score_thresh=0.3):
    """Return (boxes_xyxy_square, scores, labels) for drawing on the letterboxed image."""
    logits = caches["pred_logits"][0].sigmoid()
    boxes = caches["pred_boxes"][0]
    scores, labels = logits.max(-1)
    keep = scores > score_thresh
    bx = (box_cxcywh_to_xyxy(boxes[keep]) * size).cpu().numpy()
    return bx, scores[keep].cpu().numpy(), labels[keep].cpu().numpy()


def draw_on_square(square_pil: Image.Image, caches: Dict, classes: List[str],
                   score_thresh=0.3) -> Image.Image:
    bx, s, l = square_boxes(caches, square_pil.size[0], score_thresh)
    return draw_detections(square_pil, bx, s, l, classes)
