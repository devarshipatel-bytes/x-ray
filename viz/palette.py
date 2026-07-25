"""Consistent, colorblind-friendly class palette + drawing helpers."""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Okabe-Ito colorblind-safe palette (extended); indexed by class id.
_PALETTE = [
    (230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
    (0, 114, 178), (213, 94, 0), (204, 121, 167), (153, 153, 153),
    (255, 105, 97), (119, 221, 119), (170, 170, 255), (255, 179, 71),
]


def class_color(cid: int):
    return _PALETTE[cid % len(_PALETTE)]


def _font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(pil: Image.Image, boxes, scores, labels, classes: List[str],
                    thickness=3) -> Image.Image:
    """boxes: Nx4 xyxy (original pixels). Returns a copy with boxes + labels drawn."""
    img = pil.convert("RGB").copy()
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(max(12, img.size[0] // 45))
    for box, s, l in zip(np.asarray(boxes), np.asarray(scores), np.asarray(labels)):
        x1, y1, x2, y2 = [float(v) for v in box]
        col = class_color(int(l))
        for t in range(thickness):
            d.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=col)
        name = classes[int(l)] if int(l) < len(classes) else str(int(l))
        tag = f"{name} {s:.2f}"
        tw = d.textlength(tag, font=font)
        d.rectangle([x1, y1 - font.size - 4, x1 + tw + 6, y1], fill=col + (230,))
        d.text((x1 + 3, y1 - font.size - 3), tag, fill=(0, 0, 0), font=font)
    return img
