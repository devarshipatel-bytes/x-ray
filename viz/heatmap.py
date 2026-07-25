"""Heatmap helpers: colorize a [0,1] map and alpha-blend it over an image."""
from __future__ import annotations

import numpy as np
from PIL import Image
import matplotlib.cm as cm


def colorize(heat: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    heat = np.asarray(heat, dtype=np.float32)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    rgba = cm.get_cmap(cmap)(heat)
    return (rgba[..., :3] * 255).astype(np.uint8)


def overlay(pil: Image.Image, heat: np.ndarray, alpha: float = 0.5, cmap: str = "inferno") -> Image.Image:
    base = pil.convert("RGB")
    h = Image.fromarray(colorize(heat, cmap)).resize(base.size, Image.BILINEAR)
    return Image.blend(base, h, alpha)
