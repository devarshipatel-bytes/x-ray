"""One-shot multi-panel visualization for a single X-ray image.

Produces the headline figure: input | detections | Eigen-CAM | encoder saliency |
top-object cross-attention | operator attention map (ranked). Saves a PNG.
"""
from __future__ import annotations

from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import infer_single, tensor_to_rgb
from .detect_overlay import draw_on_original, draw_on_square
from .attention import decode_detections, cross_attention_map, encoder_saliency
from .gradcam import EigenCAM
from .operator_map import operator_attention_map
from .heatmap import overlay


def make_gallery(model, pil: Image.Image, cfg, device, out_path: str,
                 classes: List[str], score_thresh: float = 0.3):
    size = cfg["input"]["size"]
    result, caches, tensor, target = infer_single(model, pil, cfg, device, score_thresh)
    square = tensor_to_rgb(tensor, cfg)

    # panels
    det_orig = draw_on_original(pil, result, classes)
    # reuse a single EigenCAM (its forward hook must not be re-registered per image)
    cam_obj = getattr(model, "_eigencam", None)
    if cam_obj is None:
        cam_obj = EigenCAM(model)
        model._eigencam = cam_obj
    cam = cam_obj(tensor, size)
    cam_img = overlay(square, cam, alpha=0.55, cmap="inferno")
    enc = encoder_saliency(caches, size)
    enc_img = overlay(square, enc, alpha=0.55, cmap="viridis")

    dets = decode_detections(caches, score_thresh)
    if dets:
        cxa = cross_attention_map(caches, dets[0]["query"], size)
        cxa_img = overlay(square, cxa, alpha=0.6, cmap="magma")
        cxa_title = f"cross-attn: {classes[dets[0]['label']]} {dets[0]['score']:.2f}"
    else:
        cxa_img = square; cxa_title = "cross-attn: (no detection)"

    op_map, op_dets = operator_attention_map(caches, size, score_thresh=max(0.15, score_thresh - 0.1))
    op_img = overlay(square, op_map, alpha=0.55, cmap="inferno")

    panels = [
        (square, "input (letterboxed)"),
        (det_orig.resize(square.size), "detections"),
        (cam_img, "Eigen-CAM (backbone)"),
        (enc_img, "encoder saliency (AIFI)"),
        (cxa_img, cxa_title),
        (op_img, "operator attention (ranked)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (im, title) in zip(axes.ravel(), panels):
        ax.imshow(im); ax.set_title(title, fontsize=12); ax.axis("off")
    # annotate operator priorities on the last panel
    ax_op = axes.ravel()[5]
    for d in op_dets[:8]:
        x, y = d["center_px"]
        ax_op.text(x, y, str(d["priority"]), color="white", fontsize=13, fontweight="bold",
                   ha="center", va="center",
                   bbox=dict(boxstyle="circle", fc="#D55E00", ec="white"))
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return out_path
