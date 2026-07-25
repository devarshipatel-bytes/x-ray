"""Shared single-image inference for the visualization suite / demo.

Loads the model, letterboxes an arbitrary PIL image, runs a forward pass, and returns
predictions in ORIGINAL coordinates plus the cached attention tensors for heatmaps.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from data.transforms import build_eval_transform, in_channels
from models import build_model, postprocess
from engine.checkpoint import load_checkpoint


def load_model(cfg: Dict, weights: Optional[str], device):
    model = build_model(cfg, in_channels(cfg)).to(device)
    if weights:
        load_checkpoint(weights, model, map_location=device)
    model.eval()
    return model


def preprocess(pil: Image.Image, cfg: Dict):
    tfm = build_eval_transform(cfg)
    target = {"boxes": np.zeros((0, 4), np.float32), "labels": np.zeros((0,), np.int64),
              "image_id": "query"}
    tensor, target = tfm(pil.convert("RGB"), target)
    return tensor, target


def tensor_to_rgb(tensor: torch.Tensor, cfg: Dict) -> Image.Image:
    """Denormalize the RGB channels of a CxHxW input tensor back to a viewable PIL image
    (the letterboxed square that heatmaps live in)."""
    mean = torch.tensor(cfg["input"]["pixel_mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["input"]["pixel_std"]).view(3, 1, 1)
    rgb = tensor[:3].cpu() * std + mean
    rgb = (rgb.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(rgb)


@torch.no_grad()
def infer_single(model, pil: Image.Image, cfg: Dict, device, score_thresh: float = 0.3):
    tensor, target = preprocess(pil, cfg)
    out = model(tensor.unsqueeze(0).to(device))
    result = postprocess(out, [target], score_thresh=score_thresh,
                         max_dets=cfg["eval"]["max_dets"])[0]
    caches = {
        "enc_attn": getattr(model.encoder, "last_attn", None),
        "dec_cross_attn": getattr(model.decoder, "last_cross_attn", None),
        "level_shapes": getattr(model.decoder, "level_shapes", []),
        "pred_logits": out["pred_logits"].detach(),
        "pred_boxes": out["pred_boxes"].detach(),
    }
    return result, caches, tensor, target
