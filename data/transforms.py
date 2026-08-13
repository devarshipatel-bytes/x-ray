"""Image + box transforms for X-DETR.

Design notes specific to X-ray pseudo-color:
  * We letterbox to a fixed square so images batch cleanly and boxes map back exactly.
  * Photometric jitter is MILD and hue is left untouched — color encodes material.
  * Material-proxy channels are computed on the final [0,1] RGB (post-jitter, post-resize)
    and concatenated after ImageNet-normalizing the RGB, giving a 5-channel tensor.

A transform is a callable(pil_img, target_dict) -> (tensor CxHxW, target_dict).
Boxes in the incoming target are absolute xyxy in the ORIGINAL image pixel space.
Outgoing target['boxes'] are normalized cxcywh in [0,1] w.r.t. the letterboxed square.
target also carries orig_size, size, ratio, pad for inverse mapping at eval time.

Everything here is a module-level class, never a closure: on Windows (and anywhere using
the 'spawn' start method) DataLoader workers pickle the dataset, and a local function
inside a factory cannot be pickled.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .material_proxy import rgb_to_material


def _letterbox(img: Image.Image, size: int, fill=(0, 0, 0),
               photometric=None) -> Tuple[Image.Image, float, Tuple[int, int]]:
    w, h = img.size
    ratio = min(size / w, size / h)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    img_r = img.resize((nw, nh), Image.BILINEAR)
    # Jitter the DOWNSCALED image: brightness/contrast are per-pixel, so doing them
    # after the resize is equivalent but ~6x cheaper (OPIXray is 1225x954 -> 512).
    # Applied before the paste so the letterbox padding stays exactly `fill`.
    if photometric is not None:
        img_r = photometric(img_r)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img_r, (pad_x, pad_y))
    return canvas, ratio, (pad_x, pad_y)


class Compose:
    def __init__(self, fns: List[Callable]):
        self.fns = fns

    def __call__(self, img, target):
        for fn in self.fns:
            img, target = fn(img, target)
        return img, target


class PhotometricJitter:
    """Mild brightness/contrast jitter. Callable(img) -> img; hue is deliberately untouched."""

    def __init__(self, brightness: float, contrast: float):
        self.brightness = float(brightness)
        self.contrast = float(contrast)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.brightness > 0:
            f = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = ImageEnhance.Brightness(img).enhance(f)
        if self.contrast > 0:
            f = 1.0 + random.uniform(-self.contrast, self.contrast)
            img = ImageEnhance.Contrast(img).enhance(f)
        return img


class RandomHorizontalFlip:
    """Flip image and boxes together. Boxes are absolute xyxy at this stage."""

    def __init__(self, p: float):
        self.p = float(p)

    def __call__(self, img, target):
        if self.p > 0 and random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.size[0]
            b = target["boxes"]
            if len(b):
                b = b.copy()
                b[:, [0, 2]] = w - b[:, [2, 0]]  # flip xyxy about image width
                target["boxes"] = b
        return img, target


class LetterboxToTensor:
    """Letterbox to a square, build the 3- or 5-channel tensor, normalize boxes to cxcywh."""

    def __init__(self, size: int, use_material: bool, mean: torch.Tensor, std: torch.Tensor,
                 jitter: Optional[PhotometricJitter] = None):
        self.size = int(size)
        self.use_material = bool(use_material)
        self.mean = mean
        self.std = std
        self.jitter = jitter

    def __call__(self, img, target):
        size = self.size
        w0, h0 = img.size
        target["orig_size"] = torch.tensor([h0, w0])
        img, ratio, (pad_x, pad_y) = _letterbox(img, size, photometric=self.jitter)

        # map boxes: original xyxy -> letterboxed xyxy
        boxes = target["boxes"]
        if len(boxes):
            boxes = boxes.astype(np.float32).copy()
            boxes *= ratio
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            boxes[:, 0::2] = boxes[:, 0::2].clip(0, size)
            boxes[:, 1::2] = boxes[:, 1::2].clip(0, size)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)

        rgb01 = np.asarray(img, dtype=np.float32) / 255.0            # HxWx3 in [0,1]
        chans = torch.from_numpy(rgb01).permute(2, 0, 1)             # 3xHxW
        chans = (chans - self.mean) / self.std
        if self.use_material:
            mat = rgb_to_material(rgb01)                              # HxWx2
            mat = torch.from_numpy(mat).permute(2, 0, 1)             # 2xHxW
            chans = torch.cat([chans, mat], dim=0)                    # 5xHxW

        # boxes -> normalized cxcywh
        if len(boxes):
            cx = (boxes[:, 0] + boxes[:, 2]) / 2 / size
            cy = (boxes[:, 1] + boxes[:, 3]) / 2 / size
            bw = (boxes[:, 2] - boxes[:, 0]) / size
            bh = (boxes[:, 3] - boxes[:, 1]) / size
            norm = np.stack([cx, cy, bw, bh], axis=1)
            # drop degenerate boxes
            keep = (norm[:, 2] > 1e-3) & (norm[:, 3] > 1e-3)
            norm = norm[keep]
            target["labels"] = target["labels"][keep]
        else:
            norm = np.zeros((0, 4), dtype=np.float32)

        target["boxes"] = torch.from_numpy(norm.astype(np.float32))
        target["labels"] = torch.as_tensor(target["labels"], dtype=torch.long)
        target["size"] = torch.tensor([size, size])
        target["ratio"] = torch.tensor(ratio)
        target["pad"] = torch.tensor([pad_x, pad_y])
        return chans, target


def _build_transform(cfg: Dict, train: bool) -> Compose:
    size = int(cfg["input"]["size"])
    use_material = bool(cfg["input"]["use_material_proxy"])
    mean = torch.tensor(cfg["input"]["pixel_mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["input"]["pixel_std"]).view(3, 1, 1)
    aug = cfg.get("augment", {})
    p_flip = float(aug.get("hflip", 0.0)) if train else 0.0
    bri = float(aug.get("brightness", 0.0)) if train else 0.0
    con = float(aug.get("contrast", 0.0)) if train else 0.0

    jitter = PhotometricJitter(bri, con) if (bri > 0 or con > 0) else None
    letterbox = LetterboxToTensor(size, use_material, mean, std, jitter)
    fns = [RandomHorizontalFlip(p_flip), letterbox] if train else [letterbox]
    return Compose(fns)


def build_train_transform(cfg: Dict) -> Compose:
    return _build_transform(cfg, train=True)


def build_eval_transform(cfg: Dict) -> Compose:
    return _build_transform(cfg, train=False)


def in_channels(cfg: Dict) -> int:
    return 5 if cfg["input"]["use_material_proxy"] else 3


def collate_fn(batch):
    """Stack images (all same square size); keep targets as a list (variable #boxes)."""
    imgs = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return imgs, targets
