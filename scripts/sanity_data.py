"""M0 data sanity: render a few annotated samples + their material-proxy channels.

  python -m scripts.sanity_data --config configs/xdetr_opixray.yaml --n 6 --out assets/sanity.png

Confirms the loader, box mapping, letterboxing, and material-proxy channels are correct
BEFORE any training.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import build_dataset, in_channels
from data.material_proxy import material_colormap
from engine.config import load_config
from models.box_ops import box_cxcywh_to_xyxy
from viz.common import tensor_to_rgb
from viz.palette import draw_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="assets/sanity.png")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    classes = cfg["dataset"]["classes"]
    use_mat = cfg["input"]["use_material_proxy"]
    ds = build_dataset(cfg, split="train", train=True)
    print(f"[sanity] {len(ds)} images, {in_channels(cfg)}-channel input")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n = min(args.n, len(ds))
    cols = 2 if use_mat else 1
    fig, axes = plt.subplots(n, cols, figsize=(6 * cols, 5 * n))
    axes = np.atleast_2d(axes)
    size = cfg["input"]["size"]
    for i in range(n):
        img, t = ds[i]
        rgb = tensor_to_rgb(img, cfg)
        boxes = (box_cxcywh_to_xyxy(t["boxes"]) * size).numpy()
        drawn = draw_detections(rgb, boxes, np.ones(len(boxes)), t["labels"].numpy(), classes)
        axes[i, 0].imshow(drawn); axes[i, 0].set_title(f"{t['image_id']} ({len(boxes)} boxes)")
        axes[i, 0].axis("off")
        if use_mat:
            mat = img[3:5].permute(1, 2, 0).numpy()
            axes[i, 1].imshow(material_colormap(mat))
            axes[i, 1].set_title("material proxy (organic=warm, metal=cool)")
            axes[i, 1].axis("off")
    fig.tight_layout(); fig.savefig(args.out, dpi=120)
    print(f"[sanity] wrote {args.out}")


if __name__ == "__main__":
    main()
