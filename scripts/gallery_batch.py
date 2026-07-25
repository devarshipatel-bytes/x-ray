"""Generate the 6-panel visualization gallery for N test images.

  python -m scripts.gallery_batch --config configs/xdetr_opixray.yaml \
      --weights runs/opixray_xdetr/last.pth --n 12 --out assets/galleries
"""
from __future__ import annotations

import argparse
import os

from PIL import Image

from engine.config import get_device, load_config
from viz.common import load_model
from viz.gallery import make_gallery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--out", default="assets/galleries")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    from data import build_dataset
    cfg = load_config(args.config, args.set)
    device = get_device()
    classes = cfg["dataset"]["classes"]
    model = load_model(cfg, args.weights, device)

    ds = build_dataset(cfg, split="test", train=False)
    os.makedirs(args.out, exist_ok=True)
    for i in range(min(args.n, len(ds))):
        path = ds.samples[i]["image"]
        pil = Image.open(path).convert("RGB")
        out_path = os.path.join(args.out, f"gallery_{ds.samples[i]['id']}.png")
        make_gallery(model, pil, cfg, device, out_path, classes, args.score)
        print(f"[gallery] {out_path}")


if __name__ == "__main__":
    main()
