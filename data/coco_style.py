"""Generic COCO-json detection loader (shared by PIDray / HiXray in M6).

PIDray and HiXray ship COCO-format annotations, so one loader covers both — you only swap
`dataset.root`, `dataset.ann_file`, and `dataset.classes` in the config. Returns the SAME
(image_tensor, target) contract as OPIXrayDataset, so the rest of the pipeline is unchanged.

This is provided for M6. It is straightforward but UNTESTED against the real archives until
you download them — verify with `scripts/sanity_data.py` before training.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoStyleDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, transform, occlusion_level: Optional[str] = None):
        ds = cfg["dataset"]
        self.transform = transform
        self.classes = list(ds["classes"])
        self.img_dir = os.path.join(ds["root"], ds.get(f"{split}_images", split))
        ann_file = ds.get(f"{split}_ann") or os.path.join(ds["root"], f"{split}.json")
        with open(ann_file) as f:
            coco = json.load(f)

        # map COCO category_id -> contiguous class id following config order when possible
        catid2name = {c["id"]: c["name"] for c in coco["categories"]}
        name2cid = {n: i for i, n in enumerate(self.classes)}
        self.catid2cid = {cid: name2cid.get(nm, None) for cid, nm in catid2name.items()}

        images = {im["id"]: im for im in coco["images"]}
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        self.samples = []
        for iid, im in images.items():
            boxes, labels = [], []
            for a in anns.get(iid, []):
                cid = self.catid2cid.get(a["category_id"])
                if cid is None:
                    continue
                x, y, w, h = a["bbox"]  # COCO xywh
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h]); labels.append(cid)
            self.samples.append({
                "image": os.path.join(self.img_dir, im["file_name"]),
                "boxes": np.array(boxes, np.float32).reshape(-1, 4),
                "labels": np.array(labels, np.int64),
                "id": os.path.splitext(im["file_name"])[0],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        target = {"boxes": s["boxes"].copy(), "labels": s["labels"].copy(), "image_id": s["id"]}
        return self.transform(img, target)


def build_dataset(cfg, split, transform, occlusion_level=None):
    return CocoStyleDataset(cfg, split, transform, occlusion_level)
