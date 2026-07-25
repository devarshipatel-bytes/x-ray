"""OPIXray dataset loader (VOC-style boxes + occlusion-level test splits).

Expected layout (default OPIXray release; adjust in config if yours differs):

  data/OPIXray/
    train/
      train_image/*.jpg
      train_annotation/*.txt        # one .txt per image
    test/
      test_image/*.jpg
      test_annotation/*.txt
      test_occlusion/
        OL1.txt  OL2.txt  OL3.txt   # each = list of image stems for that occlusion level

Per-image annotation .txt lines are whitespace-separated, one object per line.
We accept both common OPIXray variants:
    <name> <class> <xmin> <ymin> <xmax> <ymax>
    <class> <xmin> <ymin> <xmax> <ymax>
Also falls back to Pascal-VOC .xml if ann_format: voc_xml is set in the config.

Returns per item: (image_tensor CxHxW, target dict) where target has
    boxes  : normalized cxcywh  (produced by transforms)
    labels : long class ids
    image_id, orig_size, size, ratio, pad
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OPIXrayDataset(Dataset):
    def __init__(self, cfg: Dict, split: str, transform, occlusion_level: Optional[str] = None):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        ds = cfg["dataset"]
        self.root = ds["root"]
        self.classes = list(ds["classes"])
        self.cls2id = {c: i for i, c in enumerate(self.classes)}
        # also accept case/space-insensitive lookups (annotation files vary)
        self._cls_norm = {self._norm(c): i for c, i in self.cls2id.items()}
        self.ann_format = ds.get("ann_format", "opixray_txt")

        img_dir, ann_dir = self._split_dirs(split)
        self.samples = self._index(img_dir, ann_dir)

        if occlusion_level is not None:
            keep = self._occlusion_stems(occlusion_level)
            self.samples = [s for s in self.samples if self._stem(s["image"]) in keep]
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{split}' under {self.root}. "
                f"Check dataset.root and layout (see scripts/download_opixray.md)."
            )

    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _norm(s: str) -> str:
        return s.lower().replace("-", "").replace("_", "").replace(" ", "")

    @staticmethod
    def _stem(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _split_dirs(self, split: str):
        # tolerant to a few common folder namings
        cand_img = [
            os.path.join(self.root, split, f"{split}_image"),
            os.path.join(self.root, split, "image"),
            os.path.join(self.root, split, "images"),
            os.path.join(self.root, split),
        ]
        cand_ann = [
            os.path.join(self.root, split, f"{split}_annotation"),
            os.path.join(self.root, split, "annotation"),
            os.path.join(self.root, split, "annotations"),
            os.path.join(self.root, split),
        ]
        img_dir = next((d for d in cand_img if os.path.isdir(d)), cand_img[0])
        ann_dir = next((d for d in cand_ann if os.path.isdir(d)), cand_ann[0])
        return img_dir, ann_dir

    def _find_image(self, img_dir: str, stem: str) -> Optional[str]:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            p = os.path.join(img_dir, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def _index(self, img_dir: str, ann_dir: str) -> List[Dict]:
        samples: List[Dict] = []
        ext = "*.xml" if self.ann_format == "voc_xml" else "*.txt"
        ann_files = sorted(glob.glob(os.path.join(ann_dir, ext)))
        for ann in ann_files:
            stem = self._stem(ann)
            img = self._find_image(img_dir, stem)
            if img is None:
                continue
            boxes, labels = self._parse_ann(ann)
            samples.append({"image": img, "boxes": boxes, "labels": labels, "id": stem})
        return samples

    def _parse_ann(self, path: str):
        if self.ann_format == "voc_xml":
            return self._parse_voc(path)
        return self._parse_txt(path)

    def _parse_txt(self, path: str):
        boxes, labels = [], []
        with open(path, "r") as f:
            for line in f:
                toks = line.strip().split()
                if not toks:
                    continue
                # locate the class token: first non-numeric-ish field
                # variant A: <name> <class> x1 y1 x2 y2   -> class at idx 1
                # variant B: <class> x1 y1 x2 y2          -> class at idx 0
                nums = toks[-4:]
                if len(toks) < 5:
                    continue
                cls_tok = toks[-5]
                cid = self._cls_norm.get(self._norm(cls_tok))
                if cid is None:
                    continue
                try:
                    x1, y1, x2, y2 = [float(v) for v in nums]
                except ValueError:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cid)
        return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)

    def _parse_voc(self, path: str):
        boxes, labels = [], []
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", "")
            cid = self._cls_norm.get(self._norm(name))
            if cid is None:
                continue
            bb = obj.find("bndbox")
            x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2]); labels.append(cid)
        return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(labels, dtype=np.int64)

    def _occlusion_stems(self, level: str):
        for cand in (
            os.path.join(self.root, "test", "test_occlusion", f"{level}.txt"),
            os.path.join(self.root, "test", f"{level}.txt"),
            os.path.join(self.root, f"{level}.txt"),
        ):
            if os.path.isfile(cand):
                with open(cand) as f:
                    return {self._stem(l.strip()) for l in f if l.strip()}
        raise FileNotFoundError(
            f"Occlusion list for {level} not found. Expected test/test_occlusion/{level}.txt"
        )

    # ---- Dataset API ---------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        target = {
            "boxes": s["boxes"].copy(),
            "labels": s["labels"].copy(),
            "image_id": s["id"],
        }
        img, target = self.transform(img, target)
        return img, target


def build_dataset(cfg: Dict, split: str, transform, occlusion_level: Optional[str] = None):
    return OPIXrayDataset(cfg, split, transform, occlusion_level)
